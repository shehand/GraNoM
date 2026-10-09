"""/!\
 * Copyright (c) Shehan Edirimannage, 2025
 *
 * This file is part of the GraNoM project.
 *
 * Licensed for evaluation and personal testing purposes only.
 * Redistribution, modification, or commercial use of this file,
 * in whole or in part, is strictly prohibited without explicit
 * written permission from the copyright holder.
 *
 * For license inquiries, contact developers.
 */
pf: A Flower / PyTorch app.
"""

import json
import time
from copy import deepcopy

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import (
    Context,
    ParametersRecord,
    RecordSet,
    array_from_numpy,
    MetricsRecord,
)

from pf.task import (
    apply_privacy,
    build_model_for_dataset,
    get_weights,
    load_data,
    set_weights,
    set_seed,
    test,
    train,
)


class FlowerClient(NumPyClient):
    """Client implementing GraNoM's Partial Model Memorization and the
    configured perturbation method. All behavior is driven by `priv_cfg`."""

    def __init__(self, net, client_state: RecordSet, trainloader, valloader,
                 local_epochs, priv_cfg: dict):
        self.net = net
        self.client_state = client_state
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.priv_cfg = priv_cfg
        self.memorization = bool(priv_cfg.get("memorization", True))
        self.grad_threshold = float(priv_cfg.get("grad_threshold", 0.0))
        self.grad_clip_norm = float(priv_cfg.get("grad_clip_norm", 0.0))
        self.granom_update_clip = bool(priv_cfg.get("granom_update_clip", False))
        self.attack_a3 = bool(priv_cfg.get("attack_a3", False))
        self.a3_prev_set_name = "a3-prev-sent-params"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.ignored_parameter_set_name = "model-ignored-parameters"
        self.eval_confidence_score_name = "eval-conf-score"
        self.train_confidence_score_name = "train-conf-score"

    def fit(self, parameters, config):
        """Train locally, apply the configured perturbation, return update."""
        # Received global weights (needed as reference for Fed-SMP's update delta)
        prev_params = deepcopy(parameters)

        set_weights(self.net, parameters)

        # Partial Model Memorization: restore stationary coords from last round
        if self.memorization:
            self._load_layer_weights_from_state()

        # ---- IoT instrumentation: wall-clock + peak GPU memory ----
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.perf_counter()

        t_net, train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            lr=float(config["lr"]),
            device=self.device,
            grad_clip_norm=self.grad_clip_norm,
        )

        # Save stationary coords for next round (uses the same tau threshold)
        if self.memorization:
            self._save_layer_weights_to_state(t_net)

        # Apply the configured perturbation method
        self.net = apply_privacy(
            self.net,
            noise_mode=self.priv_cfg.get("noise_mode", "granom"),
            prev_params=prev_params,
            epsilon=float(self.priv_cfg.get("epsilon", 10.0)),
            epsilon_prime=float(self.priv_cfg.get("epsilon_prime", 100.0)),
            clip_threshold=float(self.priv_cfg.get("clip_threshold", 1.0)),
            grad_threshold=self.grad_threshold,
            fedsmp_ratio=float(self.priv_cfg.get("fedsmp_ratio", 0.1)),
            fedsmp_sigma=float(self.priv_cfg.get("fedsmp_sigma", 1.0)),
            granom_update_clip=self.granom_update_clip,
        )

        fit_time_s = time.perf_counter() - t0
        peak_mem_mb = 0.0
        if self.device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        weights = get_weights(self.net)
        num_params = int(sum(w.size for w in weights))

        # A3 index-inference attack (optional): measured from the perturbed params
        # we are about to send, before grads are cleared.
        a3_auc = self._a3_index_inference() if self.attack_a3 else -1.0

        # Fraction of coordinates classified stationary (|grad| <= tau) this round.
        # Cheap; always logged so we can report %-stationary across training (R2.5/R3.2).
        stationary_frac = self._stationary_fraction()

        return (
            weights,
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "fit_time_s": float(fit_time_s),
                "peak_mem_mb": float(peak_mem_mb),
                "num_params": num_params,
                "a3_auc": float(a3_auc),
                "stationary_frac": float(stationary_frac),
            },
        )

    def _stationary_fraction(self):
        """Fraction of trainable coordinates with |grad| <= tau this round (the
        'stationary' set). Reported per round so we can show how many parameters
        are frozen and how it evolves across training and datasets."""
        tau = self.grad_threshold
        stat = tot = 0
        for _, p in self.net.named_parameters():
            if p.grad is None:
                continue
            stat += int((p.grad.detach().abs() <= tau).sum().item())
            tot += p.grad.numel()
        return stat / tot if tot else 0.0

    @staticmethod
    def _auc(score, label):
        """AUC of `score` for the positive (trainable) class via rank statistic.
        0.5 = attacker cannot separate the two groups (mask concealed); far from
        0.5 = groups distinguishable. Returns -1 if a class is absent."""
        pos = label == 1
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            return -1.0
        order = np.argsort(score, kind="mergesort")
        ranks = np.empty(len(score), dtype=np.float64)
        ranks[order] = np.arange(1, len(score) + 1)
        return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    def _a3_index_inference(self):
        """A3 / masking attack: can an adversary label each coordinate trainable
        (noise scale b) vs stationary (b') from the SENT updates across rounds?
        Attacker score = per-coord inter-round change |sent_t - sent_{t-1}|
        (small for stationary: memorized/frozen + tiny b' noise; large for
        trainable: training change + large b noise). Scored (AUC) against the true
        stationary mask (|grad| <= tau). AUC ~ 0.5 => concealed (strong defense);
        AUC far from 0.5 => the two groups are distinguishable. Needs 2 rounds
        (returns -1 on the first)."""
        tau = self.grad_threshold
        names, cur, is_train = [], [], []
        for name, p in self.net.named_parameters():
            if p.grad is None:
                continue
            names.append(name)
            cur.append(p.data.detach().cpu().numpy().ravel())
            is_train.append((p.grad.detach().abs() > tau).cpu().numpy().ravel())
        if not cur:
            return -1.0
        cur_flat = np.concatenate(cur).astype(np.float64)
        label = np.concatenate(is_train).astype(np.int8)  # 1 = trainable, 0 = stationary

        prev_rec = self.client_state.parameters_records.get(self.a3_prev_set_name, None)
        # Stash the current sent params for next round's diff.
        self.client_state.parameters_records[self.a3_prev_set_name] = ParametersRecord(
            {n: array_from_numpy(c.astype(np.float32)) for n, c in zip(names, cur)}
        )
        if prev_rec is None:
            return -1.0
        try:
            prev_flat = np.concatenate([prev_rec[n].numpy().ravel() for n in names]).astype(np.float64)
        except Exception:
            return -1.0
        if prev_flat.shape != cur_flat.shape:
            return -1.0
        score = np.abs(cur_flat - prev_flat)
        return self._auc(score, label)

    def _save_layer_weights_to_state(self, t_net):
        """Save stationary (|grad| <= tau) parameters to client state."""
        tau = self.grad_threshold
        ignored_parameter_index_map = {}
        for name, param in t_net.named_parameters():
            if param.requires_grad and param.grad is not None:
                zero_grad_mask = (param.grad.abs() <= tau)
                if zero_grad_mask.any():
                    ignored_parameter_index_map[name] = (
                        param.data.clone().detach() * zero_grad_mask
                    )

        ignored_parameter_index_set = {}
        for k, v in ignored_parameter_index_map.items():
            ignored_parameter_index_set[k] = array_from_numpy(v.cpu().numpy())

        self.client_state.parameters_records[self.ignored_parameter_set_name] = (
            ParametersRecord(ignored_parameter_index_set)
        )

    def _load_layer_weights_from_state(self):
        """Restore previously-saved stationary parameters."""
        if self.ignored_parameter_set_name not in self.client_state.parameters_records:
            return
        param_records = self.client_state.parameters_records[self.ignored_parameter_set_name]
        for name, param in self.net.named_parameters():
            if name in param_records:
                numpy_array = param_records[name].numpy()
                restored_tensor = torch.from_numpy(numpy_array).to(param.device)
                frozen_mask = (restored_tensor != 0)
                param.data[frozen_mask] = restored_tensor[frozen_mask]

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        if self.memorization:
            self._load_layer_weights_from_state()
        loss, accuracy = test(self.net, self.valloader, self.device)

        self.client_state.metrics_records[self.train_confidence_score_name] = MetricsRecord(
            {"train-record": self._get_confidence_scores(self.trainloader)}
        )
        self.client_state.metrics_records[self.eval_confidence_score_name] = MetricsRecord(
            {"test-record": self._get_confidence_scores(self.valloader)}
        )
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

    def _get_confidence_scores(self, loader):
        confidence_scores = []
        self.net.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                outputs = self.net(images)
                softmax_scores = torch.softmax(outputs, dim=1)
                max_scores, _ = torch.max(softmax_scores, dim=1)
                confidence_scores.extend(max_scores.tolist())
        return confidence_scores

    def get_properties(self, config):
        train_json = json.dumps(
            self.client_state.metrics_records[self.train_confidence_score_name]["train-record"]
        )
        test_json = json.dumps(
            self.client_state.metrics_records[self.eval_confidence_score_name]["test-record"]
        )
        return {"train_confidence_scores": train_json, "test_confidence_scores": test_json}


def _priv_cfg_from_run(run_config: dict) -> dict:
    """Extract perturbation knobs from run-config with defaults."""
    return {
        "noise_mode": str(run_config.get("noise-mode", "granom")),
        "memorization": bool(run_config.get("memorization", True)),
        "epsilon": float(run_config.get("epsilon", 10.0)),
        "epsilon_prime": float(run_config.get("epsilon-prime", 100.0)),
        "clip_threshold": float(run_config.get("clip-threshold", 1.0)),
        "grad_threshold": float(run_config.get("grad-threshold", 0.0)),
        "grad_clip_norm": float(run_config.get("grad-clip-norm", 0.0)),
        "granom_update_clip": bool(run_config.get("granom-update-clip", False)),
        "attack_a3": bool(run_config.get("attack-a3", False)),
        "fedsmp_ratio": float(run_config.get("fedsmp-ratio", 0.1)),
        "fedsmp_sigma": float(run_config.get("fedsmp-sigma", 1.0)),
    }


def client_fn(context: Context):
    run_config = context.run_config
    dataset = str(run_config.get("dataset", "mnist"))
    model_override = str(run_config.get("model", ""))
    seed = int(run_config.get("seed", 42))
    set_seed(seed)

    net = build_model_for_dataset(dataset, model_override)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions, dataset=dataset, seed=seed)
    local_epochs = run_config["local-epochs"]

    priv_cfg = _priv_cfg_from_run(run_config)
    client_state = context.state
    return FlowerClient(
        net, client_state, trainloader, valloader, local_epochs, priv_cfg
    ).to_client()


app = ClientApp(client_fn)
