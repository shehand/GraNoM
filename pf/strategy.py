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
from logging import INFO

import torch
import torch.utils
import torch.utils.data
import wandb
import numpy as np
from pf.task import (
    build_model_for_dataset,
    create_run_dir,
    dataset_spec,
    membership_inference,
    set_weights,
)
from pf.model_inversion_attack import ModelInversionAttack

from flwr.common import logger, parameters_to_ndarrays, GetPropertiesIns
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg

from copy import deepcopy

PROJECT_NAME = "pf"


class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy: (1) saves results to the filesystem, (2) saves a
    checkpoint of the global  model when a new best is found, (3) logs
    results to W&B if enabled.
    """

    def __init__(self, run_config: UserConfig, use_wandb: bool,
                 dataset: str = "mnist", model_override: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.use_wandb = use_wandb

        # Dataset-specific model + attack input shape
        self.dataset = dataset
        self.model_override = model_override
        spec = dataset_spec(dataset)
        self.image_size = (spec["channels"], spec["crop"], spec["crop"])
        self.norm_mean, self.norm_std = spec["norm"]  # for MoIA input normalization

        # Attack-cost controls
        self.moia_iterations = int(run_config.get("moia-iterations", 1000))
        self.attack_every = max(1, int(run_config.get("attack-every", 1)))

        # Initialise W&B if set
        if use_wandb:
            self._init_wandb_project()

        # Keep track of best acc
        self.best_acc_so_far = 0.0

        # A dictionary to store results as they come
        self.results = {}

        # Initialize model inversion attack device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _new_model(self):
        return build_model_for_dataset(self.dataset, self.model_override)

    def _init_wandb_project(self):
        # init W&B
        wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_acc(self, round, accuracy, parameters):
        """Determines if a new best global model has been found.

        If so, the model checkpoint is saved to disk.
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            # You could save the parameters object directly.
            # Instead we are going to apply them to a PyTorch
            # model and save the state dict.
            # Converts flwr.common.Parameters to ndarrays
            ndarrays = parameters_to_ndarrays(parameters)
            model = self._new_model()
            set_weights(model, ndarrays)
            # Save the PyTorch model
            file_name = f"model_state_acc_{accuracy}_round_{round}.pth"
            torch.save(model.state_dict(), self.save_path / file_name)

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""
        # Store results
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )

        if self.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=server_round)

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        train_confidence_score_set, test_confidence_score_set = [], []
        for proxy, _ in results:
            # Request properties from the client
            response = proxy.get_properties(GetPropertiesIns({}), 30, 0)
            train_confidence_scores_json = response.properties.get("train_confidence_scores", None)
            test_confidence_scores_json = response.properties.get("test_confidence_scores", None)

            if train_confidence_scores_json is not None:
                # Deserialize the JSON string back to a list
                confidence_scores = json.loads(train_confidence_scores_json)
                train_confidence_score_set.extend(confidence_scores)

            if test_confidence_scores_json is not None:
                # Deserialize the JSON string back to a list
                confidence_scores = json.loads(test_confidence_scores_json)
                test_confidence_score_set.extend(confidence_scores)

        # Store and log federated evaluation
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )

        # Membership inference only on the configured attack cadence
        if server_round % self.attack_every == 0 and train_confidence_score_set and test_confidence_score_set:
            mia_accuracy = membership_inference(train_confidence_score_set, test_confidence_score_set)
            self.store_results_and_log(server_round=server_round, tag="mia_evaluate", results_dict={"mia_accucary": mia_accuracy})
        
        return loss, metrics

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results using weighted average."""
        # Call parent class method first
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        # ---- IoT overhead metrics (mean across participating clients) ----
        try:
            fit_times = [fr.metrics.get("fit_time_s", 0.0) for _, fr in results if fr.metrics]
            peak_mems = [fr.metrics.get("peak_mem_mb", 0.0) for _, fr in results if fr.metrics]
            num_params = [fr.metrics.get("num_params", 0) for _, fr in results if fr.metrics]
            if fit_times:
                self.store_results_and_log(
                    server_round=server_round,
                    tag="iot_evaluate",
                    results_dict={
                        "mean_fit_time_s": float(np.mean(fit_times)),
                        "max_fit_time_s": float(np.max(fit_times)),
                        "mean_peak_mem_mb": float(np.mean(peak_mems)),
                        "num_params": int(np.max(num_params)) if num_params else 0,
                        # Dense uplink bytes per client (float32 params). Fed-SMP could
                        # transmit only the sparse top-k; we report the dense size here
                        # and note the achievable sparse size separately in analysis.
                        "uplink_bytes_dense": int(np.max(num_params)) * 4 if num_params else 0,
                    },
                )
        except Exception as e:
            logger.log(INFO, f"IoT metric aggregation failed: {e}")

        # ---- A3 index-inference (masking) attack: mean AUC across clients ----
        # AUC ~ 0.5 => adversary cannot separate trainable(b) vs stationary(b')
        # coords from the sent updates (mask concealed); far from 0.5 => it can.
        # Clients return -1 until they have 2 rounds of history; skip those.
        try:
            a3 = [fr.metrics.get("a3_auc", -1.0) for _, fr in results if fr.metrics]
            a3 = [v for v in a3 if v is not None and v >= 0.0]
            if a3:
                self.store_results_and_log(
                    server_round=server_round,
                    tag="a3_evaluate",
                    results_dict={
                        "a3_auc_mean": float(np.mean(a3)),
                        # distinguishing advantage = how far from chance (0.5) the
                        # best (possibly inverted) classifier gets, in [0, 0.5].
                        "a3_advantage": float(np.mean([abs(v - 0.5) for v in a3])),
                    },
                )
        except Exception as e:
            logger.log(INFO, f"A3 metric aggregation failed: {e}")

        # ---- %-stationary: mean fraction of coords classified stationary this round ----
        try:
            sf = [fr.metrics.get("stationary_frac") for _, fr in results if fr.metrics]
            sf = [v for v in sf if isinstance(v, (int, float)) and v >= 0.0]
            if sf:
                self.store_results_and_log(
                    server_round=server_round,
                    tag="stationary_evaluate",
                    results_dict={"stationary_frac": float(np.mean(sf))},
                )
        except Exception as e:
            logger.log(INFO, f"Stationary-fraction aggregation failed: {e}")

        # Perform Model Inversion Attack on individual client models after aggregation
        # (only on the configured attack cadence, to bound cost).
        if server_round % self.attack_every != 0:
            return aggregated_parameters, metrics

        try:
            client_mia_results = []
            
            for i, (client_proxy, fit_res) in enumerate(results):
                # print(f"Client {i} fit_res: {fit_res}")
                parameters = fit_res.parameters
                client_parameters = deepcopy(parameters_to_ndarrays(parameters))

                # Skip the attack on a diverged / non-finite model (e.g. an
                # unstable Fed-SMP run), otherwise the inversion optimization
                # degenerates and errors out.
                if not all(np.isfinite(p).all() for p in client_parameters):
                    logger.log(INFO, f"Skipping MoIA for client {i}: non-finite weights")
                    continue

                # Perform model inversion attack on this client's model
                try:
                    # Create a fresh model instance for this client
                    client_model = self._new_model()
                    client_model.to(self.device)

                    # Set the client's parameters
                    set_weights(client_model, client_parameters)
                    client_model.eval()

                    # Create attack instance with this client's model (dataset-correct input shape)
                    client_attack = ModelInversionAttack(
                        client_model,
                        self.device,
                        attack_config={
                            "image_size": self.image_size,
                            "num_classes": 10,
                            "num_iterations": self.moia_iterations,
                            "norm_mean": self.norm_mean,
                            "norm_std": self.norm_std,
                        },
                    )
                    
                    # Perform the attack
                    inversion_results = client_attack.invert_model_for_all_classes(list(range(10)))
                    
                    # Evaluate quality for each class
                    quality_metrics = {}
                    for class_idx, (data, confidence, metrics) in inversion_results.items():
                        quality = client_attack.evaluate_inversion_quality(data, class_idx)
                        quality_metrics[class_idx] = quality
                    
                    model_inversion_results = {
                        'inversion_results': inversion_results,
                        'quality_metrics': quality_metrics,
                        'client_parameters': client_parameters
                    }
                    
                except Exception as e:
                    logger.log(INFO, f"Model inversion attack failed for client {i}: {e}")
                    continue
                
                # Analyze results for this client
                client_mia_analysis = client_attack.analyze_attack_results(model_inversion_results)
                client_mia_results.append(client_mia_analysis)
            
            # Calculate mean values across all clients
            if client_mia_results:
                avg_confidence = np.mean([r.get('average_target_confidence', 0.0) for r in client_mia_results])
                avg_correct_classification = np.mean([r.get('average_correct_classification', 0.0) for r in client_mia_results])
                max_confidence = np.max([r.get('max_confidence', 0.0) for r in client_mia_results])
                min_confidence = np.min([r.get('min_confidence', 0.0) for r in client_mia_results])
                
                # Store only the mean values
                self.store_results_and_log(
                    server_round=server_round,
                    tag="inversion_evaluate",
                    results_dict={
                        "ie_average_confidence": avg_confidence,
                        "ie_average_correct_classification": avg_correct_classification,
                        "ie_max_confidence": max_confidence,
                        "ie_min_confidence": min_confidence,
                    }
                )
                
        except Exception as e:
            logger.log(INFO, f"Model inversion attack on client models during fit failed: {e}")

        return aggregated_parameters, metrics