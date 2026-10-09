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

Refactored so that dataset, model, and perturbation method are all selected
via run-config rather than by editing this file. See pyproject.toml
[tool.flwr.app.config] for the knobs, and pbs/ for the sweep scripts.
"""

import json
import os
import random
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from flwr.common.typing import UserConfig

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Seed python, numpy and torch RNGs for reproducible runs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry: HuggingFace dataset id, source image column, channel count,
# crop size, normalization stats, and the model to use.
DATASETS = {
    "mnist": {
        "hf": "ylecun/mnist",
        "image_key": "image",
        "channels": 1,
        "crop": 28,
        "norm": ((0.1307,), (0.3081,)),
        "model": "lenet",
    },
    "fmnist": {
        "hf": "zalando-datasets/fashion_mnist",
        "image_key": "image",
        "channels": 1,
        "crop": 28,
        "norm": ((0.2860,), (0.3530,)),
        "model": "lenet",
    },
    "cifar10": {
        "hf": "uoft-cs/cifar10",
        "image_key": "img",
        "channels": 3,
        "crop": 32,
        "norm": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "model": "vgg11",
    },
}


def dataset_spec(dataset: str) -> dict:
    key = str(dataset).lower()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Options: {list(DATASETS)}")
    return DATASETS[key]


def build_transforms(dataset: str):
    """Return (train_transform, eval_transform) for the dataset."""
    spec = dataset_spec(dataset)
    crop = spec["crop"]
    norm = spec["norm"]
    train_t = Compose(
        [
            RandomCrop(crop, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(*norm),
        ]
    )
    eval_t = Compose([ToTensor(), Normalize(*norm)])
    return train_t, eval_t


def make_transform_fn(dataset: str, train: bool):
    """Return a batch-transform that reads the dataset's source column and
    writes a standardized 'image' tensor column (so downstream code is uniform)."""
    spec = dataset_spec(dataset)
    src_key = spec["image_key"]
    train_t, eval_t = build_transforms(dataset)
    tf = train_t if train else eval_t

    def _apply(batch):
        batch["image"] = [tf(img) for img in batch[src_key]]
        if src_key != "image":
            # Drop the original PIL column (e.g. CIFAR-10's "img"); otherwise the
            # DataLoader's default_collate chokes trying to stack PIL images.
            batch.pop(src_key, None)
        return batch

    return _apply


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class LeNet(nn.Module):
    """LeNet-5 for MNIST / Fashion-MNIST (1x28x28 input, 10 classes)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGG(nn.Module):
    """VGG (default VGG11) for CIFAR-10 (3x32x32 input, 10 classes)."""

    def __init__(self, vgg_name: str = "VGG11", num_classes: int = 10):
        super().__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, layer_cfg):
        layers = []
        in_channels = 3
        for x in layer_cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def get_model(model_name: str = "lenet", in_channels: int = 1, num_classes: int = 10):
    name = str(model_name).lower()
    if name == "lenet":
        return LeNet(in_channels=in_channels, num_classes=num_classes)
    if name in ("vgg", "vgg11"):
        return VGG("VGG11", num_classes=num_classes)
    raise ValueError(f"Unknown model '{model_name}'. Options: lenet | vgg11")


def build_model_for_dataset(dataset: str, model_override: str = ""):
    """Instantiate the right model for a dataset (or an explicit override)."""
    spec = dataset_spec(dataset)
    model_name = model_override if model_override else spec["model"]
    return get_model(model_name, in_channels=spec["channels"], num_classes=10)


# Backwards-compatible alias: some code paths import `Net` directly.
Net = LeNet


# ---------------------------------------------------------------------------
# Membership-inference attack model (unchanged)
# ---------------------------------------------------------------------------
class AttackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)


def train_attack_model(train_data, test_data):
    attack_model = AttackModel()
    # AttackModel already ends in Sigmoid, so the loss must be plain BCELoss.
    # BCEWithLogitsLoss would apply a SECOND sigmoid to the [0,1] output, squashing
    # the gradients so the attack cannot learn (it then defaults to ~chance even on
    # perfectly separable confidences). This regression made mia_accucary useless.
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    attack_model.train()
    for _ in range(10):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = attack_model(inputs).view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    attack_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = attack_model(inputs).view(-1)
            preds = (outputs > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    attack_accuracy = correct / total
    print(f"Attack Model Accuracy (Privacy Leakage): {attack_accuracy:.2f}")
    return attack_accuracy


def membership_inference(all_train_confidences, all_test_confidences):
    # BALANCE members (train) vs non-members (test) so the attack's chance level is
    # 50%. Clients are split 80:20 train:val, so members outnumber non-members ~4:1;
    # on the raw (imbalanced) set the attack floors at the 0.80 majority-class
    # fraction and a working defense reads as "no defense". Subsampling to 1:1 makes
    # chance = 50% (defended -> ~50%, leaky/vanilla -> well above 50%), matching the
    # reported MeIA. Seeded so a rerun is deterministic.
    import random
    k = min(len(all_train_confidences), len(all_test_confidences))
    if k == 0:
        return 0.5
    rng = random.Random(0)
    members = rng.sample(list(all_train_confidences), k)
    non_members = rng.sample(list(all_test_confidences), k)

    attack_data = torch.tensor(members + non_members).unsqueeze(1).float()
    attack_labels = torch.tensor([1] * k + [0] * k).float()
    attack_dataset = TensorDataset(attack_data, attack_labels)

    train_size = int(0.8 * len(attack_dataset))
    test_size = len(attack_dataset) - train_size
    train_attack_data, test_attack_data = random_split(
        attack_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(0),
    )

    return train_attack_model(train_attack_data, test_attack_data)


# ---------------------------------------------------------------------------
# Train / test
# ---------------------------------------------------------------------------

def train(net, trainloader, epochs, lr, device, grad_clip_norm=0.0):
    """Train the model on the training set. Returns (net, avg_loss).

    grad_clip_norm > 0 enables L2 gradient-norm clipping in local SGD. Without it,
    once GraNoM converges the fixed Laplace noise is large relative to the (now tiny)
    gradients, and a noise-perturbed step can send momentum-SGD into divergence
    (loss -> inf), which then poisons FedAvg and never recovers (observed as a sudden
    collapse to chance mid-run). Clipping the gradient norm bounds that blow-up. It only
    rescales gradients that exceed the threshold, so it leaves normal updates and the
    zero-gradient (stationary) mask untouched.
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_norm)
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / max(1, len(trainloader))
    return net, avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set. Returns (loss, accuracy)."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / max(1, len(testloader))
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, dataset: str = "mnist", seed: int = 42):
    """Load a Dirichlet-partitioned client shard for the chosen dataset."""
    global fds
    spec = dataset_spec(dataset)
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=1.0,
            seed=int(seed),
        )
        fds = FederatedDataset(
            dataset=spec["hf"],
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=int(seed))

    train_partition = partition_train_test["train"].with_transform(
        make_transform_fn(dataset, train=True)
    )
    test_partition = partition_train_test["test"].with_transform(
        make_transform_fn(dataset, train=False)
    )
    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=32)
    return trainloader, testloader


def reset_dataset_cache():
    """Clear the cached FederatedDataset (call if the dataset changes in-process)."""
    global fds
    fds = None


def get_global_testloader(dataset: str = "mnist", batch_size: int = 32):
    """Load the centralized test split for server-side evaluation."""
    from datasets import load_dataset

    spec = dataset_spec(dataset)
    global_test_set = load_dataset(spec["hf"])["test"]
    testloader = DataLoader(
        global_test_set.with_transform(make_transform_fn(dataset, train=False)),
        batch_size=batch_size,
    )
    return testloader


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

def create_run_dir(config: UserConfig) -> Path:
    """Create an output directory. The path encodes the key experiment knobs
    so a downstream plotting/collection script can find each run."""
    current_time = datetime.now()
    stamp = current_time.strftime("%Y-%m-%d/%H-%M-%S")

    dataset = str(config.get("dataset", "mnist"))
    noise_mode = str(config.get("noise-mode", "granom"))
    mem = "mem" if bool(config.get("memorization", True)) else "nomem"
    eps = config.get("epsilon", "NA")
    epsp = config.get("epsilon-prime", "NA")
    tau = config.get("grad-threshold", "NA")
    seed = config.get("seed", "NA")
    clients = config.get("client-count", "NA")   # threaded in from the PBS CLIENTS var
    tag = str(config.get("results-tag", ""))

    # Base label. NOTE: client count is included because the client count comes
    # from the federation (CLI), not otherwise from run-config, so 10- vs 50-client
    # runs of the same config would otherwise share a directory.
    label = f"{dataset}_{noise_mode}_{mem}_eps{eps}_epsp{epsp}_tau{tau}_c{clients}_seed{seed}"
    # Append the knob that actually varies within each sweep so runs are distinct:
    if noise_mode in ("fedsmp", "dpfedavg"):
        # the sigma-sweep varies these; without them every sigma collides.
        label += f"_sig{config.get('fedsmp-sigma', 'NA')}_q{config.get('fedsmp-ratio', 'NA')}"
    elif noise_mode in ("granom", "uniform"):
        # distinguish the per-round update-clip mechanism from the full-model default.
        label += f"_uc{1 if bool(config.get('granom-update-clip', False)) else 0}"
    if tag:
        label = f"{tag}_{label}"

    save_path = Path.cwd() / "outputs" / label / stamp
    save_path.mkdir(parents=True, exist_ok=True)

    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump({k: config[k] for k in config}, fp)

    return save_path, f"{label}/{stamp}"


# ---------------------------------------------------------------------------
# Perturbation methods
# ---------------------------------------------------------------------------

def _two_scale_laplace(shape, stationary, clip_threshold, epsilon, epsilon_prime, device):
    """Per-coordinate two-scale Laplace noise: scale C/beta' where `stationary`,
    else C/beta. Sampled via inverse-CDF on uniform noise."""
    scale = torch.where(
        stationary,
        torch.full(shape, clip_threshold / epsilon_prime, device=device),
        torch.full(shape, clip_threshold / epsilon, device=device),
    )
    u = torch.rand(shape, device=device) - 0.5
    return -scale * torch.sign(u) * torch.log1p(-2.0 * u.abs())


def apply_granom(model, epsilon, epsilon_prime, clip_threshold, grad_threshold,
                 prev_params=None, update_clip=False):
    """GraNoM two-scale Laplace allocation on the full parameter set.

    A coordinate is 'stationary' when |grad| <= grad_threshold (tau); with
    tau=0.0 this is the exact-zero-gradient rule of Algorithm 1. Stationary
    coordinates get the small scale C/beta_prime, trainable coordinates get
    the larger scale C/beta (here epsilon==beta, epsilon_prime==beta'). Every
    coordinate is perturbed, which is what conceals the stationary mask.

    NOTE on clipping (why the default path does NOT rescale):
    Earlier revisions projected the whole model onto the l1-ball of radius
    C=clip_threshold (``w *= C/||w||_1``) before adding noise. For a ~62k-param
    LeNet ||w||_1 ~ 2000, so that step scales every weight by ~C/||w||_1
    (~5e-4) while the noise scale stays at C/beta, collapsing the model to
    chance. C is therefore kept solely as the numerator of the two Laplace
    scales, not applied as a projection.

    update_clip=True (requires prev_params): bound the PER-ROUND perturbation
    for stability. Add the two-scale noise to the local update
    delta = trained - received, then L2-clip the whole noisy delta to radius C
    and return received + clipped(delta + noise). This caps how far the global
    model can move each round, which is the missing bound that lets the
    no-rescale default drift/collapse after convergence (the fixed noise
    random-walks the converged model to chance). Bounding the update stops that.
    """
    tau = float(grad_threshold)

    if update_clip:
        if prev_params is None:
            raise ValueError("apply_granom(update_clip=True) requires prev_params.")
        keys = list(model.state_dict().keys())
        device = next(model.parameters()).device
        prev = {k: torch.as_tensor(v).to(device) for k, v in zip(keys, prev_params)}
        grads = {n: p.grad for n, p in model.named_parameters()}
        with torch.no_grad():
            sd = model.state_dict()
            noisy_delta = {}
            flat_parts = []
            for k in keys:
                v = sd[k]
                if not v.is_floating_point():
                    continue  # integer buffers pass through unchanged
                d = v.detach() - prev[k]
                g = grads.get(k, None)
                stationary = (g.abs() <= tau) if g is not None \
                    else torch.ones_like(d, dtype=torch.bool)
                noise = _two_scale_laplace(d.shape, stationary, clip_threshold,
                                           epsilon, epsilon_prime, device)
                dn = d + noise
                noisy_delta[k] = dn
                flat_parts.append(dn.reshape(-1))
            # L2-clip the whole noisy update to radius C (bounds per-round change)
            l2 = torch.norm(torch.cat(flat_parts), p=2).item()
            coef = min(1.0, clip_threshold / (l2 + 1e-10))
            new_state = OrderedDict()
            for k in keys:
                if k in noisy_delta:
                    new_state[k] = prev[k] + noisy_delta[k] * coef
                else:
                    new_state[k] = sd[k].detach().clone()
            model.load_state_dict(new_state, strict=True)
        return model

    # Default: two-scale Laplace on the full model, no per-round bound.
    for param in model.parameters():
        with torch.no_grad():
            if param.grad is not None:
                stationary = (param.grad.abs() <= tau)
            else:
                stationary = torch.ones_like(param.data, dtype=torch.bool)
            noise = _two_scale_laplace(param.data.shape, stationary, clip_threshold,
                                       epsilon, epsilon_prime, param.data.device)
            param.data.add_(noise)
    return model


def apply_uniform(model, epsilon, clip_threshold, prev_params=None, update_clip=False):
    """Uniform LDP baseline: single-scale Laplace on every coordinate.

    update_clip=True (requires prev_params) uses the same per-round update bound as
    apply_granom, so the baseline is stabilized the same way GraNoM is (fair comparison):
    add single-scale Laplace to the update delta=trained-received, L2-clip the noisy delta
    to radius C, return received + clipped(delta+noise). Without it, C is only the numerator
    of the Laplace scale C/beta (the no-bound path, which drifts/collapses like GraNoM did).
    """
    scale = clip_threshold / epsilon
    if update_clip:
        if prev_params is None:
            raise ValueError("apply_uniform(update_clip=True) requires prev_params.")
        keys = list(model.state_dict().keys())
        device = next(model.parameters()).device
        prev = {k: torch.as_tensor(v).to(device) for k, v in zip(keys, prev_params)}
        with torch.no_grad():
            sd = model.state_dict()
            noisy_delta = {}
            flat_parts = []
            for k in keys:
                v = sd[k]
                if not v.is_floating_point():
                    continue
                d = v.detach() - prev[k]
                u = torch.rand_like(d) - 0.5
                noise = -scale * torch.sign(u) * torch.log1p(-2.0 * u.abs())
                dn = d + noise
                noisy_delta[k] = dn
                flat_parts.append(dn.reshape(-1))
            l2 = torch.norm(torch.cat(flat_parts), p=2).item()
            coef = min(1.0, clip_threshold / (l2 + 1e-10))
            new_state = OrderedDict()
            for k in keys:
                if k in noisy_delta:
                    new_state[k] = prev[k] + noisy_delta[k] * coef
                else:
                    new_state[k] = sd[k].detach().clone()
            model.load_state_dict(new_state, strict=True)
        return model

    for param in model.parameters():
        with torch.no_grad():
            u = torch.rand_like(param.data) - 0.5
            noise = -scale * torch.sign(u) * torch.log1p(-2.0 * u.abs())
            param.data.add_(noise)
    return model


def apply_fedsmp(model, prev_params, sparsify_ratio, sigma, clip_threshold):
    """Fed-SMP baseline (sparsified model perturbation).

    Reference: Hu, Gong, Guo, "Federated Learning with Sparsified Model
    Perturbation: Improving Accuracy under Client-Level Differential Privacy,"
    IEEE Trans. Mobile Computing, 2024 (arXiv:2202.07178).

    Steps (top-k variant):
      1. delta = w_trained - w_received  (the local update)
      2. L2-clip delta to radius C
      3. keep the top-k fraction of coordinates by |delta|, zero the rest
      4. add Gaussian noise N(0, (sigma*C)^2) to the KEPT coordinates only
      5. transmit w_received + noisy_sparse_delta

    This is a HARD/BINARY selective mechanism: unkept coordinates are sent
    unchanged (delta=0), which is precisely the index-inference-exposing
    pattern GraNoM's soft two-scale allocation avoids.

    `prev_params` is the list of received global ndarrays (pre-training).
    """
    if prev_params is None:
        raise ValueError("Fed-SMP requires prev_params (received global weights).")

    # Reconstruct received weights as tensors keyed like the state_dict
    keys = list(model.state_dict().keys())
    prev = {k: torch.tensor(v) for k, v in zip(keys, prev_params)}

    with torch.no_grad():
        # Only perturb floating-point tensors. Integer buffers (e.g. BatchNorm
        # num_batches_tracked) are passed through as the trained value.
        float_keys = [k for k, v in model.state_dict().items() if v.is_floating_point()]

        deltas = OrderedDict()
        flat_parts = []
        for k in float_keys:
            v = model.state_dict()[k]
            d = v.detach().cpu() - prev[k]
            deltas[k] = d
            flat_parts.append(d.reshape(-1))
        flat = torch.cat(flat_parts)
        total = flat.numel()

        # L2-clip the whole (float) update
        l2 = torch.norm(flat, p=2).item()
        clip_coef = min(1.0, clip_threshold / (l2 + 1e-10))
        if clip_coef < 1.0:
            for k in deltas:
                deltas[k] = deltas[k] * clip_coef
            flat = flat * clip_coef

        # Top-k mask by |delta| (global, over float coordinates)
        k_keep = max(1, int(round(float(sparsify_ratio) * total)))
        if k_keep < total:
            thresh = torch.topk(flat.abs(), k_keep, largest=True).values.min()
        else:
            thresh = torch.tensor(-1.0)

        noise_std = float(sigma) * float(clip_threshold)
        new_state = OrderedDict()
        for k, v in model.state_dict().items():
            if k not in deltas:  # non-float buffer: keep trained value unchanged
                new_state[k] = v.detach().clone()
                continue
            d = deltas[k]
            keep = d.abs() >= thresh
            gauss = torch.randn_like(d) * noise_std
            d_noisy = torch.where(keep, d + gauss, torch.zeros_like(d))
            new_state[k] = (prev[k] + d_noisy).to(v.dtype)
        model.load_state_dict(new_state, strict=True)
    return model


def apply_privacy(
    model,
    noise_mode="granom",
    prev_params=None,
    epsilon=10.0,
    epsilon_prime=100.0,
    clip_threshold=1.0,
    grad_threshold=0.0,
    fedsmp_ratio=0.1,
    fedsmp_sigma=1.0,
    granom_update_clip=False,
):
    """Dispatch to the configured perturbation method."""
    mode = str(noise_mode).lower()
    if mode in ("none", "vanilla", "nonoise"):
        return model  # vanilla FL: send the trained model unperturbed (no privacy)
    if mode == "granom":
        return apply_granom(model, epsilon, epsilon_prime, clip_threshold, grad_threshold,
                            prev_params=prev_params, update_clip=granom_update_clip)
    if mode == "uniform":
        return apply_uniform(model, epsilon, clip_threshold,
                             prev_params=prev_params, update_clip=granom_update_clip)
    if mode == "fedsmp":
        return apply_fedsmp(model, prev_params, fedsmp_ratio, fedsmp_sigma, clip_threshold)
    if mode in ("dpfedavg", "dp-fedavg", "dpfa"):
        # DP-FedAvg = Fed-SMP without sparsification: L2-clip the update to C and add
        # Gaussian noise to ALL coordinates (ratio=1.0). Standard DP-FL baseline; like
        # Fed-SMP it perturbs the UPDATE, leaving a clean converged model.
        return apply_fedsmp(model, prev_params, 1.0, fedsmp_sigma, clip_threshold)
    raise ValueError(f"Unknown noise-mode '{noise_mode}'. Options: granom | uniform | fedsmp | dpfedavg")


# Backwards-compatible wrapper matching the old signature/name.
def local_differential_privacy(model, clip_threshold=1.0, epsilon=10.0, use_granom=True):
    """Deprecated shim kept for compatibility. Prefer apply_privacy()."""
    mode = "granom" if use_granom else "uniform"
    return apply_privacy(
        model, noise_mode=mode, epsilon=epsilon, clip_threshold=clip_threshold
    )
