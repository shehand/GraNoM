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
import torch
from pf.strategy import CustomFedAvg
from pf.task import (
    build_model_for_dataset,
    get_global_testloader,
    set_seed,
    set_weights,
    test,
)
from torch.utils.data import DataLoader

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig


def _resolve_server_device(spec) -> torch.device:
    """Resolve a `server-device` config string to an available torch.device.

    'cuda' / 'gpu' / 'auto' -> cuda:0 when CUDA is visible in this (driver)
    process, else cpu. An explicit 'cuda:N' is honored when CUDA is visible.
    Anything else (or CUDA unavailable) -> cpu. Falling back instead of raising
    keeps a run alive on a node whose driver cannot see the GPU."""
    s = str(spec).strip().lower()
    if s in ("cuda", "gpu", "auto"):
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if s.startswith("cuda:") and torch.cuda.is_available():
        return torch.device(s)
    return torch.device("cpu")


def gen_evaluate_fn(testloader: DataLoader, device: torch.device, dataset: str, model_override: str):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = build_model_for_dataset(dataset, model_override)
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate


def make_on_fit_config(base_lr: float = 0.1, decay: float = 0.5, decay_after: int = 10):
    """Learning-rate schedule: base_lr, scaled by `decay` after `decay_after` rounds."""

    def on_fit_config(server_round: int):
        lr = base_lr
        if server_round > decay_after:
            lr *= decay
        return {"lr": lr}

    return on_fit_config


# Define metric aggregation function
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    # Read from config
    run_config = context.run_config
    num_rounds = run_config["num-server-rounds"]
    fraction_fit = run_config["fraction-fit"]
    fraction_eval = run_config["fraction-evaluate"]
    server_device = _resolve_server_device(run_config["server-device"])
    print(f"[server] centralized eval device: {server_device} "
          f"(requested '{run_config['server-device']}', cuda_available={torch.cuda.is_available()})",
          flush=True)
    dataset = str(run_config.get("dataset", "mnist"))
    model_override = str(run_config.get("model", ""))
    seed = int(run_config.get("seed", 42))
    set_seed(seed)

    from pf.task import get_weights

    # Initialize model parameters for the chosen dataset/model
    ndarrays = get_weights(build_model_for_dataset(dataset, model_override))
    parameters = ndarrays_to_parameters(ndarrays)

    # Centralized test set for the chosen dataset
    testloader = get_global_testloader(dataset, batch_size=32)

    strategy = CustomFedAvg(
        run_config=run_config,
        use_wandb=run_config["use-wandb"],
        dataset=dataset,
        model_override=model_override,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        initial_parameters=parameters,
        on_fit_config_fn=make_on_fit_config(),
        evaluate_fn=gen_evaluate_fn(testloader, server_device, dataset, model_override),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)