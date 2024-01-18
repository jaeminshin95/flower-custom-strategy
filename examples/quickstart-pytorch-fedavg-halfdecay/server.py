import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict

import flwr as fl
from flwr_datasets import FederatedDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from models import Net, test
from parameters import DEVICE, NUM_ROUNDS, LEARNING_RATE_DECAY


# Define dataloader function
def load_data():
    """Load CIFAR10 test data."""
    fds = FederatedDataset(dataset="cifar10", partitioners={})
    test_dataset = fds.load_full("test")
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    test_dataset = test_dataset.with_transform(apply_transforms)
    testloader = DataLoader(test_dataset, batch_size=32)
    return testloader


# Load data
testloader = load_data()


# Define evaluate function for global accuracy
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    """Use the entire CIFAR-10 test set for evaluation."""
    net = Net().to(DEVICE)
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    loss, accuracy = test(net, testloader)
    return loss, {"accuracy": accuracy}


# Define strategy
strategy = fl.server.strategy.FedAvgHalfDecay(
    evaluate_fn=evaluate, learning_rate_decay_factor=LEARNING_RATE_DECAY
)

# Start Flower server
hist = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)

# Plot the global accuracy after each round
plt.figure(figsize=(13, 7))
plt.plot(
    # Global accuracy result before round 1 is discarded
    list(range(1, NUM_ROUNDS + 1)),
    [metric for round, metric in hist.metrics_centralized["accuracy"][1:]],
    "o-",
)
plt.xticks(list(range(5, NUM_ROUNDS + 1, 5)))
plt.xlabel("Rounds", fontsize=30)
plt.ylabel("Global Test Accuracy", fontsize=30)
plt.tick_params(labelsize=26)
plt.savefig("./global_test_accuracy.png", bbox_inches="tight")
