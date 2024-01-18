import argparse
import warnings
from collections import OrderedDict

import flwr as fl
from flwr_datasets import FederatedDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from models import Net, train, test
from parameters import DEVICE, NUM_CLIENTS, LEARNING_RATE


warnings.filterwarnings("ignore", category=UserWarning)


# Define dataloader function
def load_data(node_id):
    """Load partition CIFAR10 data."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(node_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


# Get node id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--node-id",
    choices=range(NUM_CLIENTS),
    required=True,
    type=int,
    help="Partition of the dataset divided into 10 iid partitions created artificially.",
)
node_id = parser.parse_args().node_id

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data(node_id=node_id)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        learning_rate: float,
    ):  # pylint: disable=too-many-arguments
        self.learning_rate = learning_rate

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Decay the learning rate if there is a decay factor in the config
        if "learning_rate_decay_factor" in config:
            self.learning_rate *= config["learning_rate_decay_factor"]

        train(net, trainloader, epochs=1, learning_rate=self.learning_rate)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(learning_rate=LEARNING_RATE),
)
