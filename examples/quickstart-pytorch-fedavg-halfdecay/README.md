# Example of FedAvg-HalfDecay Based on PyTorch

This example demonstrates the client and server implementation for a custom strategy named [fedavg_halfdecay](https://github.com/jaeminshin95/flower-custom-strategy/blob/main/src/py/flwr/server/strategy/fedavg_halfdecay.py). In this strategy, half of the clients use constant learning rate, while the other half of the clients are told to perform learning rate decay at each round.

This example uses [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset. A simple CNN adapted from '[Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)' is used for training.

## Setup

- Install Python 3.11 (any version ≥ 3.8 would work)
```shell
# Clone this repo
git clone https://github.com/jaeminshin95/flower-custom-strategy.git

# Install flwr from source
cd flower-custom-strategy
python -m pip install -e .

# Install required packages
pip install torch torchvision matplotlib flwr-datasets
```

## Files

- This example directory contains the following files:
```shell
-- README.md
-- client.py
-- global_test_accuracy.png
-- models.py
-- parameters.py
-- run.sh
-- server.py
```

## Running the example

```shell
cd flower-custom-strategy/examples/quickstart-pytorch-fedavg-halfdecay
./run.sh
```

## Parameters

* You can change the following parameters in `parameters.py`:

```shell
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Hardware to conduct PyTorch operations. You could fix this to use either GPU or CPU.
NUM_CLIENTS = 10 # Total number of clients participating in FL.
NUM_ROUNDS = 50 # Number of rounds to train in FL.
LEARNING_RATE = 0.01 # Learning rate for clients to perform local training.
LEARNING_RATE_DECAY = 0.99 # Learning rate decay factor, for half of the clients to decay their learning rates with the custom strategy.
```

* To change `NUM_CLIENTS`, you need to change the number of clients that are spawned in `run.sh`. To achieve this, change the number `9` in `line 9`'s `for-loop` statement into a number that is equal to `NUM_CLIENTS - 1`.
* You can change the model architecture in `models.py`.

## Global accuracy results

* Below is the graph showing the global accuracy after each round, when there were 10 clients training for 50 rounds, with a learning rate of 0.01 and a learning rate decay factor of 0.99:
<p align="center">
  <img src="https://github.com/jaeminshin95/flower-custom-strategy/blob/main/examples/quickstart-pytorch-fedavg-halfdecay/global_test_accuracy.png" width="800px" alt="Global Accuracy Results" />
</p>

