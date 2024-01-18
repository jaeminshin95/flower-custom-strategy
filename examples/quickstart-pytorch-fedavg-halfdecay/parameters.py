import torch


# Parameters for the FL run
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 10
NUM_ROUNDS = 50
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.99
