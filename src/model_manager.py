import os
import torch
from src.model import Model
from src.config import CONFIG
from copy import deepcopy

class ModelManager:

    def __init__(self, device):
        self.device = device

    def initialise_model(self):
        config = deepcopy(CONFIG)

        if not os.path.exists("model_checkpoints"):
            os.makedirs("model_checkpoints")
        num_existing_models = len(os.listdir("model_checkpoints"))
        checkpoint_directory = f"model_checkpoints/{num_existing_models}"
        os.mkdir(checkpoint_directory)

        hyperparams = config["hyperparameters"]
        model = Model().to(device=self.device)
        optimiser = config["model"]["optimiser"](model.parameters(), lr=hyperparams["learning_rate"])

        return config, model, optimiser, hyperparams, checkpoint_directory
    
    def load_model(self, model_num, epoch_num):
        config = torch.load(f"model_checkpoints/{model_num}/{epoch_num}.pt")
        checkpoint_directory = f"model_checkpoints/{model_num}"

        hyperparams = config["hyperparameters"]
        model = Model().to(device=self.device)
        optimiser = config["model"]["optimiser"](model.parameters(), lr=hyperparams["learning_rate"])

        model.load_state_dict(config["model"]["model_state_dict"])
        optimiser.load_state_dict(config["model"]["optimiser_state_dict"])

        return config, model, optimiser, hyperparams, checkpoint_directory