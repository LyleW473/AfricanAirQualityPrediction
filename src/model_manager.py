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