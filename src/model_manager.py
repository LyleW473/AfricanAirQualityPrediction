import os
import torch
from src.model import Model
from src.config import CONFIG
from copy import deepcopy

class ModelManager:

    def __init__(self, device):
        self.device = device

    def initialise_model(self):

        self.clean_empty_directories()

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
    
    def clean_empty_directories(self):
        
        models_dir = "model_checkpoints"
        for model in os.listdir(models_dir):
            if len(os.listdir(f"{models_dir}/{model}")) == 0:
                os.rmdir(f"{models_dir}/{model}")

        self.rename_files()

    def rename_files(self):
        
        models_dir = "model_checkpoints"
        model_dir_names = sorted(os.listdir(models_dir), key=lambda x:int(x), reverse=True) # Start from the end
        new_dir_num = len(model_dir_names) - 1 # Zero-indexed
        
        # Rename directories to avoid clashes with renaming
        for old_model_num in model_dir_names:
            os.rename(src=f"{models_dir}/{old_model_num}", dst=f"{models_dir}/_{old_model_num}")

        model_dir_names = sorted(os.listdir(models_dir), key=lambda x:int(x.strip("_")), reverse=True) # Start from the end again
        for old_model_num in model_dir_names:
            # Rename directory
            os.rename(src=f"{models_dir}/{old_model_num}", dst=f"{models_dir}/{new_dir_num}")
            new_dir_num -= 1
        