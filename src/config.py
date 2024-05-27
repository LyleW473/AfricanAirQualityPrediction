import torch

STATS_TRACK_INTERVAL = 1
SAVE_INTERVAL = 100

BATCH_SIZE = 64
TOTAL_EPOCHS = 1000

CONFIG = {
            "model": {
                    "model_state_dict": None,
                    "optimiser_state_dict": None,
                    "optimiser": torch.optim.AdamW,
                    },
            "hyperparameters": {
                                "batch_size": 64,
                                "learning_rate": 0.0005,
                                "num_epochs": 10000,
                                },
            "stats": {
                    "train_losses": [],
                    "val_losses": []
                    },

            "misc": {
                    "current_epoch": 1,
                    }
          
          
          }