import torch
from src.data_handler import DataHandler
from src.model import Model
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
from src.train_utils import create_submission
from src.trainer import Trainer
from src.config import CONFIG

def train():
    

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(2004)
    g = torch.Generator(device=DEVICE)
    g.manual_seed(2004)

    random_seed = 42

    # Load data and get splits
    data_handler = DataHandler()
    train_dataset, test_dataset = data_handler.load_data()
    X_train, X_val, Y_train, Y_val, test_df = data_handler.get_data(
                                                                    train=train_dataset, 
                                                                    test=test_dataset, 
                                                                    test_size=0.3, 
                                                                    random_state_seed=random_seed
                                                                    )
    trainer = Trainer(
                    device=DEVICE,
                    generator=g,
                    model_num=18,
                    epoch_num=1000
                    )
    print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

    # Get data in PyTorch format
    train_inputs, train_targets = data_handler.get_batches(X=X_train, Y=Y_train, device=trainer.device, batch_size=CONFIG["hyperparameters"]["batch_size"])
    val_inputs, val_targets = data_handler.get_batches(X=X_val, Y=Y_val, device=trainer.device, batch_size=CONFIG["hyperparameters"]["batch_size"])

    print(train_inputs.shape, train_targets.shape, val_inputs.shape, val_targets.shape)
    
    # Final evaluation
    trainer.evaluate(all_inputs=val_inputs, all_targets=val_targets, losses_list = [], verbose=True)

if __name__ == "__main__":
    train()
