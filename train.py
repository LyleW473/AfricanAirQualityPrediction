import torch
from src.data_handler import DataHandler
from src.model import Model


def train():
    data_handler = DataHandler()
    X_train, X_val, Y_train, Y_val, test_df = data_handler.get_data(test_size=0.2, random_state_seed=42)

    print(X_train.columns)

    

if __name__ == "__main__":
    train()
