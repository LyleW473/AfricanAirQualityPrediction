import torch
from src.data_handler import DataHandler
from src.model import Model
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
from src.train_utils import create_submission
from src.trainer import Trainer

def train():
    
    torch.manual_seed(2004)
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
    for column in X_train.columns:
        print(column)
        print(X_train[column])

    # Initialise + train model
    model = LGBMRegressor(
                        learning_rate=0.1, 
                        n_estimators=100,

                        # force_col_wise=True, # Set to True to use a column-wise tree construction
                        max_depth= -1, # Maximum tree depth for base learners, -1 means no limit
                        num_leaves=31, # Maximum number of leaves in one tree
                        random_state=random_seed,
                        device="gpu"
                        )
    model.fit(X_train, Y_train)

    # Get predictions on validation set
    preds = model.predict(X_val)
    score = root_mean_squared_error(Y_val, preds)
    print(f"Local RMSE: {score}")

    # Get predictions on test set
    test_preds = model.predict(test_df)
    # create_submission(predictions=test_preds, test_set=test_dataset)

    BATCH_SIZE = 64
    TOTAL_EPOCHS = 1
    trainer = Trainer(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                    learning_rate=0.01
                    )
    print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)
    trainer.train(X_train, Y_train, batch_size=BATCH_SIZE, total_epochs=TOTAL_EPOCHS)
    trainer.evaluate(X_val, Y_val, batch_size=BATCH_SIZE)
    test_preds_2 = trainer.get_predictions_for_dataset(test_df, batch_size=BATCH_SIZE)
    print(test_preds.shape, test_preds_2.shape, test_dataset.shape, type(test_preds), type(test_preds_2), type(test_dataset))
    create_submission(predictions=test_preds_2, test_set=test_dataset)

if __name__ == "__main__":
    train()
