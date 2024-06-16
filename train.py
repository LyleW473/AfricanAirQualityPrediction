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
    # for column in X_train.columns:
    #     print(column)
    #     print(X_train[column])

    # Initialise + train model
    model = LGBMRegressor(
                        learning_rate=0.1, 
                        n_estimators=100,

                        # force_col_wise=True, # Set to True to use a column-wise tree construction
                        max_depth= -1, # Maximum tree depth for base learners, -1 means no limit
                        num_leaves=31, # Maximum number of leaves in one tree
                        random_state=random_seed,
                        # device="gpu" # Note: Will cause reproducibility issues when set to "gpu"
                        num_iterations=1000,
                        )
    model.fit(X_train, Y_train)

    # Get predictions on validation set
    preds = model.predict(X_val)
    score = root_mean_squared_error(Y_val, preds)
    print(f"Local RMSE: {score}")

    # Get predictions on test set
    test_preds = model.predict(test_df)
    create_submission(predictions=test_preds, test_set=test_dataset)

    trainer = Trainer(
                    device=DEVICE,
                    generator=g,
                    )
    print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

    # Get data in PyTorch format
    train_inputs, train_targets = data_handler.get_batches(X=X_train, Y=Y_train, device=trainer.device, batch_size=CONFIG["hyperparameters"]["batch_size"])
    val_inputs, val_targets = data_handler.get_batches(X=X_val, Y=Y_val, device=trainer.device, batch_size=CONFIG["hyperparameters"]["batch_size"])

    print(train_inputs.shape, train_targets.shape, val_inputs.shape, val_targets.shape)

    # Train model, validating after each epoch
    trainer.execute(
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    val_inputs=val_inputs,
                    val_targets=val_targets,
                    total_epochs=CONFIG["hyperparameters"]["num_epochs"],
                    )
    
    # Final evaluation
    trainer.evaluate(all_inputs=val_inputs, all_targets=val_targets, losses_list = [], verbose=True)

    test_preds_2 = trainer.get_predictions_for_dataset(test_df, batch_size=CONFIG["hyperparameters"]["batch_size"])
    print(test_preds.shape, test_preds_2.shape, test_dataset.shape, type(test_preds), type(test_preds_2), type(test_dataset))
    create_submission(predictions=test_preds_2, test_set=test_dataset)

if __name__ == "__main__":
    train()
