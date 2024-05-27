from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
import seaborn as sns
from .data_visualiser import DataVisualiser
import torch

class DataHandler:
    def __init__(self):
        self.data_visualiser = DataVisualiser()

    def load_data(self):
        DATA_PATH = Path('')
        train = pd.read_csv(DATA_PATH / 'Train.csv')
        test = pd.read_csv(DATA_PATH / 'Test.csv')

        return train, test
    
    def transform_columns(self, column):

        column_name = column.name
        if column_name.endswith("angle"):
            column = column.apply(lambda x: x / 360)
        elif column_name == "hour":
            column = column.apply(lambda x: x / 24)
        elif column_name == "month":
            column = column.apply(lambda x: x / 12) 
        elif column_name == "site_latitude":
            column = column.apply(lambda x: (x + 90) / 180) # Latitude ranges from -90 to 90
        elif column_name == "site_longitude":
            column = column.apply(lambda x: (x + 180) / 360) # Longitude ranges from -180 to 180
        elif column.name.endswith("index"):
            pass # Do nothing
    
        # Altitude - Check max height of satellite
        else:
            # Standardise columns
            column = self._standardise_column(column, epsilon=0.0)
            if column.isnull().sum() > 0:
                column = self._standardise_column(column, epsilon=1e-8) # Use epsilon to avoid division by zero
            
            # # Min-Max scale columns
            # column = self._min_max_scale_column(column, epsilon=0.0)
            # if column.isnull().sum() > 0:
            #     column = self._min_max_scale_column(column, epsilon=1e-8) # Use epsilon to avoid division by zero

        return column

    def _standardise_column(self, column, epsilon):
        mean = column.mean()
        std = column.std()
        column = (column - mean) / (std + epsilon)
        return column

    def _min_max_scale_column(self, column, epsilon):
        maximum = column.max()
        minimum = column.min()
        column = (column - minimum) / ((maximum - minimum)+ epsilon)
        return column

    def _process_data(self, train, test):

        # Select only numerical features
        train_num_df = train.select_dtypes(include=['number'])

        # Select X and Y features for modelling
        X = train_num_df.drop("pm2_5", axis = 1)

        # Replace NaN values
        X.interpolate(method="linear", inplace=True)
        X = X.apply(self.transform_columns, axis=0)
        X.fillna(X.median(), inplace=True) # Median because of outliers

        # Set targets
        Y = train.pm2_5
        test_df = test[X.columns]
        test_df.interpolate(method="linear", inplace=True)
        test_df = test_df.apply(self.transform_columns, axis=0)
        test_df.fillna(test_df.median(), inplace=True) # Median because of outliers

        return X, Y, test_df


    def get_data(self, train, test, test_size, random_state_seed):

        # Visualise data
        corr_matrix = self.data_visualiser.get_correlation_matrix(train_dataset=train.select_dtypes(include=["number"]), top_k=10)
        self.data_visualiser.plot_correlation_matrix(train_dataset=train, top_k_corrs=corr_matrix)

        # Process data Ad
        train_x, train_y, test_df = self._process_data(train=train, test=test)
    
        # Split data
        X_train, X_val, Y_train, Y_val = train_test_split(train_x, train_y, test_size=test_size, random_state=random_state_seed)

        return X_train, X_val, Y_train, Y_val, test_df
    
    def get_batches(self, X, Y, batch_size, device):
        
        num_features = X.shape[1] # Number of features
        max_steps = X.shape[0] - (X.shape[0] % batch_size) 
        X = X[:max_steps]
        Y = Y[:max_steps]
        batch_inputs = torch.tensor(X.values, dtype=torch.float32).reshape(-1, batch_size, num_features).to(device)
        batch_targets = torch.tensor(Y.values, dtype=torch.float32).reshape(-1, batch_size, 1).to(device)

        return batch_inputs, batch_targets