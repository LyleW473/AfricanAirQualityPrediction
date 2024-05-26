from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
import seaborn as sns
from .data_visualiser import DataVisualiser

class DataHandler:
    def __init__(self):
        self.data_visualiser = DataVisualiser()

    def load_data(self):
        DATA_PATH = Path('')
        train = pd.read_csv(DATA_PATH / 'Train.csv')
        test = pd.read_csv(DATA_PATH / 'Test.csv')

        return train, test
    
    def _process_data(self, train, test):

        # Select only numerical features
        train_num_df = train.select_dtypes(include=['number'])

        # Select X and y features for modelling
        X = train_num_df.drop("pm2_5", axis = 1)
        Y = train.pm2_5

        test_df = test[X.columns]

        return X, Y, test_df


    def get_data(self, train, test, test_size, random_state_seed):

        # Visualise data
        corr_matrix = self.data_visualiser.get_correlation_matrix(train_dataset=train.select_dtypes(include=["number"]), top_k=10)
        self.data_visualiser.plot_correlation_matrix(train_dataset=train, top_k_corrs=corr_matrix)

        # Process data 
        train_x, train_y, test_df = self._process_data(train=train, test=test)
    
        # Split data
        X_train, X_val, Y_train, Y_val = train_test_split(train_x, train_y, test_size=test_size, random_state=random_state_seed)

        return X_train, X_val, Y_train, Y_val, test_df