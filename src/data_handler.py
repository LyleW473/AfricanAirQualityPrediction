from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
import seaborn as sns

class DataHandler:
    def __init__(self):
        # top10_corrs = abs(train_num_df.corr()['pm2_5']).sort_values(ascending = False).head(10)
        # corr = train_num_df[list(top10_corrs.index)].corr()
        # sns.heatmap(corr, cmap='RdYlGn', annot = True, center = 0)
        # plt.title('Correlations between the target and other variables', pad = 15, fontdict={'size': 13})
        # plt.show()
        pass

    def _load_data(self):
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


    def get_data(self, test_size, random_state_seed):

        train, test = self._load_data()
        train_x, train_y, test_df = self._process_data(train=train, test=test)

        X_train, X_val, Y_train, Y_val = train_test_split(train_x, train_y, test_size=test_size, random_state=random_state_seed)

        return X_train, X_val, Y_train, Y_val, test_df
