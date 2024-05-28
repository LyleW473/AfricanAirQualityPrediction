from pathlib import Path
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
import seaborn as sns
from .data_visualiser import DataVisualiser
import torch
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import math

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
    
    def _add_time_features(self, train_dataset, test_dataset):
        # Get features that are not numerical features
        non_numerical_features = train_dataset.select_dtypes(exclude=["number"]).columns
        print(non_numerical_features)

        # Create a new column for days
        train_dataset["day"] = pd.to_datetime(train_dataset["date"]).apply(lambda x: x.day)
        test_dataset["day"] = pd.to_datetime(test_dataset["date"]).apply(lambda x: x.day)
        print(train_dataset["day"])

    def _apply_haversine(self, latitude1, longitude1, latitude2, longitude2):
        # Haversine formula to calculate distance between two points on Earth
        
        # Distance between latitudes and longitudes
        dLat = (latitude1 - latitude2) * (math.pi / 180.0)
        dLon = (longitude1 - longitude2) * (math.pi / 180.0)

        # Convert to radians
        lat1 = (latitude1) * (math.pi / 180.0)
        lat2 = (latitude2) * (math.pi / 180.0)

        # Apply Haversine formula
        a = math.pow(math.sin(dLat / 2), 2) + math.pow(math.sin(dLon / 2), 2) * math.cos(lat1) * math.cos(lat2)
        distance = 6371 * (2 * math.asin(math.sqrt(a))) # 6371 = Earth radius in km

        return distance

    def _add_distance_features(self, dataset):

        print(dataset.columns)

        all_sites = dataset[["site_id", "site_latitude", "site_longitude"]].drop_duplicates()
        
        # Add haversine distance
        dist_df = pd.DataFrame(index=all_sites['site_id'], columns=all_sites['site_id'])
        for site1 in all_sites["site_id"]:
            for site2 in all_sites["site_id"]:
                lat1, lat2 =  all_sites[all_sites["site_id"] == site1]["site_latitude"].values[0], all_sites[all_sites["site_id"] == site2]["site_latitude"].values[0]
                lon1, lon2 =  all_sites[all_sites["site_id"] == site1]["site_longitude"].values[0], all_sites[all_sites["site_id"] == site2]["site_longitude"].values[0]
                haversine_distance = self._apply_haversine(latitude1=lat1, longitude1=lon1, latitude2=lat2, longitude2=lon2)
                dist_df.loc[site1, site2] = haversine_distance

        print(dist_df)

        # Convert the entire dataframe to float32 (otherwise it will be object type)
        dist_df = dist_df.astype("float32")
        print(dist_df.dtypes)

        # Rename columns
        dist_df.columns = [f"distance_to_site_{i}" for i in range(len(dist_df.columns))]
        print(dist_df)
        
        # Merge distance features
        print(dataset.shape)
        dataset = pd.merge(dataset, dist_df, left_on="site_id", right_index=True, how="left") # Merge on site_id 
        print("------------")
        print(dataset.shape)
        print(dataset)
        print()
        return dataset

    def _impute_values(self, dataset, feature_columns, num_iterations):

        for _ in range(num_iterations):
            for column in dataset.columns:

                # Impute missing values
                if dataset[column].isna().sum() > 0:
                    
                    train_data = dataset[dataset[column].notna()]
                    predict_data = dataset[dataset[column].isna()]

                    # Data used to train regression model
                    x_train = train_data[feature_columns]
                    y_train = train_data[column]

                    # Data used to get predictions
                    x_predict = predict_data[feature_columns]

                    # Fit model
                    model = DecisionTreeRegressor(
                                                criterion="squared_error",
                                                random_state=42,
                                                )
                    
                    model.fit(x_train, y_train)
                    preds = model.predict(x_predict) # Get predictions

                    # print(column, preds)
                    # print()

                    # Replace missing values
                    dataset.loc[dataset[column].isna(), column] = preds

        # print(dataset.isna().sum())

    def _process_data(self, train, test):

        # Add time features
        self._add_time_features(train_dataset=train, test_dataset=test)

        # Add distance features
        train = self._add_distance_features(dataset=train)
        test = self._add_distance_features(dataset=test)

        # Select only numerical features
        train_num_df = train.select_dtypes(include=['number'])

        # # Select X and Y features for modelling
        # X = train_num_df.drop("pm2_5", axis = 1)
        X = train_num_df

        # Replace NaN values
        # X.interpolate(method="linear", inplace=True) 

        train_feature_columns = [column for column in X.columns if column.startswith("distance_to_site")] + ["hour", "day", "month", "pm2_5"]
        self._impute_values(dataset=X, feature_columns=train_feature_columns, num_iterations=1)
        X = X.apply(self.transform_columns, axis=0)
        X.fillna(X.median(), inplace=True) # Median because of outliers

        # Set targets
        Y = train_num_df["pm2_5"]
        allowed_test_columns = [column for column in test.columns if column.startswith("distance_to_site")] + [column for column in X.columns if not column.startswith("distance_to_site") and column != "pm2_5"]
        test_df = test[allowed_test_columns]
        # test_df.interpolate(method="linear", inplace=True)
        test_feature_columns = [column for column in test_df.columns if column.startswith("distance_to_site")] + ["hour", "day", "month"]
        self._impute_values(dataset=test, feature_columns=test_feature_columns, num_iterations=1) # pm2_5 is not in test dataset

        test_df = test_df.apply(self.transform_columns, axis=0)
        test_df.fillna(test_df.median(), inplace=True) # Median because of outliers

        # Remove distance features (They )
        X = X.drop([column for column in X.columns if column.startswith("distance_to_site")], axis=1)
        test_df = test_df.drop([column for column in test_df.columns if column.startswith("distance_to_site")], axis=1)

        return X, Y, test_df


    def get_data(self, train, test, test_size, random_state_seed):

        # Visualise data
        corr_matrix = self.data_visualiser.get_correlation_matrix(train_dataset=train.select_dtypes(include=["number"]), top_k=10)
        self.data_visualiser.plot_correlation_matrix(train_dataset=train, top_k_corrs=corr_matrix)

        # Process data Ad
        train_x, train_y, test_df = self._process_data(train=train, test=test)
    
        # Split data
        X_train, X_val, Y_train, Y_val = train_test_split(train_x, train_y, test_size=test_size, random_state=random_state_seed)
        
        print(X_train.shape, type(X_train), Y_train.shape)

        # Remove outliers (temp)
        condition = X_train["pm2_5"] < 150
        X_train = X_train[condition]
        Y_train = Y_train[condition]
        print(X_train.shape, Y_train.shape)
        
        X_train = X_train.drop("pm2_5", axis=1)
        X_val = X_val.drop("pm2_5", axis=1)

        return X_train, X_val, Y_train, Y_val, test_df
    
    def get_batches(self, X, Y, batch_size, device):
        
        num_features = X.shape[1] # Number of features
        max_steps = X.shape[0] - (X.shape[0] % batch_size) 
        X = X[:max_steps]
        Y = Y[:max_steps]
        batch_inputs = torch.tensor(X.values, dtype=torch.float32).reshape(-1, batch_size, num_features).to(device)
        batch_targets = torch.tensor(Y.values, dtype=torch.float32).reshape(-1, batch_size, 1).to(device)

        return batch_inputs, batch_targets