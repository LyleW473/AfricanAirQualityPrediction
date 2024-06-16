import numpy as np


class OutlierDetector:
    def __init__(self):
        pass
        
    def handle_outliers(self, train, use_median, base_window_size=10):
        
        # Get stats for each city
        city_stats = {}
        for city in train["city"].unique():
            city_rows = train[train["city"] == city]
            std = city_rows["pm2_5"].std()
            median = city_rows["pm2_5"].median()
            mean = city_rows["pm2_5"].mean()
            print(f"City: {city} | Pm2.5 median| {median} | Pm2.5 mean | {mean} | Pm2.5 Std {std}")

            city_stats[city] = {"mean": mean, "median": median, "std": std}

        # Compute dynamic parameters dependent on city statistics
        city_params = compute_city_params(city_stats=city_stats, base_window_size=base_window_size)

        # Find outliers
        indexes, num_outliers = find_indexes_of_outliers(use_median=use_median, city_params=city_params, train=train)

        for city, num_outlier in num_outliers.items():
            print(f"City: {city} | Number of outliers: {num_outlier}")
        
        # Handle outliers
        print("Total number of outliers", len(indexes))
        print("Train shape before removing outliers", train.shape)
        train = train.drop(indexes)
        print("Train shape after removing outliers", train.shape)
        return train
    
# Computing city parameters based on the statistics of each city's pm2.5 emissions
# Used for anomaly detection
def compute_city_params(city_stats, base_window_size):
    city_params = {}
    for city, stats in city_stats.items():
        mean = stats['mean']
        std = stats['std']
        
        # Determine the standard deviation multiplier
        std_multiplier = 2 if std < 0.5 * mean else 3

        # Set a threshold at the 95th percentile, assuming a normal distribution for simplicity
        threshold = mean + 1.645 * std

        # Get window size, based on the variability (std / mean ratio)
        variability_ratio = std / mean
        window_size = int(base_window_size * (1 + variability_ratio))
        
        city_params[city] = {"threshold": threshold, "std_multiplier": std_multiplier, "window_size": window_size}
    
    return city_params


# Function to calculate rolling average and finding outliers
def is_outlier(value, rolling_stat, rolling_iqr_or_std, threshold, std_multiplier):
    if np.isnan(rolling_stat):
        return False
    dynamic_threshold = std_multiplier * rolling_iqr_or_std
    return abs(value - rolling_stat) > dynamic_threshold or value > threshold

def rolling_window_stats(data, window_size, use_median):
    
    if use_median:
        rolling_stat = data.rolling(window=window_size).median()
        rolling_iqr_or_std = data.rolling(window=window_size).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25)) # IQR
    else:
        rolling_stat = data.rolling(window=window_size).mean()
        rolling_iqr_or_std = data.rolling(window=window_size).std()

    return rolling_stat, rolling_iqr_or_std

def find_indexes_of_outliers(use_median, city_params, train):
    """
    use_median: Use median and IQR instead of mean and std for rolling window
    """

    # Finding indexes of all outliers
    indexes = []
    cities_to_remove_outliers = ["Lagos", "Nairobi"]
    num_outliers = {city: 0 for city in cities_to_remove_outliers}
    for city, params, in city_params.items():
        if not city in cities_to_remove_outliers:
            continue
            
        # Get rolling stats
        city_data = train[train["city"] == city]
        window_size = params["window_size"]
        rolling_stats, rolling_iqr_or_stds = rolling_window_stats(city_data["pm2_5"], window_size, use_median=use_median)


        # Find outliers
        threshold = params["threshold"]
        std_multiplier = params["std_multiplier"]
        outliers = city_data.apply(lambda row: is_outlier(
                                                            value=row["pm2_5"],
                                                            rolling_stat=rolling_stats[row.name], 
                                                            rolling_iqr_or_std=rolling_iqr_or_stds[row.name],

                                                            # Dynamic threshold and std multiplier based on city statistics
                                                            threshold=threshold,
                                                            std_multiplier=std_multiplier
                                                            ),
                                                            axis=1)


        indexes.extend(outliers[outliers == True].index)
        num_outliers[city] = outliers[outliers == True].shape[0]
        print("City", city, outliers[outliers == True].shape[0])
        print()

        print(city_data[outliers == True]["pm2_5"])

    return indexes, num_outliers
