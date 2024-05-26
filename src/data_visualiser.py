from matplotlib import pyplot as plt
import seaborn as sns

class DataVisualiser:
    def __init__(self):
        pass

    def get_correlation_matrix(self, train_dataset, top_k):
        # Get the correlation matrix
        top_k_corrs = abs(train_dataset.corr()["pm2_5"]).sort_values(ascending = False).head(top_k) # Get the top "k" correlations
        return top_k_corrs

    def plot_correlation_matrix(self, train_dataset, top_k_corrs):
        corr = train_dataset[list(top_k_corrs.index)].corr()
        sns.heatmap(corr, cmap='RdYlGn', annot = True, center = 0)
        plt.title('Correlations between the target and other variables', pad = 15, fontdict={'size': 13})
        plt.show()