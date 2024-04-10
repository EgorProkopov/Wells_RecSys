import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


class ClusterWrapper:
    def __init__(self, cluster_model=KMeans, K_range=range(2, 5)):
        self.model = cluster_model
        self.K_range = K_range
        self.models = []  # trained models on dataset
        self.dataset = None
        self.labels = []  # 2d matrix of shape len(K_range) x len(dataset)

    def fit(self, X: pd.DataFrame):
        """Fit the len(K_range) times the self.cluster_models

        Args:
            X (pd.DataFrame): dataset
        """
        # init the dataset and clear the class attributes
        self.dataset = X
        self.models = []
        self.labels = []
        
        for i, K in enumerate(self.K_range):
            # fir K-th model
            model = self.model(n_clusters=K)
            model.fit(X)
            self.models.append(model)
            
            # Store cluster labels for each data point
            labels = model.predict(X)
            self.labels.append(labels)

    def predict(self, x):
        if self.dataset is None or self.models == []:
            raise ValueError("The model has not been trained yet. Please call the fit method first.")

        result = np.zeros((len(self.dataset)))

        for i, model in enumerate(self.models):
            pred = model.predict(x)
            pred = np.repeat(pred[0], len(self.dataset))
            result += (pred == self.labels[i]).astype(int)
        
        sorted_indices = np.argsort(result)[::-1]
        sorted_values = result[sorted_indices]
        return sorted_indices, sorted_values
