from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.models.well_relevant import ClusterWrapper


# Calculate hits@n metric for a given value of n
def hits_n(cluster_wrapper, x_test, y_test, y_train, n=5):
    result = []
    for i in range(len(x_test)):
        # Predict the cluster assignments for the i-th test data point
        sorted_indices, _ = cluster_wrapper.predict([x_test[i]])

        # Check if the true label is among the top n predicted labels
        top_n_labels = sorted_indices[:n]
        same_classter = (np.repeat(y_test[i], n) == y_train[top_n_labels])
        result.append(same_classter.sum()/n)

    return sum(result)/len(result)


def plot_well_sim(cluster_wrapper, x, x_train):
    # Extract sorted data points and sorted values
    sorted_indices, sorted_values = cluster_wrapper.predict([x])

    # Plot the sorted data points with a gradient color map
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_train[sorted_indices, 0], x_train[sorted_indices, 1], c=sorted_values, cmap='viridis_r', label='Training Data')
    # Adjust color bar range to match the range of sorted values
    plt.colorbar(scatter, label='Sorted Values', ticks=np.arange(min(sorted_values), max(sorted_values)+1, 1))

    plt.scatter(x[0], x[1], c='red', marker='x', label='Test Data')
    #  plt.scatter(X_train[sorted_indices, 0], X_train[sorted_indices, 1], c=y_train[sorted_indices],label='Real Marks', s=200, alpha=0.2)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Sorted Data Points with Gradient Color Map')
    plt.legend()
    plt.grid(True)
    plt.show()
