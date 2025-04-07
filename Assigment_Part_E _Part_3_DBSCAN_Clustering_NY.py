# DBSCAN Clustering

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Hyperparameters for DBSCAN
eps_values = [0.3, 0.5, 0.7]      # List of eps values
min_samples = 5                   # Minimum number of points to define a core point
metric = 'euclidean'
algorithm = 'auto'

# Indices for the features to be plotted
feature_x_index = 8  # x-axis
feature_y_index = 10  # y-axis

# Load the dataset and prepare the data
df = pd.read_csv('winequality-red.csv', sep=';')
df = df.drop_duplicates()         # Remove duplicates
X = df.drop('quality', axis=1)      # Remove the target variable "quality"

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Experiment with different eps values
plt.figure(figsize=(15, 5))
for i, eps in enumerate(eps_values):
    # Initialize and fit DBSCAN with the specified hyperparameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm)
    dbscan.fit(X_scaled)
    labels = dbscan.labels_
    
    # Compute the Silhouette Score, excluding noise
    unique_labels = set(labels)
    if len(unique_labels) > 1 and -1 in unique_labels:
        mask = labels != -1
        if len(set(labels[mask])) > 1:
            score = silhouette_score(X_scaled[mask], labels[mask])
        else:
            score = np.nan
    elif len(unique_labels) > 1:
        score = silhouette_score(X_scaled, labels)
    else:
        score = np.nan
    
    plt.subplot(1, len(eps_values), i + 1)
    plt.scatter(X_scaled.iloc[:, feature_x_index], X_scaled.iloc[:, feature_y_index],
                c=labels, cmap='viridis', alpha=0.6)
    plt.title(f"DBSCAN eps = {eps}\nSilhouette: {score:.2f}")
    plt.xlabel(X_scaled.columns[feature_x_index])
    plt.ylabel(X_scaled.columns[feature_y_index])

plt.tight_layout()
plt.show()
