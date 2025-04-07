# K-Means Clustering

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # <-- 1. IMPORT StandardScaler

# Hyperparameters for K-Means
k_values = [2, 3, 4, 5, 6, 7]      # List of numbers of clusters to test
max_iter = 300                     # Maximum number of iterations
tol = 1e-4                         # Tolerance for convergence
init_method = 'k-means++'          # Initialization method
random_state = 42                  # Seed for reproducibility

# Indices for the features to be plotted
feature_x_index = 0  # x-axis
feature_y_index = 1  # y-axis

# Load the dataset and prepare the data
df = pd.read_csv('winequality-red.csv', sep=';')
df = df.drop_duplicates()         # Remove duplicates
X = df.drop('quality', axis=1)      # Remove the target variable "quality"

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

silhouette_scores = []

plt.figure(figsize=(15, 10))
num_rows = 2
num_cols = 3

for i, k in enumerate(k_values):
    # Initialize K-Means with the hyperparameters
    kmeans = KMeans(n_clusters=k, init=init_method, max_iter=max_iter, tol=tol, random_state=random_state)
    
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    
    # Compute the Silhouette Score for k > 1 on scaled data
    score = silhouette_score(X_scaled, labels) if k > 1 else np.nan
    silhouette_scores.append(score)
    
    plt.subplot(num_rows, num_cols, i + 1)
    plt.scatter(X.iloc[:, feature_x_index], X.iloc[:, feature_y_index], c=labels, cmap='viridis', alpha=0.6)
    plt.title(f"K-Means with k = {k}\nSilhouette (scaled): {score:.2f}")
    plt.xlabel(X.columns[feature_x_index])
    plt.ylabel(X.columns[feature_y_index])

plt.tight_layout()
plt.show()

print("Silhouette scores for K-Means (computed on scaled data):")

for k, score in zip(k_values, silhouette_scores):
    print(f"k = {k}: {score:.2f}")
