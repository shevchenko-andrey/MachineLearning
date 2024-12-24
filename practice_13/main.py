import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset
data = pd.read_csv("Mall_Customers.csv")
data.head()

# Step 2: Select features for the first model
features_2d = data[['Spending Score (1-100)', 'Annual Income (k$)']].values

# Step 3: Create a scatter plot to visualize the features
plt.scatter(features_2d[:, 0], features_2d[:, 1], s=50, c='blue')
plt.title("Scatter plot of Spending Score vs Annual Income")
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Annual Income (k$)")
plt.show()

# Step 4: Apply the elbow method to find the optimal number of clusters
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_2d)
    inertia.append(kmeans.inertia_)

plt.plot(k_values, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# Step 5: Evaluate clustering performance using silhouette coefficient
# Choose the optimal number of clusters from the elbow method (e.g., k=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(features_2d)

# Calculate silhouette coefficient
silhouette_avg = silhouette_score(features_2d, labels)
print(f"Silhouette Coefficient for k={optimal_k}: {silhouette_avg}")

# Step 6: Visualize clustering results for the 2D model
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
plt.title("Clustering Results (2D)")
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Annual Income (k$)")
plt.show()
