import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
print("Starting program...")

# Load dataset
file_path = "KDDTrain+.txt"

df = pd.read_csv(file_path, header=None)
print(df.shape)

# Split features and label
y = df.iloc[:, -1]
X = df.iloc[:, :-1]

# Convert categorical to numerical
X = pd.get_dummies(X)
X.columns = X.columns.astype(str)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA reduction
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_scaled)

print(X_reduced.shape)

# Use smaller dataset
# Use smaller dataset
df = pd.read_csv(file_path, header=None, nrows=1000)

print(df.shape)

y = df.iloc[:, -1]
X = df.iloc[:, :-1]
print(X.shape, y.shape)

X = pd.get_dummies(X)
X.columns = X.columns.astype(str)

X_scaled = StandardScaler().fit_transform(X)

X_reduced = PCA(n_components=10).fit_transform(X_scaled)
print(X_reduced.shape)
# -----------------------------
# EPSILON COMPARISON
# -----------------------------
eps_values = [0.3, 0.5, 0.7, 1.0]

print("\n--- EPSILON COMPARISON ---")

for eps in eps_values:
    preds = DBSCAN(eps=eps, min_samples=10).fit_predict(X_reduced)
    
    print(f"\nEPS = {eps}")
    print("Clusters:", len(set(preds)))
    print("Anomalies:", np.sum(preds == -1))

# Apply DBSCAN
clusters = DBSCAN(eps=0.7, min_samples=10).fit_predict(X_reduced)
# -----------------------------
# DEBUG OUTPUT (VERY IMPORTANT)
# -----------------------------
print("\nTotal records:", len(df))
print("Clusters found:", len(set(clusters)))
print("Anomalies detected:", np.sum(clusters == -1))

df = df.copy()
df["cluster"] = clusters

# THIS WAS MISSING
df["label"] = y.astype(str)

print(df["cluster"].value_counts())

# Clean labeled output
result = df.groupby("cluster")["label"].value_counts().unstack(fill_value=0)
print(result)

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=df["cluster"], cmap="rainbow", s=10)
plt.title("DBSCAN Clustering (PCA Reduced Data)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster ID")
plt.show()

plt.figure(figsize=(8,6))

# normal clusters
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c="lightgray", s=10, label="Normal clusters")

# anomalies
anomalies = df["cluster"] == -1
plt.scatter(X_reduced[anomalies, 0], X_reduced[anomalies, 1], c="red", s=20, label="Anomalies (-1)")

plt.title("Anomaly Detection using DBSCAN")
plt.legend()
plt.show()

df["cluster"].value_counts().plot(kind="bar", figsize=(10,5))
plt.title("Cluster Distribution")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Points")
plt.show()

plt.figure(figsize=(8,6))

plt.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=df["cluster"],
    cmap="rainbow",
    s=10
)

plt.title("DBSCAN Clustering (PCA Reduced Data)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.colorbar(label="Cluster ID")
plt.show()

plt.figure(figsize=(10,5))

df["cluster"].value_counts().sort_index().plot(kind="bar")

plt.title("DBSCAN Cluster Distribution")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Samples")

plt.show()
# -----------------------------
# VISUALIZATION 3: EPSILON COMPARISON
# -----------------------------
eps_values = [0.3, 0.5, 0.7]

plt.figure()

for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_scaled)
    plt.scatter(X_scaled[:,0], X_scaled[:,1], label=f"eps={eps}", alpha=0.3)

plt.legend()
plt.title("Epsilon Comparison")
plt.show()
