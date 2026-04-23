import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

st.title("Anomaly Detection using DBSCAN")

# -----------------------------
# DATA INPUT
# -----------------------------
uploaded_file = st.file_uploader("Upload Dataset", type=["txt", "csv"])
use_default = st.checkbox("Use default dataset (KDDTrain+.txt)", value=True)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)
    st.success("Using uploaded dataset")

elif use_default:
    try:
        df = pd.read_csv("KDDTrain+.txt", header=None)
        st.success("Using default dataset")
    except:
        st.error("Dataset not found. Upload file.")
        st.stop()
else:
    st.warning("Upload dataset or use default")
    st.stop()

# Limit size (important)
df = df.head(1000)

st.write("Dataset shape:", df.shape)

# -----------------------------
# PREPROCESSING
# -----------------------------
y = df.iloc[:, -1]
X = df.iloc[:, :-1]

X = pd.get_dummies(X)
X.columns = X.columns.astype(str)

X_scaled = StandardScaler().fit_transform(X)
X_reduced = PCA(n_components=10).fit_transform(X_scaled)

st.write("Reduced shape:", X_reduced.shape)

# -----------------------------
# EPSILON COMPARISON
# -----------------------------
st.subheader("Epsilon Comparison")

eps_values = [0.3, 0.5, 0.7, 1.0]

for eps in eps_values:
    preds = DBSCAN(eps=eps, min_samples=10).fit_predict(X_reduced)
    
    st.write(f"EPS = {eps}")
    st.write("Clusters:", len(set(preds)))
    st.write("Anomalies:", np.sum(preds == -1))

# -----------------------------
# MODEL CONTROL
# -----------------------------
st.subheader("Run DBSCAN")

eps = st.slider("Select Epsilon", 0.1, 1.5, 0.7)

clusters = DBSCAN(eps=eps, min_samples=10).fit_predict(X_reduced)

# -----------------------------
# DEBUG OUTPUT
# -----------------------------
st.subheader("Debug Output")

st.write("Total records:", len(df))
st.write("Clusters found:", len(set(clusters)))
st.write("Anomalies detected:", np.sum(clusters == -1))

df = df.copy()
df["cluster"] = clusters
df["label"] = y.astype(str)

# -----------------------------
# CLUSTER ANALYSIS
# -----------------------------
st.subheader("Cluster Distribution")
st.write(df["cluster"].value_counts())

result = df.groupby("cluster")["label"].value_counts().unstack(fill_value=0)

st.subheader("Cluster vs Label Table")
st.dataframe(result)

# Anomaly analysis
st.subheader("Anomaly Breakdown (-1 cluster)")
if -1 in result.index:
    st.write(result.loc[-1].sort_values(ascending=False))
else:
    st.write("No anomalies detected")

# -----------------------------
# VISUALIZATION
# -----------------------------

# 1. Cluster Scatter
st.subheader("Cluster Visualization")

fig1, ax1 = plt.subplots()
scatter = ax1.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=clusters,
    cmap="rainbow",
    s=10
)
plt.colorbar(scatter)
ax1.set_title("DBSCAN Clustering (PCA Reduced Data)")
st.pyplot(fig1)

# 2. Anomaly Highlight
st.subheader("Anomaly Detection Visualization")

fig2, ax2 = plt.subplots()

# Normal
ax2.scatter(X_reduced[:, 0], X_reduced[:, 1],
            c="lightgray", s=10, label="Normal")

# Anomalies
anomalies = clusters == -1
ax2.scatter(X_reduced[anomalies, 0],
            X_reduced[anomalies, 1],
            c="red", s=20, label="Anomalies (-1)")

ax2.legend()
ax2.set_title("Anomaly Detection using DBSCAN")

st.pyplot(fig2)

# 3. Cluster Distribution Bar Chart
st.subheader("Cluster Distribution Chart")

fig3, ax3 = plt.subplots()
df["cluster"].value_counts().sort_index().plot(kind="bar", ax=ax3)

ax3.set_title("Cluster Distribution")
ax3.set_xlabel("Cluster ID")
ax3.set_ylabel("Number of Samples")

st.pyplot(fig3)