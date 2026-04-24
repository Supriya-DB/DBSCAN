import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

st.title("Network Intrusion Detection using DBSCAN (NSL-KDD Dataset)")
st.markdown("""
This project detects network intrusions using the NSL-KDD dataset.
DBSCAN is used to identify anomalous (malicious) network traffic patterns
without requiring labeled training data.
""")

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

for eps_val in eps_values:
    preds = DBSCAN(eps=eps_val, min_samples=10).fit_predict(X_reduced)
    st.write(f"EPS = {eps_val}")
    st.write("Clusters:", len(set(preds)) - (1 if -1 in preds else 0))
    st.write("Anomalies:", np.sum(preds == -1))

# -----------------------------
# MODEL RUN
# -----------------------------
st.subheader("Run DBSCAN")

eps = st.slider("Select Epsilon", 0.1, 1.5, 0.7)

clusters = DBSCAN(eps=eps, min_samples=10).fit_predict(X_reduced)

df = df.copy()
df["cluster"] = clusters
df["label"] = y.astype(str)

# -----------------------------
# DEBUG OUTPUT
# -----------------------------
st.subheader("Debug Output")

st.write("Total records:", len(df))
st.write("Clusters found:", len(set(clusters)) - (1 if -1 in clusters else 0))
st.write("Anomalies detected:", np.sum(clusters == -1))

# -----------------------------
# EPSILON vs ANOMALY %
# -----------------------------
def anomaly_percentage_per_epsilon(X, eps_values, min_samples=10):
    results = []

    for eps_val in eps_values:
        labels = DBSCAN(eps=eps_val, min_samples=min_samples).fit_predict(X)
        results.append({
            "epsilon": eps_val,
            "anomaly_percentage": np.mean(labels == -1) * 100
        })

    return pd.DataFrame(results)

eval_df = anomaly_percentage_per_epsilon(X_reduced, eps_values)

# =========================================================
# 🔥 VISUALS AT TOP (RIGHT AFTER EPSILON)
# =========================================================
st.subheader(" Visualizations ")

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. Cluster Distribution
cluster_counts = pd.Series(clusters).value_counts().sort_index()
cluster_counts.plot(kind="bar", ax=axs[0, 0])
axs[0, 0].set_title("Cluster Distribution")

# 2. Cluster Scatter
scatter = axs[0, 1].scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=clusters,
    cmap="rainbow",
    s=10
)
axs[0, 1].set_title("DBSCAN Clusters")
fig.colorbar(scatter, ax=axs[0, 1])

# 3. Anomaly Visualization
axs[1, 0].scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c="lightgray",
    s=10,
    label="Normal"
)

anomalies = clusters == -1

axs[1, 0].scatter(
    X_reduced[anomalies, 0],
    X_reduced[anomalies, 1],
    c="red",
    s=20,
    label="Anomalies"
)

axs[1, 0].legend()
axs[1, 0].set_title("Anomaly Detection")

# 4. Epsilon vs Anomaly %
axs[1, 1].plot(
    eval_df["epsilon"],
    eval_df["anomaly_percentage"],
    marker="o"
)

axs[1, 1].set_title("Epsilon vs Anomaly %")

plt.tight_layout()
st.pyplot(fig)

# =========================================================
# 🔚 TABLES AT THE END (AS REQUESTED)
# =========================================================

st.subheader("Cluster vs Label Table")

result = df.groupby("cluster")["label"].value_counts().unstack(fill_value=0)
st.dataframe(result)

st.subheader("Anomaly Breakdown (-1 cluster)")

if -1 in result.index:
    st.write(result.loc[-1].sort_values(ascending=False))
else:
    st.write("No anomalies detected")