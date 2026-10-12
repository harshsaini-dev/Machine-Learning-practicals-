# Lab Sheet 04 - Harsh Saini

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Run this file or notebook from the practicals folder.
DATA = Path("datasets")
OUT = Path("outputs") / "lab_04"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 78)
pd.set_option("display.max_columns", 8)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
np.set_printoptions(precision=3, suppress=True)
sns.set_theme(style="whitegrid", font_scale=0.9)

def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=160)
    plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram


# Program 01: Load a dataset suitable for clustering using the Pandas library.

try:
    data = pd.read_csv(DATA / "iris.csv")
except (FileNotFoundError, pd.errors.ParserError) as error:
    raise RuntimeError("Check datasets/iris.csv") from error
print("Iris feature dataset:", data.shape)
print("Missing values:", data.isna().sum().sum())


# Program 02: Display the first and last five records of the dataset.

print("First five:")
print(data.head())
print("Last five:")
print(data.tail())


# Program 03: Explore dataset information, summary statistics, and data types.

data.info()
print("Data types:")
print(data.dtypes)
print("Summary:")
print(data.describe().round(2))


# Program 04: Select relevant numerical features for clustering.

X = data.select_dtypes(include="number").copy()
print("Selected:", X.columns.tolist())
print("Shape:", X.shape)


# Program 05: Standardize the dataset before clustering using StandardScaler.

scaler = StandardScaler()
scaled = scaler.fit_transform(X)
print("First three standardized rows:")
print(scaled[:3])
print("Feature means:", scaled.mean(axis=0).round(3))


# Program 06: Implement the K-Means clustering algorithm.

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(scaled)
print("Model:", kmeans)
print("Inertia:", round(kmeans.inertia_, 3))


# Program 07: Apply K-Means clustering with K = 2 clusters.

km2 = KMeans(n_clusters=2, n_init=10, random_state=42)
labels2 = km2.fit_predict(scaled)
print("Cluster counts:", np.bincount(labels2))
print("Silhouette:", round(silhouette_score(scaled, labels2), 3))


# Program 08: Apply K-Means clustering with K = 3 clusters.

labels3 = kmeans.labels_
print("Cluster counts:", np.bincount(labels3))
print("Silhouette:", round(silhouette_score(scaled, labels3), 3))


# Program 09: Apply K-Means clustering with different values of K and compare the results.

k_results = []
for k in range(2, 7):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(scaled)
    k_results.append({"K": k, "inertia": model.inertia_,
                      "silhouette": silhouette_score(scaled, labels)})
k_table = pd.DataFrame(k_results)
print(k_table)


# Program 10: Determine the optimal number of clusters using the Elbow Method.

elbow_values = []
for k in range(1, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(scaled)
    elbow_values.append(model.inertia_)
plt.figure(figsize=(6, 3))
plt.plot(range(1, 9), elbow_values, marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow method")
save_plot("10_elbow")
print("The bend suggests a small K; inspect K=2 and K=3.")


# Program 11: Visualize clusters using a scatter plot.

plt.figure(figsize=(6, 3.5))
plt.scatter(X.iloc[:, 2], X.iloc[:, 3], c=labels3,
            cmap="viridis", s=25)
plt.xlabel(X.columns[2])
plt.ylabel(X.columns[3])
plt.title("K-Means clusters (K=3)")
save_plot("11_kmeans_scatter")


# Program 12: Display the cluster centroids obtained from the K-Means algorithm.

centroids = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=X.columns)
print("Centroids in original units:")
print(centroids)


# Program 13: Assign cluster labels to the original dataset.

clustered = data.copy()
clustered["cluster"] = labels3
print(clustered.head(8))
print(clustered["cluster"].value_counts().sort_index())


# Program 14: Compare clustering results before and after feature scaling.

raw_model = KMeans(n_clusters=3, n_init=10, random_state=42)
raw_labels = raw_model.fit_predict(X)
print("Raw-space silhouette:", round(
    silhouette_score(X, raw_labels), 3))
print("Scaled-space silhouette:", round(
    silhouette_score(scaled, labels3), 3))
print("Partition agreement (ARI):", round(
    adjusted_rand_score(raw_labels, labels3), 3))


# Program 15: Implement Agglomerative Hierarchical Clustering.

hierarchical = AgglomerativeClustering(n_clusters=3)
hier_labels = hierarchical.fit_predict(scaled)
print("Cluster counts:", np.bincount(hier_labels))
print("Silhouette:", round(silhouette_score(scaled, hier_labels), 3))


# Program 16: Generate a dendrogram using the SciPy library.

linkage_matrix = linkage(scaled, method="ward")
plt.figure(figsize=(7, 3.5))
dendrogram(linkage_matrix, truncate_mode="lastp", p=15,
           show_leaf_counts=True)
plt.xlabel("Merged groups (truncated view)")
plt.ylabel("Ward linkage distance")
plt.title("Iris dendrogram")
save_plot("16_dendrogram")


# Program 17: Apply Hierarchical Clustering using Ward linkage.

ward = AgglomerativeClustering(n_clusters=3, linkage="ward")
ward_labels = ward.fit_predict(scaled)
print("Ward silhouette:", round(
    silhouette_score(scaled, ward_labels), 3))


# Program 18: Apply Hierarchical Clustering using Complete linkage.

complete = AgglomerativeClustering(n_clusters=3,
                                   linkage="complete")
complete_labels = complete.fit_predict(scaled)
print("Complete silhouette:", round(
    silhouette_score(scaled, complete_labels), 3))
print("Counts:", np.bincount(complete_labels))


# Program 19: Compare different linkage methods.

linkage_results = []
for method in ["ward", "complete", "average", "single"]:
    model = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = model.fit_predict(scaled)
    linkage_results.append({"linkage": method,
        "silhouette": silhouette_score(scaled, labels),
        "smallest_cluster": np.bincount(labels).min()})
print(pd.DataFrame(linkage_results))


# Program 20: Visualize Hierarchical Clustering results.

plt.figure(figsize=(6, 3.5))
plt.scatter(X.iloc[:, 2], X.iloc[:, 3], c=ward_labels,
            cmap="viridis", s=25)
plt.xlabel(X.columns[2])
plt.ylabel(X.columns[3])
plt.title("Ward hierarchical clusters")
save_plot("20_hierarchical")


# Program 21: Compare Hierarchical Clustering with K-Means clustering.

cluster_comparison = pd.DataFrame({
    "method": ["K-Means", "Ward"],
    "silhouette": [silhouette_score(scaled, labels3),
                   silhouette_score(scaled, ward_labels)]})
print(cluster_comparison)
print("Partition agreement (ARI):", round(
    adjusted_rand_score(labels3, ward_labels), 3))


# Program 22: Apply Principal Component Analysis (PCA) to reduce dataset dimensions.

full_pca = PCA().fit(scaled)
print("Original dimensions:", scaled.shape[1])
print("All explained variance ratios:",
      full_pca.explained_variance_ratio_.round(4))


# Program 23: Reduce a dataset from multiple features to two principal components.

pca2 = PCA(n_components=2)
reduced = pca2.fit_transform(scaled)
print("Reduced shape:", reduced.shape)
print("First five rows:")
print(reduced[:5])


# Program 24: Visualize the transformed data using a two-dimensional scatter plot.

plt.figure(figsize=(6, 3.5))
plt.scatter(reduced[:, 0], reduced[:, 1], c=labels3,
            cmap="viridis", s=25)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Two-dimensional PCA view")
save_plot("24_pca_scatter")


# Program 25: Display the explained variance ratio of principal components.

print("PC1 and PC2 variance ratios:",
      pca2.explained_variance_ratio_.round(4))
print("Total retained variance:", round(
    pca2.explained_variance_ratio_.sum(), 4))


# Program 26: Determine the cumulative explained variance.

cumulative = np.cumsum(full_pca.explained_variance_ratio_)
print("Cumulative variance:", cumulative.round(4))
print("Components needed for at least 95% variance:",
      np.searchsorted(cumulative, 0.95) + 1)


# Program 27: Compare the dataset before and after dimensionality reduction.

reconstructed = pca2.inverse_transform(reduced)
print("Before PCA:", scaled.shape)
print("After PCA:", reduced.shape)
print("Reconstruction MSE:", round(
    np.mean((scaled - reconstructed) ** 2), 5))


# Program 28: Perform K-Means clustering on the PCA-transformed dataset.

pca_kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
pca_labels = pca_kmeans.fit_predict(reduced)
print("PCA cluster counts:", np.bincount(pca_labels))
print("PCA-space silhouette:", round(
    silhouette_score(reduced, pca_labels), 3))


# Program 29: Compare clustering performance before and after applying PCA.

print("Four-feature silhouette:", round(
    silhouette_score(scaled, labels3), 3))
print("PCA-space silhouette:", round(
    silhouette_score(reduced, pca_labels), 3))
print("PCA labels scored in original scaled space:", round(
    silhouette_score(scaled, pca_labels), 3))
print("Partition agreement (ARI):", round(
    adjusted_rand_score(labels3, pca_labels), 3))


# Program 30: Calculate the Silhouette Score for cluster evaluation.

score = silhouette_score(scaled, labels3)
print("K=3 silhouette:", round(score, 3))
print("Range: -1 to 1; larger usually means clearer separation.")


# Program 31: Compare Silhouette Scores for different values of K.

print(k_table[["K", "silhouette"]])
print("Highest silhouette among tested K:",
      int(k_table.loc[k_table["silhouette"].idxmax(), "K"]))


# Program 32: Analyze the characteristics of each cluster using descriptive statistics.

print("Cluster mean measurements:")
print(clustered.groupby("cluster").mean().round(3))
print("Cluster standard deviations:")
print(clustered.groupby("cluster").std().round(3))


# Program 33: Visualize cluster distributions using pair plots or scatter plots.

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
axes[0].scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels3, s=18)
axes[0].set_xlabel("Sepal length")
axes[0].set_ylabel("Sepal width")
axes[1].scatter(X.iloc[:, 2], X.iloc[:, 3], c=labels3, s=18)
axes[1].set_xlabel("Petal length")
axes[1].set_ylabel("Petal width")
save_plot("33_cluster_distributions")


# Program 34: Save the clustered dataset as a CSV file.

clustered.to_csv(OUT / "iris_clustered.csv", index=False)
print("Saved: iris_clustered.csv")
print("Reloaded shape:",
      pd.read_csv(OUT / "iris_clustered.csv").shape)


# Program 35: Compare the performance of K-Means and Hierarchical Clustering and summarize the findings.

print(cluster_comparison)
print("K-Means: centroid-based, requires K and a seed.")
print("Ward: merge-based, can be inspected with a dendrogram.")
print("Highest score here:", cluster_comparison.loc[
    cluster_comparison["silhouette"].idxmax(), "method"])
print("These scores describe this dataset and scaling choice.")
