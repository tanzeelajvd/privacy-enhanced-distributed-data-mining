import copy
import time
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from algorithms.rss_algo import rss_rmd_ratio


# --------------------------------------------------
# Distance function (Euclidean)
# --------------------------------------------------
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


# --------------------------------------------------
# K-Means with optional RSS-RMD centroid update
# --------------------------------------------------
def k_means_algorithm(k, clients_data, use_rss=True, tol=1e-6):
    point_dimension = len(clients_data[0][0])
    num_clients = len(clients_data)

    # Initialize centroids from data
    centroids = []
    for i in range(k):
        centroids.append(clients_data[i % num_clients][i // num_clients])

    centroids = np.array(centroids)
    iteration = 0

    while True:
        iteration += 1

        # Step 1: Assign points to nearest centroid
        client_clusters = [[[] for _ in range(k)] for _ in range(num_clients)]

        for client_id, client_points in enumerate(clients_data):
            for point in client_points:
                distances = [euclidean_distance(point, c) for c in centroids]
                cluster_idx = int(np.argmin(distances))
                client_clusters[client_id][cluster_idx].append(point)

        # Step 2: Merge clusters across clients
        all_clusters = []
        for i in range(k):
            merged = []
            for j in range(num_clients):
                merged.extend(client_clusters[j][i])
            all_clusters.append(merged)

        # Step 3: Compute new centroids
        new_centroids = []

        for idx, cluster_points in enumerate(all_clusters):
            if len(cluster_points) == 0:
                new_centroids.append(centroids[idx])
                continue

            cluster_points = np.array(cluster_points)
            centroid = []

            for dim in range(point_dimension):
                values = cluster_points[:, dim]
                ones = np.ones(len(values))

                if use_rss:
                    mean_val = rss_rmd_ratio(values, ones)
                else:
                    mean_val = np.mean(values)

                centroid.append(mean_val)

            new_centroids.append(centroid)

        new_centroids = np.array(new_centroids)

        # Step 4: Convergence check
        if np.allclose(centroids, new_centroids, atol=tol):
            print(f"Converged in {iteration} iterations")
            return new_centroids, all_clusters

        centroids = new_centroids

        # Step 5: Update client data
        new_clients_data = []
        for i in range(num_clients):
            flat = []
            for c in client_clusters[i]:
                flat.extend(c)
            new_clients_data.append(flat)

        clients_data = copy.deepcopy(new_clients_data)


# --------------------------------------------------
# Convert clusters to labels (for silhouette)
# --------------------------------------------------
def clusters_to_labels(clusters):
    labels = []
    for idx, cluster in enumerate(clusters):
        labels.extend([idx] * len(cluster))
    return np.array(labels)


# --------------------------------------------------
# Data loading utilities
# --------------------------------------------------
def load_data(filename, num_points=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "transformed_datasets")
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    data = df.select_dtypes(include=[np.number]).to_numpy()

    if num_points:
        data = data[:num_points]

    clients = 3
    return np.array_split(data, clients)


# --------------------------------------------------
# Validation curve: Silhouette vs k
# --------------------------------------------------
def validation_curve_kmeans(X, clients_data, k_values):
    simple_scores = []
    rss_scores = []

    for k in k_values:
        print(f"\nEvaluating k = {k}")

        rss_centroids, rss_clusters = k_means_algorithm(
            k, copy.deepcopy(clients_data), use_rss=True
        )
        rss_labels = clusters_to_labels(rss_clusters)
        rss_scores.append(silhouette_score(X, rss_labels))

        simple_centroids, simple_clusters = k_means_algorithm(
            k, copy.deepcopy(clients_data), use_rss=False
        )
        simple_labels = clusters_to_labels(simple_clusters)
        simple_scores.append(silhouette_score(X, simple_labels))

    return simple_scores, rss_scores


# --------------------------------------------------
# Experiment runner
# --------------------------------------------------
def main():
    file_name = input("Enter transformed CSV filename:\n").strip()
    num_points = int(input("Enter number of points (0 for all):\n").strip())
    k = int(input("Enter number of clusters:\n").strip())

    clients_data = load_data(file_name, None if num_points == 0 else num_points)
    X_all = np.vstack(clients_data)

    print("\nRunning RSS-RMD K-Means...")
    start = time.time()
    rss_centroids, rss_clusters = k_means_algorithm(
        k, copy.deepcopy(clients_data), use_rss=True
    )
    rss_time = time.time() - start
    print(f"RSS-RMD completed in {rss_time:.4f} seconds")

    print("\nRunning Simple K-Means...")
    start = time.time()
    simple_centroids, simple_clusters = k_means_algorithm(
        k, copy.deepcopy(clients_data), use_rss=False
    )
    simple_time = time.time() - start
    print(f"Simple K-Means completed in {simple_time:.4f} seconds")

    print(f"\nTime difference (RSS - Simple): {rss_time - simple_time:.4f} seconds")

    # --------------------------------------------------
    # Validation Curve
    # --------------------------------------------------
    print("\nGenerating clustering validation curve...")

    k_values = range(2, min(8, len(X_all) - 1))
    simple_scores, rss_scores = validation_curve_kmeans(
        X_all, clients_data, k_values
    )

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, simple_scores, marker="o", label="Simple K-Means")
    plt.plot(k_values, rss_scores, marker="s", label="RSS-RMD K-Means")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Validation Curve: K-Means vs RSS-RMD K-Means")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
