import copy
import time
import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from algorithms.rss_algo import rss_rmd_ratio

# --------------------------------------------------
# Distance function (Euclidean)
# --------------------------------------------------
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


# --------------------------------------------------
# K-Means with pluggable centroid update
# --------------------------------------------------
def k_means_algorithm(k, clients_data, use_rss=True):
    point_dimension = len(clients_data[0][0])
    num_clients = len(clients_data)

    # Initialize centroids randomly from data
    centroids = []
    for i in range(k):
        centroids.append(clients_data[i % num_clients][i // num_clients])

    prev_clusters = None
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

        # Step 2: Merge client clusters
        all_clusters = []
        for i in range(k):
            merged = []
            for j in range(num_clients):
                merged.extend(client_clusters[j][i])
            all_clusters.append(merged)

        # Convergence check
        if prev_clusters == all_clusters:
            print(f"Converged in {iteration} iterations")
            return centroids, all_clusters

        # Step 3: Update centroids
        new_centroids = []

        for cluster_points in all_clusters:
            if len(cluster_points) == 0:
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

        # Prepare for next iteration
        centroids = copy.deepcopy(new_centroids)
        prev_clusters = copy.deepcopy(all_clusters)

        # Update client data (as in your original logic)
        new_clients_data = []
        for i in range(num_clients):
            flat = []
            for c in client_clusters[i]:
                flat.extend(c)
            new_clients_data.append(flat)

        clients_data = copy.deepcopy(new_clients_data)


# --------------------------------------------------
# Data utilities
# --------------------------------------------------
def split_into_n(arr, parts):
    return np.array_split(arr, parts)


def load_data(filename, num_points=None):
    df = pd.read_csv(filename)
    data = df.select_dtypes(include=[np.number]).to_numpy()

    if num_points:
        data = data[:num_points]

    clients = 3
    return split_into_n(data, clients)


# --------------------------------------------------
# Experiment runner
# --------------------------------------------------
def main():
    file_name = input("Enter transformed CSV filename:\n").strip()
    num_points = int(input("Enter number of points (0 for all):\n").strip())
    k = int(input("Enter number of clusters:\n").strip())

    clients_data = load_data(file_name, None if num_points == 0 else num_points)

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

    print(
        f"\nTime difference (RSS - Simple): {rss_time - simple_time:.4f} seconds"
    )


if __name__ == "__main__":
    main()
