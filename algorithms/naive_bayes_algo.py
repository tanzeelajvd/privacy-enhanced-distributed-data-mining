import numpy as np
import pandas as pd
import math
import os

from algorithms.rss_algo import rss_rmd_ratio


# --------------------------------------------------
# Gaussian PDF
# --------------------------------------------------
def gaussian_pdf(x, mean, var):
    if var <= 0:
        return 1e-9
    exponent = math.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / math.sqrt(2 * math.pi * var)) * exponent


# --------------------------------------------------
# Train Naive Bayes (Simple or RSS-RMD)
# --------------------------------------------------
def train_naive_bayes(X, y, use_rss=True):
    classes = np.unique(y)
    model = {}
    N = len(X)

    for c in classes:
        X_c = X[y == c]
        n_c = len(X_c)

        # Prior P(c)
        if use_rss:
            prior = rss_rmd_ratio(
                np.ones(n_c),
                np.ones(N)
            )
        else:
            prior = n_c / N

        means = []
        variances = []

        for j in range(X.shape[1]):
            values = X_c[:, j]
            ones = np.ones(len(values))

            # Mean
            if use_rss:
                mean = rss_rmd_ratio(values, ones)
            else:
                mean = np.mean(values)

            # Variance (not RSS-based)
            var = np.var(values) + 1e-9

            means.append(mean)
            variances.append(var)

        model[c] = {
            "prior": prior,
            "mean": means,
            "var": variances
        }

    return model


# --------------------------------------------------
# Predict single sample
# --------------------------------------------------
def predict_sample(model, x):
    posteriors = {}

    for c, params in model.items():
        log_prob = math.log(params["prior"] + 1e-12)

        for i in range(len(x)):
            pdf = gaussian_pdf(x[i], params["mean"][i], params["var"][i])
            log_prob += math.log(pdf + 1e-12)

        posteriors[c] = log_prob

    return max(posteriors, key=posteriors.get)


# --------------------------------------------------
# Predict dataset
# --------------------------------------------------
def predict(model, X):
    return np.array([predict_sample(model, x) for x in X])


# --------------------------------------------------
# Data loading utility
# --------------------------------------------------
def load_data(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "transformed_datasets")
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=[np.number])

    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    return X, y

import matplotlib.pyplot as plt

def validation_curve_naive_bayes(X, y, train_sizes=None):
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 0.9, 9)

    simple_acc = []
    rss_acc = []

    # Fixed validation split (last 20%)
    split_val = int(0.8 * len(X))
    X_train_full, X_val = X[:split_val], X[split_val:]
    y_train_full, y_val = y[:split_val], y[split_val:]

    for frac in train_sizes:
        n_train = int(frac * len(X_train_full))

        X_train = X_train_full[:n_train]
        y_train = y_train_full[:n_train]

        # Simple NB
        simple_model = train_naive_bayes(X_train, y_train, use_rss=False)
        y_pred_simple = predict(simple_model, X_val)
        simple_acc.append(np.mean(y_pred_simple == y_val))

        # RSS-RMD NB
        rss_model = train_naive_bayes(X_train, y_train, use_rss=True)
        y_pred_rss = predict(rss_model, X_val)
        rss_acc.append(np.mean(y_pred_rss == y_val))

    return train_sizes, simple_acc, rss_acc


# --------------------------------------------------
# Experiment runner
# --------------------------------------------------
def main():
    file_name = input("Enter transformed CSV filename:\n").strip()

    X, y = load_data(file_name)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("\nTraining Simple Naive Bayes...")
    simple_model = train_naive_bayes(X_train, y_train, use_rss=False)
    y_pred_simple = predict(simple_model, X_test)
    acc_simple = np.mean(y_pred_simple == y_test)

    print("\nTraining RSS-RMD Naive Bayes...")
    rss_model = train_naive_bayes(X_train, y_train, use_rss=True)
    y_pred_rss = predict(rss_model, X_test)
    acc_rss = np.mean(y_pred_rss == y_test)

    print("\nResults:")
    print(f"Simple Naive Bayes Accuracy : {acc_simple:.4f}")
    print(f"RSS-RMD Naive Bayes Accuracy: {acc_rss:.4f}")
    print(f"Accuracy Difference        : {acc_simple - acc_rss:.4f}")

    print("\nGenerating validation curve...")

    train_sizes, simple_acc, rss_acc = validation_curve_naive_bayes(X, y)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes * 100, simple_acc, marker='o', label="Simple Naive Bayes")
    plt.plot(train_sizes * 100, rss_acc, marker='s', label="RSS-RMD Naive Bayes")

    plt.xlabel("Training Data Percentage (%)")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Curve: Naive Bayes vs RSS-RMD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
