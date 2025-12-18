import numpy as np
import pandas as pd
import math
import copy
from collections import defaultdict

from algorithms.rss_algo import rss_rmd_ratio


# --------------------------------------------------
# Gaussian PDF
# --------------------------------------------------
def gaussian_pdf(x, mean, var):
    if var == 0:
        return 1e-9
    exponent = math.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / math.sqrt(2 * math.pi * var)) * exponent


# --------------------------------------------------
# Train Naive Bayes (simple or RSS-RMD)
# --------------------------------------------------
def train_naive_bayes(X, y, use_rss=True):
    classes = np.unique(y)
    model = {}

    for c in classes:
        X_c = X[y == c]
        n_c = len(X_c)

        # Prior P(c)
        if use_rss:
            prior = rss_rmd_ratio(
                np.ones(n_c),
                np.ones(len(X))
            )
        else:
            prior = n_c / len(X)

        means = []
        variances = []

        for j in range(X.shape[1]):
            values = X_c[:, j]
            ones = np.ones(len(values))

            if use_rss:
                mean = rss_rmd_ratio(values, ones)
            else:
                mean = np.mean(values)

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
        log_prob = math.log(params["prior"])

        for i in range(len(x)):
            pdf = gaussian_pdf(x[i], params["mean"][i], params["var"][i])
            log_prob += math.log(pdf + 1e-9)

        posteriors[c] = log_prob

    return max(posteriors, key=posteriors.get)


# --------------------------------------------------
# Predict dataset
# --------------------------------------------------
def predict(model, X):
    return np.array([predict_sample(model, x) for x in X])


# --------------------------------------------------
# Experiment runner
# --------------------------------------------------
def main():
    file_name = input("Enter transformed CSV filename:\n").strip()

    df = pd.read_csv(file_name)
    df = df.select_dtypes(include=[np.number])

    # Last column assumed as label
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

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


if __name__ == "__main__":
    main()
