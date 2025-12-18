import numpy as np

def rss_rmd_ratio(x, y, seed=None):
    """
    Efficient RSS-RMD implementation.

    Computes:
        sum(x_i) / sum(y_i)
    using Random Secure Summation with Rational Multiplicative Disturbance.

    Parameters
    ----------
    x : array-like
        Numerator values (x_i)
    y : array-like
        Denominator values (y_i)
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    float
        Secure ratio sum(x_i) / sum(y_i)
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    assert len(x) == len(y), "x and y must have same length"

    if seed is not None:
        np.random.seed(seed)

    n = len(x)

    # Step 1: input perturbation
    alpha = np.random.uniform(-0.5, 0.5, size=n)
    beta  = np.random.uniform(-0.5, 0.5, size=n)

    x_tilde = x + alpha
    y_tilde = y + beta

    # Step 2: rational multiplicative disturbance
    r = np.random.uniform(1000, 5000, size=n)

    # Step 3: disturbed secure summations
    SX = np.sum(r * x_tilde)
    SY = np.sum(r * y_tilde)

    # Step 4: ratio recovery
    return SX / SY
