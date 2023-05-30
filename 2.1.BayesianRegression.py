# Author: Yoshihiro Uchida <nimbus1after2a1sun7shower@gmail.com>
# from https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge_curvefit.html
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge


def func(x):

    # just returns a sine wave of 2pi*x
    return np.sin(2 * np.pi * x)


# #############################################################################
# Generate sinusoidal data with noise
size = 25

# a seed set-up
rng = np.random.RandomState(1234)

# training data
x_train = rng.uniform(0.0, 1.0, size)
# y values with an error obtained from random dist with std of 0.1
y_train = func(x_train) + rng.normal(scale=0.1, size=size)

# test values
x_test = np.linspace(0.0, 1.0, 100)


# #############################################################################
# Fit by cubic polynomial
n_order = 3
# creates a Vandermonde matrix
# True means that the powers increase from left to right
# If increasing is True, the columns are x^0, x^1, ..., x^(N-1)
X_train = np.vander(x_train, n_order + 1, increasing=True)
X_test = np.vander(x_test, n_order + 1, increasing=True)

# #############################################################################
# Plot the true and predicted curves with log marginal likelihood (L)
# tol refers to convergence of the algorithm
# compute score adds the log marginal likelihood
reg = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for i, ax in enumerate(axes):
    # Bayesian ridge regression with different initial value pairs
    if i == 0:
        init = [1 / np.var(y_train), 1.0]  # Default values
    elif i == 1:
        init = [1.0, 1e-3]
        reg.set_params(alpha_init=init[0], lambda_init=init[1])

    reg.fit(X_train, y_train)
    ymean, ystd = reg.predict(X_test, return_std=True)

    # plotting
    ax.plot(x_test, func(x_test), color="blue", label="sin($2\\pi x$)")
    ax.scatter(x_train, y_train, s=50, alpha=0.5, label="observation")
    ax.plot(x_test, ymean, color="red", label="predict mean")
    # fill the area between the two horizontal curves
    ax.fill_between(
        x_test, ymean - ystd, ymean + ystd, color="pink", alpha=0.5, label="predict std"
    )
    ax.set_ylim(-1.3, 1.3)
    ax.legend()
    title = "$\\alpha$_init$={:.2f},\\ \\lambda$_init$={}$".format(init[0], init[1])
    if i == 0:
        title += " (Default)"
    ax.set_title(title, fontsize=12)
    text = "$\\alpha={:.1f}$\n$\\lambda={:.3f}$\n$L={:.1f}$".format(
        reg.alpha_, reg.lambda_, reg.scores_[-1]
    )
    ax.text(0.05, -1.0, text, fontsize=12)

plt.tight_layout()
plt.show()

