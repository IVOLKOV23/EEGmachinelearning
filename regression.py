import matplotlib.pyplot as plt
from eeghelpers import *
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random


def generateStraightLineDataset(n=10000, min_x=0, max_x=100, seed=None):
    # straight line equation
    # if you use the same seed you will get the rnd values
    random.seed(seed)

    # generate x values (equally distributed)
    x_vals = np.linspace(min_x, max_x, num=n)

    # create corresponding y values
    y_vals = 0.5 * x_vals + 2

    # add noise: random, normally dist
    # mean of 0
    # std of 3
    # size of y (# of samples)
    errs = np.random.normal(0, 3, y_vals.shape)
    y_vals = y_vals + errs

    return x_vals, y_vals


def best_fit(X, Y):
    # fit y=mx+c least squares method
    # xbar - Sum(X)/n
    # ybar - Sum(Y)/n
    # both are just to simplify the formula
    xbar = sum(X) / len(X)
    ybar = sum(Y) / len(Y)

    # number of elements
    n = len(X)  # or len(Y)

    # zip() produces tuples of form (x, y) - get (x, y) coordinates
    # n*sum(xy) - sum(x)*sum(y)
    numer = sum([xi * yi for xi, yi in zip(X, Y)]) - n * xbar * ybar

    # n*sum(x^2) - (sum(x))^2
    denum = sum([xi ** 2 for xi in X]) - n * xbar ** 2
    m = numer / denum

    # c = (sum(y) - m*sum(x))/n
    c = ybar - m * xbar
    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(c, m))

    return m, c


def leastSquares(x_true, y_true, plot=False):
    # fit y=mx+c
    m, c = best_fit(x_true, y_true)
    x = np.arange(0, 100)
    y = m * x + c
    if plot:
        plt.plot(x, y)
        plt.legend(["Data", f"y={m}x+{c}"])
        plt.show()
    return m, c


def generatePolyDataset(n=10000, min_x=0, max_x=100, seed=None, plot=False):
    # if you use the same seed you will get the rnd values
    random.seed(seed)

    # generate x values (equally distributed)
    x_vals = np.linspace(min_x, max_x, num=n)

    # create corresponding y values
    # vals=x_vals+errs
    # y_vals=gnd_truth(vals)
    y_vals = 0.5 * np.power(x_vals, 2) + 3 * x_vals + 2

    # create error in the data so that it is not perfect.
    # errs=2*np.random.random_sample(len(x_vals))
    # errs=3*np.random.random_sample(len(x_vals))
    errs = np.random.normal(0, 3, y_vals.shape)

    y_vals = y_vals + errs

    # get a plot when plot = True
    if plot:
        plt.plot(x_vals, y_vals)
        plt.legend(["Data"])
        plt.show()
    return x_vals, y_vals


def func(x, a, b, c):
    # poly function
    return a + b * x + c * x * x


if __name__ == '__main__':

    ####
    # the question is what can we build a system to predict y given a value of x?
    # 1. First we create some toy data in this case F(x)=mx+c
    # the dataset (true values)
    # straight line
    x_true, y_true = generateStraightLineDataset()
    plt.plot(x_true, y_true, 'o')

    ####
    # 2. Make an assumption about the model - in this case y=mx+c
    # and then use various methods to determine the parameters m and c.
    # we assume that the True dataset is linear
    m, c = leastSquares(x_true, y_true, plot=False)

    # plotting
    plt.figure(1)
    plt.plot(x_true, m * x_true + c)
    plt.title("Straight Line of Best Fit (LS Method)")
    plt.legend(["Generated Data", "Line of best fit"])
    plt.text(60, 25, 'y = ' + '{:.2f}'.format(c) + ' + {:.2f}'.format(m) + 'x', size=10)

    ###############
    # Achieving the same above using sci-kit learn
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    from sklearn import linear_model

    ### LinearRegression fits a linear model
    # with coefficients w = (w1, …, wp) to minimize the residual sum of squares
    # between the observed targets in the dataset, and the targets predicted by the linear approximation.

    # shape the data as required by sklearn
    # make a column array
    x_true = np.reshape(x_true, newshape=(-1, 1))

    # what model will I try to fit, a model with coefficients w1,...wp?
    # there is an intercept in this data
    # the data is assumed to be not centred
    # regular LS
    reg = linear_model.LinearRegression(fit_intercept=True)

    # fit the model
    reg.fit(x_true, y_true)
    print(f"Linear Regression Y={reg.coef_[0]}x +{reg.intercept_}")

    # plotting
    plt.figure(2)
    plt.plot(x_true, y_true, 'o')
    plt.plot(x_true, reg.coef_ * x_true + reg.intercept_)
    plt.title("Linear Regression using SciKit")
    plt.legend(["Generated Data", "Line of best fit"])
    plt.text(60, 25, 'y = ' + '{:.2f}'.format(reg.intercept_) + ' + {:.2f}'.format(reg.coef_[0]) + 'x', size=10)

    # Regularisation
    # is used to reduce the impact of insignificant features
    # avoids over-fitting by penalising the regression coefficients of high value.
    # More specifically, It decreases the parameters and shrinks (simplifies) the model

    ###Using ridge regression
    # Linear least squares with l2 regularization
    # alpha - regularization strength
    # imposes a penalty to the loss function: squared magnitude
    # minimise the objective function
    # usually used for complex systems
    reg = linear_model.Ridge(alpha=.5)

    # fit the created model
    reg.fit(x_true, y_true)
    print(f"Ridge Regression Y={reg.coef_[0]}x +{reg.intercept_}")

    # plotting
    plt.figure(3)
    plt.plot(x_true, y_true, 'o')
    plt.plot(x_true, reg.coef_ * x_true + reg.intercept_)
    plt.title("Ridge Regression using SciKit")
    plt.legend(["Generated Data", "Line of best fit"])
    plt.text(60, 25, 'y = ' + '{:.2f}'.format(reg.intercept_) + ' + {:.2f}'.format(reg.coef_[0]) + 'x', size=10)

    ###Using lasso regression
    # l1 regularisation
    # imposes a penalty to the loss function: absolute value of magnitude
    # The key difference between these techniques is that Lasso shrinks the less important feature’s coefficient
    # to zero thus, removing some feature altogether.
    # So, this works well for feature selection in case we have a huge number of features
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(x_true, y_true)
    print(f"Lasso Regression Y={reg.coef_[0]}x +{reg.intercept_}")

    # plotting
    plt.figure(4)
    plt.plot(x_true, y_true, 'o')
    plt.plot(x_true, reg.coef_ * x_true + reg.intercept_)
    plt.title("Lasso Regression using SciKit")
    plt.legend(["Generated Data", "Line of best fit"])
    plt.text(60, 25, 'y = ' + '{:.2f}'.format(reg.intercept_) + ' + {:.2f}'.format(reg.coef_[0]) + 'x', size=10)

    ###Using lassolars regression
    # adds the least angle regression to lasso regression
    # forward step-wise for high dimensional data
    # finds the feature most correlated with the target
    # There may be more than one attribute that has the same correlation. In this scenario,
    # LARS averages the attributes and proceeds in a direction that is at the same angle to the attributes.
    # This is exactly why this algorithm is called Least Angle regression.
    # Basically, LARS makes leaps in the most optimally calculated direction without overfitting the model.
    reg = linear_model.LassoLars(alpha=.1, normalize=False)
    reg.fit(x_true, y_true)
    print(f"LassoLars Regression Y={reg.coef_[0]}x + {reg.intercept_}")

    # plotting
    plt.figure(5)
    plt.plot(x_true, y_true, 'o')
    plt.plot(x_true, reg.coef_ * x_true + reg.intercept_)
    plt.title("Lasso Regression using SciKit")
    plt.legend(["Generated Data", "Line of best fit"])
    plt.text(60, 25, 'y = ' + '{:.2f}'.format(reg.intercept_) + ' + {:.2f}'.format(reg.coef_[0]) + 'x', size=10)

    ###Baysian
    # conditional probability based
    # linear regression
    # to survive insufficient and poorly distributed data
    reg = linear_model.BayesianRidge()

    # make a column array
    x_true = np.reshape(x_true, newshape=(-1, 1))

    reg.fit(x_true, y_true)
    print(f"Bayesian Regression Y={reg.coef_[0]}x + {reg.intercept_}")

    # plotting
    plt.figure(6)
    plt.plot(x_true, y_true, 'o')
    plt.plot(x_true, reg.coef_ * x_true + reg.intercept_)
    plt.title("Bayesian Regression using SciKit")
    plt.legend(["Generated Data", "Line of best fit"])
    plt.text(60, 25, 'y = ' + '{:.2f}'.format(reg.intercept_) + ' + {:.2f}'.format(reg.coef_[0]) + 'x', size=10)

    ####  Polynomial
    # Optimize some model function.
    import scipy.optimize as optimization

    # generate polynomial dataset (2nd order)
    x_true, y_true = generatePolyDataset(plot=False)

    # starting values for the parameters
    ## why???????
    # x0 = np.zeros(3)

    # non-linear curve fitting through optimisation
    # mapping function
    # first array is the coefficients (order increases)
    # then covariance
    val = optimization.curve_fit(func, x_true, y_true)
    print(f"Fn Optimisation Y={val[0][2]}x^2 +{val[0][1]}x+{val[0][0]}")

    # y values based on the coefficients
    y = []
    for i in range(len(x_true)):
        y.append(val[0][2] * x_true[i] ** 2 + val[0][1] * x_true[i] + val[0][0])

    # plotting
    plt.figure(7)
    plt.plot(x_true, y_true, 'o')
    plt.plot(x_true, y)
    plt.title("Function Optimisation using Scipy")
    plt.legend(["Generated Data", "Line of best fit"])
    plt.text(60, 980, 'y = ' + '{:.2f}'.format(val[0][2]) + 'x^2' + ' + {:.2f}'.format(val[0][1]) + 'x' +
             ' + {:.2f}'.format(val[0][0]), size=10)

    ###Numpy poly fit.
    # just a regular polynomial fit using numpy
    # the least square poly fit
    # outputs only one array (order decreases)
    coefs = np.polyfit(x_true, y_true, 2)
    print(f"Fn NP Polyfit Y={coefs[0]}x^2 +{coefs[1]}x+{coefs[2]}")

    # y values based on the coefficients
    y = []
    for i in range(len(x_true)):
        y.append(coefs[0] * x_true[i] ** 2 + coefs[1] * x_true[i] + coefs[2])

    # plotting
    plt.figure(8)
    plt.plot(x_true, y_true, 'o')
    plt.plot(x_true, y)
    plt.title("PolyFit using Numpy")
    plt.legend(["Generated Data", "Line of best fit"])
    plt.text(60, 980, 'y = ' + '{:.2f}'.format(coefs[0]) + 'x^2' + ' + {:.2f}'.format(coefs[1]) + 'x' +
             ' + {:.2f}'.format(coefs[2]), size=10)

    print("Done")
    plt.show()
