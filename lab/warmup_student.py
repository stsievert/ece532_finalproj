# Emulate Matlab's environment
from pylab import *
from scipy.io import loadmat
import seaborn as sns

__author__ = {'Scott Sievert': ['stsievert@me.com', 'scottsievert.com']}

def soft_threshold(z, threshold):
    r"""
    :param z: The input vector
    :param threshold: The threshold (\lambda in the case of LASSO).
    :returns: See below.

    This implements

        x_hat = \arg \min_x ||z - x||_2^2 + threshold*||x||_1

    This function is the proximity operator for the LASSO algorithm (aka L1
    regularizaiton)
    """
    x = abs(z) - threshold
    x[x < 0] = 0
    return sign(z) * x


def grad(A, x, y):
    """
    The least squares gradient.

    :param A: The input measurement matrix.
    :param x: The point the gradient is taken at.
    :param y: The observations.
    :returns: The gradient of ||Ax - y||_2^2 at the point x_k.
    """
    return 2*A.T@(A@x - y)


def prox(z, lambda1, lambda2, tau):
    r"""
    Performs the proximal operator for elastic net.

    :param z: The estimate from the gradient function.
    :param lambda1: The scalar that says how important the L1 regularizaiton is.
    :param lambda2: The scalar that says how important the L2 regularizaiton is.
    :param tau: The step size used for the gradient step.
    :returns: x_hat. This function performs

        x_hat = \arg \min_x ||z - x||_2^2 + \lambda_1 ||x||_1 + \lambda_2||x||_2

    This function is the proximity operator for the elastic net problem
    formulation.
    """
    raise NotImplementedError, "Implement this function!"


def fMRI_data():
    """
    :returns: The breast cancer data as (X, y). y are the observations (-1 or 1
    depending on cancer/cancer free) and X are the level of gene expression for
    each patient.
    """
    data = loadmat('./BreastCancer.mat')
    return data['X'], data['y'].flat[:]


if __name__ == "__main__":
    A, y = fMRI_data()

    # parameters to tune (we found good results with these parameters)
    tau = 1 / norm(A)**2
    lambda1, lambda2 = 1e2, 1e-1

    # implement a for-loop here

    figure()
    plot(x_k)
    show()
