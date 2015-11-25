""" Helper Functions For Linear Loss Formulations """
import numpy as np


###############################################################################


def log_logistic_sigmoid(z):
    """
    Numerically stable computation for the log of the sigmoid function.
    log(sigmoid(z_i)) = -log(1 + exp(-z_i)) for z_i > 0
    log(sigmoid(z_i)) = z - log(1 + exp(z_i)) for z_i <= 0
    z_i = y_i(<A_i,x> + c) : R^1
    :param z: Decision score vector : R^n
    :param z: vector of decision scores R^n
    :return: Vector of transformed decision scored : R^n
    """

    out = np.empty(z.size, dtype=float)
    pos_idx = z > 0
    out[pos_idx] = -np.log(1 + np.exp(-z[pos_idx]))
    out[~pos_idx] = z[~pos_idx] + np.log(1 + np.exp(z[~pos_idx]))
    return out


def logistic_sigmoid(z):
    """
    Numerically stable computation of the sigmoid function.
    sigmoid(z_i) = 1 / (1 + exp(-z_i)) for z_i > 0
    sigmoid(z_i) = exp(z_i) / (1 + exp(z_i)) for z_i <= 0
    z_i = y_i(<A_i,x> + c) : R^1
    :param z: Decision score vector : R^n
    :return: Vector of transformed decision scored : R^n
    """

    out = np.empty(z.size, dtype=np.float)
    pos_idx = z > 0
    out[pos_idx] = 1 / (1 + np.exp(-z[pos_idx]))
    out[~pos_idx] = np.exp(z[~pos_idx]) / (1 + np.exp(z[~pos_idx]))
    return out
