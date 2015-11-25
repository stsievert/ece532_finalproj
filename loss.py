""" Loss Term Mixin Classes """

import numpy as np
from utils import logistic_sigmoid, log_logistic_sigmoid

###############################################################################


class HingeLossMixin(object):
    """ Mixin Class For Models Using Hinge Loss """
    @staticmethod
    def loss(A, x, y):
        """ Computes the value of the hinge loss function at the current
        estimate x.

        L(x) = (1/n) * sum_i{y_i - <A_i,x>}

        :param x: Regression weight vector : R^p
        :param A: Design/Sensing matrix : R^(n,p)
        :param y: Observation label vector : {-1,1}^n
        :return: Normalized sum of the hinge loss function L(x) : R^1
        """
        z = y * A.dot(x)  # decision value for each observation
        cum_loss = np.sum(np.maximum(1-z, np.zeros(z.size)))  # cumulative loss
        return np.sum(cum_loss) / y.size  # normalized cumulative loss

    @staticmethod
    def gradient(A, x, y):
        """ Computes the gradient of the hinge loss function at the current
        estimate x.

        grad_x = (1/n) * sum_i{(-y_i * A_i)} for all z_i < 1 : R^p
        z_i = y_i(<A_i,x>) : R^1

        :param x: Regression weight vector : R^p
        :param A: Design/Sensing matrix : R^(n,p)
        :param y: Observation label vector : {-1,1}^n
        :return: Normalized gradient grad_x : R^p
        """
        z = y * A.dot(x)  # decision value for each observation
        grad_x = -1 * A[z < 1].T.dot(y[z < 1])
        # Gradient normalized by the num obs
        return grad_x / y.size


###############################################################################


class LogisticLossMixin(object):
    """ Mixin Class For Models Using Logistic Loss """
    @staticmethod
    def loss(A, x, y):
        """ Computes the value of the logistic loss function at the current
        estimate x.

        L(x) = (1/n) * sum_i{-log(sigmoid(z_i))}
        z_i = y_i(<A_i,x>) : R^1

        :param x: Regression weight vector : R^p
        :param A: Design/Sensing matrix : R^(n,p)
        :param y: Observation label vector : {-1,1}^n
        :return: Normalized sum of the logistic loss function L(x) : R^1
        """
        z = y * A.dot(x)  # decision value for each observation
        cum_loss = np.sum(-log_logistic_sigmoid(z))  # cumulative loss
        return np.sum(cum_loss) / y.size  # normalized cumulative loss

    @staticmethod
    def gradient(A, x, y):
        """ Computes the gradient of the logistic loss function at the current
        estimate x.

        grad_x = (1/n) * sum_i{(y_i * A_i)(sigmoid(z_i) - 1)} : R^p
        z_i = y_i(<A_i,x>) : R^1

        :param x: Regression weight vector : R^p
        :param A: Design/Sensing matrix : R^(n,p)
        :param y: Observation label vector : {-1,1}^n
        :return: Normalized gradient grad_x : R^p
        """
        z = y * A.dot(x)  # decision value for each observation
        phi = (logistic_sigmoid(z) - 1) * y
        grad_x = A.T.dot(phi)
        # Gradient normalized by the num obs
        return grad_x / y.size

    @staticmethod
    def hessian(A, x, y):
        """ Computes the hessian of the logistic loss function at the current
        estimate x.

        H(x) = (1/n)(<A',D,A>)
        D is the diagonal matrix D_ii=(sigmoid(z_i)(1 - sigmoid(z_i)) : R^(n,n)
        z_i = y_i(<A_i,x>) : R^1

        :param x: Regression weight vector : R^p
        :param A: Design/Sensing matrix : R^(n,p)
        :param y: Observation label vector : {-1,1}^n
        :return: Hessian H(x) : R^(p,p)
        """
        z = y * A.dot(x)  # decision value for each observation
        # transform z:R^n into diag_matrix:R^(n,n)
        D = np.diag(logistic_sigmoid(z))
        return A.T.dot(D).dot(A) / y.size  # normalized hessian


###############################################################################


class SquaredErrorMixin(object):
    """ Mixin Class For Models Using Squared Error Loss """
    @staticmethod
    def loss(A, x, y):
        """ Computes the value of the squared error loss function at the current
        estimate x.

        L(x) = (1/n) * sum_i{(y_i - <A_i,x>)^2}

        :param x: Regression weight vector : R^p
        :param A: Design/Sensing matrix : R^(n,p)
        :param y: Observation label vector : {-1,1}^n
        :return: Normalized sum of the squared error loss function L(x) : R^1
        """
        cum_loss = np.sum(np.power(A.dot(x) - y, 2))  # cumulative loss
        return np.sum(cum_loss) / y.size  # normalized cumulative loss

    @staticmethod
    def gradient(A, x, y):
        """ Computes the gradient of the squared error loss function at the current
        estimate x.

        grad_x = (1/n) * <A^T,(<A,x> - y)> : R^p

        :param x: Regression weight vector : R^p
        :param A: Design/Sensing matrix : R^(n,p)
        :param y: Observation label vector : {-1,1}^n
        :return: Normalized gradient grad_x : R^p
        """
        grad_x = A.T.dot((A.dot(x)-y))
        # Gradient normalized by the num obs
        return grad_x / y.size
