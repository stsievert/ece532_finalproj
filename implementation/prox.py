""" Regularization Term Mixing Classes """

import numpy as np

###############################################################################


class LassoMixin(object):
    """ Mixin Class For Models Using L1 Norm Regularization """
    @staticmethod
    def penalty(x, tau):
        """ Computes the penalty incurred by the L1 norm at the current search
        point x.

        L(x) = ||x||_1^1

        :param x: Regression weight vector : R^p
        :param tau: L1 norm regularization strength : R+^1
        :return: Incurred regularization penalty L(x) : R^1
        """
        return tau * np.sum(np.absolute(x))  # L1 Norm Penalty

    @staticmethod
    def prox(x, tau):
        """ The prox operator For the L1 Norm aka. soft thresholding:

        x_i = x_i - tau for x_i > tau
        x_i = x_i + tau for x_i < -tau
        x_i = 0            for |x_i| <= tau

        :param x: Regression weight vector : R^p
        :param tau: L1 norm regularization strength : R+^1
        :return: Regression weight vector : R^p
        """
        out = np.zeros(x.size)
        pos_idx = x > tau
        neg_idx = x < -tau
        out[pos_idx] = x[pos_idx] - tau
        out[neg_idx] = x[neg_idx] + tau
        return out

    @staticmethod
    def _prox(x, tau):
        """ The prox operator For the L1 Norm aka. soft thresholding:

        x_i = x_i - tau for x_i > tau
        x_i = x_i + tau for x_i < -tau
        x_i = 0            for |x_i| <= tau

        :param x: Regression weight vector : R^p
        :param # TODO: au: L1 norm regularization strength : R+^1
        :return: Regression weight vector : R^p
        """
        # Alternative Schema for calculation
        return np.maximum(x - tau, 0) - np.maximum(-x - tau, 0)


###############################################################################


class RidgeMixin(object):
    """ Mixin Class For Models Using L2 Norm Regularization """
    @staticmethod
    def penalty(x, tau):
        """ Computes the penalty incurred by the L2 norm at the current search
        point x.

        L(x) = .5 * tau * ||x||_2^2

        :param x: Regression weight vector : R^p
        :param tau: L2 norm regularization strength : R+^1
        :return: Incurred regularization penalty L(x) : R^1
        """
        return .5 * tau * x.dot(x)  # Squared L2 Norm Penalty

    @staticmethod
    def prox(x, tau):
        """ The prox operator for the L2 norm.

        x_i =  (1 / (1 + tau)* x_i

        :param x: Regression weight vector : R^p
        :param tau: L2 norm regularization strength : R+^1
        :return: Regression weight vector : R^p
        """
        return (1/(1+tau)) * x

    @staticmethod
    def __gradient(x, tau):
        """ Computes the gradient penalty incurred by ridge regularization at the
        current search point x.

        gradient = tau * x

        :param x: Regression weight vector : R^p
        :param tau: L2 norm regularization strength : R+^1
        :return: Gradient of the ridge penalty : R^p, R^1
        """
        return tau * x  # gradient of Squared L2 Norm

    @staticmethod
    def __hessian(x, tau):
        """ Computes the hessian of the ridge penalty at the current
        search point x.

        hessian = tau * I^(p,p)

        :param x: Regression weight vector : R^p
        :param tau: L2 norm regularization strength : R+^1
        :return: Hessian of the ridge penalty : R^(p,p)
        """
    return tau * np.identity(x.size)  # hessian of L2 Norm


###############################################################################


class ElasticNetMixin(object):
    """ Mixin Class For Models Using L1 + L2 Norm Regularization """
    @staticmethod
    def penalty(x, tau_l1, tau_l2):
        """ Computes the penalty incurred by the elastic net regularization term
        at the current search point x.

        L(x) = .5 * tau_l2 * ||x||_2^2 + tau_l1 * ||x||_1^1

        :param x: Regression weight vector : R^p
        :param tau_l1: L1 norm regularization strength : R+^1
        :param tau_l2: L2 norm regularization strength : R+^1
        :return: Incurred regularization penalty L(x) : R^1
        """
        return l2.penalty(x, tau_l2) + l1.penalty(x, tau_l1)

    @staticmethod
    def prox(x, tau_l1, tau_l2):
        """ Prox operator for the combined l2 + l1 norms.

        x =  (1/(1+tau_l2)) * soft(x, tau_gl1)

        where soft(x, tau) is defined as:
            x_i = x_i - tau for x_i > tau
            x_i = x_i + tau for x_i < -tau
            x_i = 0            for |x_i| <= tau

        :param x: Regression weight vector : R^p
        :param tau_l1: L1 norm regularization strength : R+^1
        :param tau_l2: L2 norm regularization strength : R+^1
        :return: Regression weight vector : R^p
        """
        return l2.prox(l1.prox(x, tau_l1), tau_l2)
