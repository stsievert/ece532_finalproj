""" Sparsa Proximal Gradient Algorithm Solver  """

from collections import defaultdict
from sklearn.base import BaseEstimator

import numpy as np

###############################################################################


class SpaRSA(BaseEstimator):
    """

    """
    def __init__(self, stop_crit='duality gap', stop_tol=.01, max_iter=1000,
                 min_iter=5, init='zero', acceptance_iter=5, sigma=.01, eta=2,
                 alpha_selection='BB1', alpha_min=1e-30, alpha_max=1e30,
                 alpha_cycle=1, alpha_factor=.8, debias=False,
                 debias_tol=.0001, debias_max_iter=200, debias_min_iter=5,
                 verbose=False):
        """

        :param stop_crit: String (Default: 'duality gap')

        The choice of stopping criterion which determines when the SpaRSA
        algorithm is completed.

        Choose One of:
            'cardinality': Algorithm terminates when the relative change in the
                number of non-zero components of the estimate falls below
                stop_tol.
            'objective': Algorithm terminates when the relative change in the
                objective function falls below stop_tol.
            'duality gap': Algorithm terminates when the relative duality gap
                falls below stop_tol.
            'LCP': Algorithm terminates when the LCP estimate of relative
                distance to the solution falls below stop_tol.
            'loss': Algorithm terminates when the objective function becomes
                equal to or less
            than stop_tol.
            'norm': Algorithm terminates when the norm of the difference
                between two consecutive estimates, divided by the norm of
                either of them falls below stop_tol.

        :param stop_tol: float (Default: .01)

        Threshold to be used for evaluation in the stopping criterion.

        :param max_iter: int (Default: 1000)

        Maximum number if iterations allowed in the main phase of the algorithm

        :param min_iter: int (Default: 5)

        Minimum number of iteration allowed in the main phase of the algorithm.

        :param init: String | 1-D np.array of shape p (Default: 'zero')

        Default initialization schema to be used whenever a warm start estimate
        is not provided as a parameter to the fit() method.

        Choose one of:
            'zero': Initial estimate is the zero vector.
            'random': Initial estimate is vector of random numbers.
            'ATy': Initial estimate calculated as <A.T, y>.

        :param acceptance_iter: int (Default: 5)

        The number of iterations to consider when determining the largest value
        of the objective function to be used in the evaluation of the
        acceptance criterion. This safeguarding process is used to enforce a
        'sufficient decrease' in the objective function for the next iteration.
        When M is 1, 'SpaRSA monotone' is implemented enforcing a monotonic
        decrease in the objective function from iteration to iteration.
        When M is 0, no safeguarding is enforced.

        x_t+1 is accepted if:
            phi(x_t+1) <= phi(x_i) - (sigma/2) * alpha_t * ||x_t+1 - x_t||^2
                i chosen a argmax_i phi(x_i) for i = max(t+1-M, 0),...,t

        :param sigma: float (0,1) (Default: .01

        Constant value used in the safeguarding equation for the acceptance
        criterion (see acceptance_iter documentation). Chosen from (0,1) as a
        value typically close to 0.

        :param eta: float > 1

        Constant factor by which alpha is multiplied within an iteration until
        an iterate meeting the acceptance criterion is found.

        :param alpha_selection: String

        Method by which alpha is chosen from the range [alpha_min, alpha_max]
        at the beginning of each iteration.

        Choose one of:
            'BB1': Barzilai-Borwein spectral approach where alpha is chosen
                   such that alpha*I mimics the Hessian of f(x) over the most
                   recent step.
                        alpha_t = <(s_t)^T, r_t> / <(s_t)^T, s_t>
            'BB2': Barzilai-Borwein spectral approach where alpha is chosen as
                   the inverse of beta, where beta mimics the inverse Hessian
                   over the most recent step.
                        beta_t = <(r_t)^T, r_t> / <(r_t)^T, s_t>
                        alpha_t = beta_t^-1
            'Last': Alpha is chosen based on the successful value of alpha from
                the previous iteration.

        :param alpha_factor: float (Default: .8)

        Factor by which the successful alpha value from the previous iteration
        is reduced when using the 'Last' alpha selection approach.

        :param alpha_min: float (Default: 1e-30)

        Lower bound on the step size alpha which is chosen for each iteration.

        :param alpha_max: float (Default: 1e30)

        Upper bound on the step size alpha which is chosen for each iteration.

        :param alpha_cycle: int >= 1 (Default: 1)

        When using a non-monotone Barzilai-Borwein method, this parameter
        specifies the number of iterations to be performed between
        recalculations of alpha.

        Non-monotone Barzilai-Borwein methods:
            (alpha_selection == 'BB1' || 'BB2') && (acceptance_iter != 1)

        :param debias: Boolean (Default: False)

        Whether or not the solution should be debiased as a post-processing
        step. During debiasing, individual components (&/or groups depending
        on the choice of regularization) are fixed at zero and the objective is
        minimized over the remaining elements.

        :param debias_tol: float (Default: .0001)

        Stopping tolerance for the debiasing phase.

        :param debias_max_iter: int (Default: 200)

        Maximum number of iterations performed during the debiasing phase.

        :param debias_min_iter: int (Defualt: 5)

        Minimum number of iterations performed during the debiasing phase.

        :param verbose: Boolean (Default: False)

        If false, the algorithm works silently. If true, updates are printed to
        stdout.

        :return: instance of self
        """
        self.stop_crit = stop_crit
        self.stop_tol = stop_tol
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.init = init
        self.acceptance_iter = acceptance_iter
        self.sigma = sigma
        self.eta = eta
        self.alpha_selection = self.alpha_selection
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_cycle = alpha_cycle
        self.alpha_factor = alpha_factor
        self.debias = debias
        self.debias_tol = debias_tol
        self.debias_max_iter = debias_max_iter
        self.debias_min_iter = debias_min_iter
        self.verbose = verbose

    def fit(self, A, y, tau, x=None):
        """

        :param A: Design/Sensing matrix : R^(n,p)
        :param y: Observation vector : {-1,1}^n
        :param tau: Regularization parameter : R+^1
        :param x: Warm start weight vector : R^p
        :return:
        """
        # TODO:check if regularization term large enough to yield zero vector
        # TODO: screen for invalid tau and check dimensionality of A, y
        """ Main Loop: Terminate at stopping criterion or maximum iterations"""
        for itr in range(self.max_iter):
            if itr is 0:
                """ Initialization """
                if x is None:  # use specified initialization
                    x = self._initialize(A, y)
                # Enforce matching dimensions
                assert x.size == A.shape, "x_0{0} isn't compatible with" \
                                          " A{1}".format(x.shape, A.shape)
                nz_mask = x != 0  # location of non-zero weights
                l0 = np.sum(nz_mask)  # number of non-zero weights
                alpha = 1  # Initial step size will be tau
                grad = self.gradient(A, x, y)  # gradient of f(x_t)
            else:
                _l0 = l0  # l0_t-1
                _grad = grad  # gradient f(x_t-1)
                grad = self.gradient(A, x, y)  # gradient of f(x_t)
                _alpha = alpha  # alpha_t-1
                alpha = self._next_alpha(_alpha, _grad, grad, _x, x)  # alpha_t
            """ Inner Loop: terminates when acceptance criterion is met """
                x = x
                x = self._line_search(A, x, y, alpha)

    def _initialize(self, A, y):
        """
        Initialize the weight vector according to the selected method

        :param A: Design/Sensing matrix : R^(n,p)
        :param y: Observation vector : {-1,1}^n
        :return: Initialization weight vector x_0 : R^p
        """
        init_methods = defaultdict(lambda: (None, None),  # incorrect param
                                   zero=np.zeros(A.shape[1]),
                                   random=np.random.randn(A.shape[1]),
                                   ATy=A.T.dot(y))
        x_0 = init_methods[self.init]
        assert x_0 is not None, "Invalid parameter init: {0}".format(self.init)
        return x_0

    def _next_alpha(self, _alpha, _grad, grad, _x, x):
        """ Selects the next alpha

        :param _alpha:
        :param _grad:
        :param grad:
        :param _x:
        :param x:
        :return:
        """
        return

    def _phi(self, A, x, y, tau):
        """ Evaluate the objective function phi:
        phi(x) = f(x) + tau * c(x)
        f(x) is the convex, lipschitz continuously differentiable loss function
        c(x) is the convex, possibly non-smooth regularization function

        :param A:
        :param x:
        :param y:
        :return:
        """
        return self.loss(A, x, y) + self.penalty(x, tau)

    def _line_search(self, A, x, y, alpha):
        """ Selects the next search point x_t+1 as the solution to:

        x_t+1 = argmin_z: (1/2)||z - u_t||_2^2 + (tau/alpha_t)c(z)
        where u_t = x_t - (1/alpha_t)(grad_f(x_t))

        :param A:
        :param x:
        :param y:
        :param alpha:
        :return:
        """
        return
