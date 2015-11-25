import warnings
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

import bokeh.plotting as bkp

colors = ['#32CD32', '#FF8C00', '#00BFFF']
#  colors = ['0fce1b', '#f84525', '#25b3f8']
line_colors = ['#00FF00', '#FFA500', '#baf6fb']
#  line_colors = [ '#6fff00','#ffb400', '#F5F5F5']
fill_colors = ['#121200', '#FF1F40', '#00000F']

fig_width = 900
fig_height = 400


###############################################################################
""" Data Set Generation """
###############################################################################


class DataGenerator(object):
    """ """
    def __init__(self, m, n, radius, kind='non_linear'):
        """ """
        self.m = m
        self.n = n
        self.radius = radius
        if kind is 'non_linear':
            self.generate = self.make_circle
        else:
            raise NotImplementedError("Unknown dataset: {}".format(kind))

    def make_circle(self):
        """ Generates a dataset with circular decision boundary"""
        A = 2*np.random.rand(self.m, self.n)-1
        b = np.sign(np.sum(A**2, 1) - self.radius)
        return A, b


class SignalGenerator(object):
    """
    Generates a random signal, blurring matrix and
    noisey, blurred observation to simulate signal reconstrcution
    """
    def __init__(self, n=500, k=30, sigma=.01, proba=.95):
        super(SignalGenerator, self).__init__()
        self.n, self.k, self.m = n, k, n+k-1
        self.sigma, self.proba = sigma, proba
        self.A = self._generate_transform()
        self.x = self.b = None

    def _generate_transform(self):
        """Generates a k-element Averaging Filter Transformation Matrix"""
        A = np.zeros([self.m, self.n])
        idx = np.arange(self.m)
        for i in idx:
            A[i, max(i-self. k+1, 0):min(i+1, self.n)] = np.ones(
                min(i+1, self.n)-max(i-self. k+1, 0)) / self.k
        return A

    def _generate_signal(self):
        """
        Generates a random signal of length n. With probability p, a sample
        will be the same as the previously measured sample, otherwise it
        will be initialized randomly
        """
        x = np.arange(self.n, dtype='float')
        resample = np.random.rand(self.n) >= self.proba
        resample[0] = True  # randomly initialize first sample
        x[resample] = np.random.randn(np.sum(resample))
        for i in x[~resample]:
            x[int(i)] = x[int(i)-1]
        return x

    def _generate_measurement(self):
        """ Generates the observed signal, b, as b = <A,x>+epsilon """
        return self.A.dot(self.x) + (self.sigma*np.random.randn(self.m))

    def generate(self):
        """ Generates a new random signal and noisey measurement"""
        self.x = self._generate_signal()
        self.b = self._generate_measurement()
        return (self.A, self.x, self.b)


###############################################################################
""" Experiment Automation """
###############################################################################


class ModelTester(object):
    """ Uses evaluate the performance of a model used to perform signal
    reconstruction over a spectrum of noise levels and regularization
    strengths
    """
    def __init__(self, params, sigmas, n=500, k=30, repetitions=100):
        """
        params - 1D np.array - Range of regularization parameters
        sigmas - 1D np.array - Range of noise variances parameters
        n - int - number of samples in the generated signal
        k - int - size in samples of the boxcar averaging filter
        repetitions - int - # signal reconstructions for averaging
        """
        self.sigmas, self.params = sigmas, params
        self.n, self.k = n, k
        self.repetitions = repetitions
        self.mse = pd.DataFrame(np.zeros([sigmas.size, params.size]),
                                columns=params, index=sigmas)

    def _evaluate(self, estimator, generator):
        """
        --- Evaluates Avg MSE Over A Single (Param, Sigma) Pair ---
            estimator - object - Model initialized with regularization param
            generator - object - Generator initialized with signal params
        """
        return np.mean([np.mean(np.power(estimator.estimate(A, b) - x, 2))
                        for A, x, b in[generator.generate()
                                       for _ in range(self.repetitions)]])

    def evaluate(self, Estimator, Generator):
        """ Evaluates Model Performance Over Pairs Of Regularization and
        Noise Parameters

        Estimator - class - estimation model implementing the estimate method
        Generator - class - signal generator implementing the generate method
        """
        assert hasattr(Estimator, 'estimate'),\
            "Estimator must implement the estimate method"
        assert hasattr(Generator, 'generate'),\
            "Generator must implement the generate method"
        for param in self.params:
            for sigma in self.sigmas:
                self.mse[param][sigma] = self._evaluate(
                    Estimator(param), Generator(self.n, self.k, sigma))
        return self.mse


class CrossValidator(object):
    """ Evaluates the performance of a model over a series of parameters """
    def __init__(self, A, b, k_folds, **kwargs):
        """
        A - np.array [m,n] - design matrix
        b - np.array [m] - observations
        k - int - number of folds over which to divide data
        """
        self.A, self.b, self.k_folds = A, b, k_folds
        self.max_outer = kwargs.get('max_outer', k_folds)
        self.max_inner = kwargs.get('max_inner', k_folds-1)
        self.cv = StratifiedKFold(y=b, n_folds=k_folds, shuffle=True)

    def _score(self, estimator, train, test):
        """Evaluates Model Performance for a single parameter fold pair
        estimator - object - Initialized model
        train - np.array [m,n] - Set of indices used to fit model
        test - np.array [m,n] - Set of indices used to evaluate fit model
        """
        b = estimator.fit(self.A[train], self.b[train]).predict(self.A[test])
        return accuracy_score(self.b[test], b)

    def evaluate(self, Estimator, params):
        """ Evaluate Model Performance Through Double Layer Cross Validation
        Estimator - class - estimation model implementing the estimate method
        params - 1D np.array - Range of regularization parameters
        """
        assert hasattr(Estimator, 'fit'),\
            "Estimator must implement the fit method"
        assert hasattr(Estimator, 'predict'),\
            "Estimator must implement the predict method"
        # Initialize Estimators
        models = [Estimator(param) for param in params]
        ac = list()
        for idx, (search, hold_out) in enumerate(self.cv):
            if idx >= self.max_outer:
                break
            cv = StratifiedKFold(y=self.b[search], n_folds=self.k_folds-1)
            for jdx, (train, test) in enumerate(cv):
                if jdx >= self.max_inner:
                    break
                scores = [self._score(model, train, test) for model in models]
            ac.append(self._score(models[np.argmax(scores)], search, hold_out))
        return np.mean(ac)


#############################################################################
""" Plotting Tools """
#############################################################################


class Plotter(object):
    """Helper Object Used To Make Bokeh Signal Plot For iPython Notebooks"""
    def __init__(self,
                 fig_width=900,
                 fig_height=400,
                 colors=['#FF8C00', '#32CD32', '#00BFFF',
                         '#a206e0', '#e6ea06', '#888686'],
                 fill_colors=['#FFA500', '#00FF00', '#5bfbf8',
                              '#fb5bdb', '#f6fa33', '#ebeaea'],
                 edge_colors=['#EA1F00', '#0d990b', '#011BFF',
                              '#533333', '#A0A100', '#111113']):
        super(Plotter, self).__init__()
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.colors = colors
        self.fill_colors = fill_colors
        self.edge_colors = edge_colors
        self.color_counter = 0
        self.num_colors = len(colors)

    def show_notebook(self, fig):
        bkp.output_notebook()
        bkp.show(fig)

    def _get_color(self, color, edge, fill, inc=True):
        """ """
        if color is None:
            color = self.color_counter
        if edge:
            color = self.edge_colors[color % self.num_colors]
        elif fill:
            color = self.fill_colors[color % self.num_colors]
        else:
            color = self.colors[color % self.num_colors]
        if inc:
            self.color_counter += 1
        return color

    def get_fig(self, **kwargs):
        title = kwargs.get('title', 'Signal Plot')
        width = kwargs.get('width', self.fig_width)
        height = kwargs.get('height', self.fig_height)
        x_label = kwargs.get('x_label', 'x')
        y_label = kwargs.get('y_label', 'y')
        self.color_counter = 0
        return bkp.figure(title=title,
                          width=width,
                          height=height,
                          x_axis_label=x_label,
                          y_axis_label=y_label)

    def add_signal(self, fig, signal, legend,
                   offset=0,
                   color=None,
                   bold=False,
                   fade=False,
                   line_width=1):
        color = self._get_color(color, bold, fade)
        fig.line(x=np.arange(signal.size) + offset,
                 y=signal,
                 color=color,
                 legend=legend,
                 line_width=line_width)
        return fig

    def add_curve(self, fig, x, y, legend,
                  color=None,
                  bold=None,
                  fade=None,
                  line_width=1):
        color = self._get_color(color, bold, fade)
        fig.line(x=x,
                 y=y,
                 color=color,
                 legend=legend,
                 line_width=line_width)
        return fig

    def add_fill(self, fig, signals, legend,
                 offsets=[0, 0],
                 color=None,
                 bold=False,
                 fade=True,
                 alpha=.25,
                 line_width=.2):
        color = self._get_color(color, bold, fade)
        for idx in range(min(signals[0].size, signals[1].size)-1):
            fig.patch([idx, idx+1, idx + 1, idx],
                      [signals[0][idx - offsets[0]],
                       signals[0][idx + 1 - offsets[0]],
                       signals[1][idx + 1 - offsets[1]],
                       signals[1][idx - offsets[1]]],
                      color=color,
                      alpha=alpha,
                      line_width=line_width,
                      legend=legend)
        return fig

    def add_data(self, fig, x, y, legend, **kwargs):
        """ """
        radius = kwargs.pop('radius', .01)
        alpha = kwargs.pop('alpha', .6)
        color = kwargs.pop('color', self.color_counter)
        fill_fade = kwargs.pop('fill_fade', True)
        fill_bold = kwargs.pop('fill_bold', False)
        line_fade = kwargs.pop('fill_fade', False)
        line_bold = kwargs.pop('fill_bold', False)
        fill_color = self._get_color(color, fill_bold, fill_fade, inc=False)
        line_color = self._get_color(color, line_bold, line_fade)
        fig.circle(x=x, y=y, radius=radius,
                   fill_color=fill_color, fill_alpha=alpha,
                   line_color=line_color, legend=legend)
        return fig

    def add_boundary(self, fig, w, legend, min_x=-1, max_x=1, **kwargs):
        """ Adds a dcision boundary to fig """
        color = kwargs.pop('color', None)
        bold = kwargs.pop('bold', False)
        fade = kwargs.pop('fade', False)
        line_width = kwargs.pop('line_width', 1)
        x, y = self.decision_boundary(w, min_x=min_x, max_x=max_x)
        #  color = self._get_color(color, bold, fade)
        self.add_curve(fig, x=x, y=y, legend=legend,
                       line_width=line_width, color=color,
                       bold=bold, fade=fade)
        return fig

    def signal_plot(self, signal, legend, **kwargs):
        fig = self.get_fig(**kwargs)
        if len(signal.shape) > 1:
            for sig, leg in zip(signal, legend):
                self.add_signal(fig, sig, leg)
        else:
            self.add_signal(fig, signal, legend)
        return fig

    def data_plot(self, A, b, **kwargs):
        """Creates a New Plot containing data points and decision boundaries"""
        # Generate Figure
        fig = self.get_fig(**kwargs)
        # Data Points
        classes = kwargs.pop('classes',
                             [str(label) for label in np.unique(b)])
        cdx = kwargs.pop('color', 0)
        for idx, label in enumerate(np.unique(b)):
            mask = b == label
            self.add_data(fig, A[mask, 0], A[mask, 1], classes[idx],
                          color=cdx+idx)
        # Decision Boudaries
        weights = kwargs.pop('weights', None)
        if weights is None:
            return fig
        if type(weights) is np.ndarray:
            weights = [weights]  # change single array to list
        titles = kwargs.pop('titles',
                            ['w ' + str(idx) for idx, _ in enumerate(weights)])
        for jdx, (w, title) in enumerate(zip(weights, titles)):
            self.add_boundary(fig, w, title, bold=True, color=cdx+idx+jdx,
                              min_x=np.min(A[:, 0]), max_x=np.max(A[:, 0]))
        return fig

    def decision_boundary(self, w, min_x, max_x):
        """Accepts weight vector w in form w_x, w_y, w_0 return x, y vectors"""
        if np.size(w) < 3:
            w = np.append(w, np.zeros(1))
        x = np.array([min_x, max_x])
        y = -1 * ((w[0] * x) - w[2]) / w[1]
        return x, y

#############################################################################
""" Prox Gradient Algoirhtms """
#############################################################################


class BaseProxGradient(object):
    """   """
    def __init__(self,
                 lambda_,
                 tau,
                 max_iter,
                 delta,
                 offset,
                 regularize=True):
        """   """
        self.lambda_ = lambda_
        self.tau = tau
        self.max_iter = max_iter
        self.delta = delta
        self.a_hat = None
        self.regularize = regularize
        if offset:
            self.offset = lambda x: -1
        else:
            self.offset = lambda x: x.size

    def fit(self, X, y, **kwargs):
        """   """
        Bk = kwargs.get("B0", np.zeros(X.shape[1]))
        for iter in range(self.max_iter):
            Bk_ = Bk
            Bk = Bk - (2 * self.tau * self.grad(X, y, Bk))
            Bk = self.prox(Bk)
            if (np.linalg.norm(Bk - Bk_) <= self.delta):
                self.B_hat = Bk
                return self
        warnings.warn("Algorithm never converged", RuntimeWarning)
        self.B_hat = Bk
        return self

    def predict(self, X):
        """   """
        return X.dot(self.B_hat)


class Lasso(BaseProxGradient):
    """   """
    def __init__(self,
                 lambda_,
                 tau=1e-5,
                 max_iter=2000,
                 delta=1e-5,
                 offset=False):
        """   """
        super().__init__(lambda_=lambda_,
                         tau=tau,
                         max_iter=max_iter,
                         delta=delta,
                         offset=offset)

    def fit(self, X, y, **kwargs):
        """   """
        assert self.tau < 1 / np.linalg.norm(X),\
            "Step size is set too large to guarantee convergence"
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """   """
        labels = np.sign(super().predict(X))
        labels[labels == 0] = -1
        return labels

    def grad(self, X, y, Bk):
        """Returns the gradient of the squared error loss function"""
        return X.T.dot((X.dot(Bk)-y)) / X.shape[0]

    def prox(self, x):
        """ The prox operator For the L1 Norm aka. soft thresholding:

        x_i = x_i - tau for x_i > tau * lambda
        x_i = x_i + tau for x_i < -tau * lambda
        x_i = 0            for |x_i| <= tau * lambda

        :param x: Regression weight vector : R^p
        :param tau: Step size
        :param lambda_: L1 norm regularization strength : R+^1
        :return: Regression weight vector : R^p
        """
        x[:self.offset(x)] = np.zeros(x[:self.offset(x)].size)
        pos_idx = x[:self.offset(x)] > self.lambda_ * self.tau
        neg_idx = x[:self.offset(x)] < -self.lambda_ * self.tau
        x[pos_idx] = x[pos_idx] - self.lambda_ * self.tau
        x[neg_idx] = x[neg_idx] + self.lambda_ * self.tau
        return x


class Linear_SVM(BaseProxGradient):
    """ Assuming offset & using l2 norm in gradient without offset"""
    def __init__(self,
                 lambda_=.1,
                 tau=.003,
                 max_iter=20000,
                 delta=1e-6,
                 offset=False,
                 regularize=True):
        """ """
        super().__init__(lambda_=lambda_,
                         tau=tau,
                         max_iter=max_iter,
                         delta=delta,
                         offset=offset,
                         regularize=regularize)

    def fit(self, X, y, **kwargs):
        """   """
        # assert self.tau < 1 / np.linalg.norm(X),\
        #     "Step size is set too large to guarantee convergence"
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """   """
        labels = np.sign(super().predict(X))
        labels[labels == 0] = -1
        return labels

    def grad(self, A, y, x):
        """Returns the gradient of the hinge function"""
        z = y * A.dot(x)  # decision value for each observation
        grad_x = -1*A[z < 1].T.dot(y[z < 1])
        # Gradient normalized by the num obs
        return grad_x / y.size

    def prox(self, x):
        """ The prox operator for the L2 norm. assuming last entry is constant offset.

        x_i =  (1 / (1 + tau)) * x_i

        :param x: Regression weight vector : R^p
        :param tau: L2 norm regularization strength : R+^1
        :return: Regression weight vector : R^p
        """
        if self.regularize:
            x[:self.offset(x)] /= (1 + 2 * self.tau * self.lambda_)
        return x


class L1LossClassifier(BaseProxGradient):
    """ No regularization term"""
    def __init__(self,
                 lambda_=1,
                 tau=.003,
                 max_iter=20000,
                 delta=1e-6,
                 offset=False,
                 regularize=True):
        """ """
        super().__init__(lambda_=lambda_,
                         tau=tau,
                         max_iter=max_iter,
                         delta=delta,
                         offset=offset,
                         regularize=regularize)

    def fit(self, X, y, **kwargs):
        """   """
        # assert self.tau < 1 / np.linalg.norm(X),\
        #     "Step size is set too large to guarantee convergence"
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """   """
        labels = np.sign(super().predict(X))
        labels[labels == 0] = -1
        return labels

    def predict_proba(self, X, **kwargs):
        return super().predict(X)

    def grad(self, A, y, x):
        """Returns the gradient of the hinge function"""
        z = y - A.dot(x)  # Error for each observation
        grad_x = -1 * A.T.dot(np.sign(z))
        # Gradient normalized by the num obs
        return grad_x / y.size

    def prox(self, x):
        """ The prox operator for the L2 norm. assuming last entry is constant offset.

        x_i =  (1 / (1 + tau)) * x_i

        :param x: Regression weight vector : R^p
        :param tau: L2 norm regularization strength : R+^1
        :return: Regression weight vector : R^p
        """
        if self.regularize:
            x[:self.offset(x)] /= (1 + 2 * self.tau * self.lambda_)
        return x


class LeastSquaresClassifier(BaseProxGradient):
    """ No regularization term"""
    def __init__(self,
                 lambda_=1,
                 tau=.003,
                 max_iter=20000,
                 delta=1e-6,
                 offset=False,
                 regularize=True):
        """ """
        super().__init__(lambda_=lambda_,
                         tau=tau,
                         max_iter=max_iter,
                         delta=delta,
                         offset=offset,
                         regularize=regularize)

    def fit(self, X, y, **kwargs):
        """   """
        assert self.tau < 1 / np.linalg.norm(X),\
            "Step size is set too large to guarantee convergence"
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """   """
        labels = np.sign(super().predict(X))
        labels[labels == 0] = -1
        return labels

    def predict_proba(self, X, **kwargs):
        return super().predict(X)

    def grad(self, X, y, Bk):
        """Returns the gradient of the squared error loss function"""
        return X.T.dot((X.dot(Bk)-y)) / X.shape[0]

    def prox(self, x):
        """ The prox operator for the L2 norm. assuming last entry is constant offset.

        x_i =  (1 / (1 + tau)) * x_i

        :param x: Regression weight vector : R^p
        :param tau: L2 norm regularization strength : R+^1
        :return: Regression weight vector : R^p
        """
        if self.regularize:
            x[:self.offset(x)] /= (1 + 2 * self.tau * self.lambda_)
        return x

############################################################################
""" Kenel Algorithms """
############################################################################


class BaseKernelClassifier(object):
    """ """
    def __init__(self, kernel, lambda_, tau, max_iter, delta, offset):
        """   """
        self.kernel = kernel
        self.lambda_ = lambda_
        self.tau = tau
        self.max_iter = max_iter
        self.delta = delta
        self.a_hat = None
        self.X_fit = None
        if offset:
            self.offset = lambda x: -1
        else:
            self.offset = lambda x: x.size

    def fit(self, X, y, **kwargs):
        """   """
        self.X_fit = X
        K = self.kernel.make(X)
        ak = kwargs.get("a0", np.zeros(X.shape[0]))
        for iter in range(self.max_iter):
            ak_ = ak
            ak = ak - (2 * self.tau * self.grad(K, y, ak))
            if (np.linalg.norm(ak - ak_) <= self.delta):
                self.a_hat = ak
                return self
        warnings.warn("Max Iter Reached Before Convergence", RuntimeWarning)
        self.a_hat = ak
        return self

    def predict(self, X):
        """   """
        return np.array(
            [self.kernel.predict(self.X_fit, self.a_hat, x) for x in X])


class KernelSVMClassifier(BaseKernelClassifier):
    """ """
    def __init__(self,
                 kernel,
                 lambda_=1e-5,
                 tau=1e-5,
                 max_iter=20000,
                 delta=1e-6,
                 offset=False):
        """ """
        super().__init__(kernel,
                         lambda_=lambda_,
                         tau=tau,
                         max_iter=max_iter,
                         delta=delta,
                         offset=offset)

    def fit(self, X, y, **kwargs):
        """   """
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """   """
        labels = np.sign(super().predict(X))
        labels[labels == 0] = -1
        return labels

    def predict_proba(self, X, **kwargs):
        return super().predict(X)

    def grad(self, K, y, ak):
        """Returns the gradient of the hinge loss + l2 norm"""
        Ka = K.dot(ak)  # precompute
        z = y * Ka  # decision value for each observation
        grad = (-1*K[z < 1].T.dot(y[z < 1])) / y.size   # gradient of hinge
        l2 = (2 * self.lambda_ * Ka)  # gradient of l2
        # Don't regularize offset dimension
        grad[:self.offset(ak)] = grad[:self.offset(ak)] + l2[:self.offset(ak)]
        # Gradient normalized by the num obs
        return grad


class KernelLeastSqauresClassifier(BaseKernelClassifier):
    """ """
    def __init__(self,
                 kernel,
                 lambda_=1e-5,
                 tau=1e-5,
                 max_iter=20000,
                 delta=1e-6,
                 offset=False):
        """ """
        super().__init__(kernel,
                         lambda_=lambda_,
                         tau=tau,
                         max_iter=max_iter,
                         delta=delta,
                         offset=offset)

    def fit(self, X, y, **kwargs):
        """   """
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """   """
        labels = np.sign(super().predict(X))
        labels[labels == 0] = -1
        return labels

    def predict_proba(self, X, **kwargs):
        return super().predict(X)

    def grad(self, K, y, ak):
        """Returns the gradient of the mean squared error + l2 norm"""
        Ka = K.dot(ak)  # precompute
        grad = K.dot((Ka - y)) / y.size  # gradient of mse
        l2 = 2 * self.lambda_ * Ka  # gradient of l2
        # Don't regularize offset dimension
        grad[:self.offset(ak)] = grad[:self.offset(ak)] + l2[:self.offset(ak)]
        return grad


class BaseKernel(object):
    """ """
    def __init__(self):
        """ """
        pass

    def make(self, X):
        """ """
        M = X.shape[0]
        K = np.zeros([M, M])
        for idx in np.arange(M):
            # Fill Diagonal Elements
            K[idx, idx] = self.func(X[idx, :], X[idx, :])
            for jdx in np.arange(idx+1, M, 1):
                # Fill Off Diagonal-Symetric Elements
                K[idx, jdx] = self.func(X[idx, :], X[jdx, :])
                K[jdx, idx] = K[idx, jdx]
        return K

    def predict(self, X, a, x):
        """ """
        k = np.array([self.func(xi, x) for xi in X])
        return a.dot(k)


class PolynomialKernel(BaseKernel):
    """ """
    def __init__(self, degree=2):
        """ """
        super().__init__()
        self.degree = degree

    def func(self, xi, xj):
        """ """
        return np.power(xi.dot(xj) + 1, self.degree)


class GaussianKernel(BaseKernel):
    """ """
    def __init__(self, sigma=1):
        """ """
        super().__init__()
        self.sigma = sigma

    def func(self, xi, xj):
        """ """
        return np.exp((-.5 * np.linalg.norm(xi - xj)**2) / self.sigma**2)


__all__ = ["colors",
           "line_colors",
           "fill_colors",
           "fig_height",
           "fig_width",
           "SignalGenerator",
           "ModelTester",
           "CrossValidator",
           "Plotter",
           "BaseProxGradient",
           "BaseKernelClassifier",
           "BaseKernel",
           "Lasso",
           "Linear_SVM",
           "L1LossClassifier",
           "LeastSquaresClassifier",
           "KernelLeastSqauresClassifier",
           "KernelSVMClassifier",
           "GaussianKernel",
           "PolynomialKernel",
           "DataGenerator"]
