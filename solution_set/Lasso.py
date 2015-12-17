import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

class ProximalGradient(object):
    def __init__(self, A, b, lambda_=0.1, tau=None, iterations=1e3):
        self.A = A
        self.b = b
        self.tau = tau if tau else 1.9 / np.linalg.norm(A)**2
        self.lambda_ = lambda_
        self.iterations = iterations
        self.initial = np.zeros(self.A.shape[1])

    def run(self, initial_x=None, delta=1e-6):
        x_k = self.initial
        for k in range(int(self.iterations)):
            z = x_k - self.tau*self.grad(self.A, x_k, self.b)
            x_k1 = self.prox(z, self.lambda_, self.tau)

            if np.linalg.norm(x_k - x_k1) < delta: break

            x_k = x_k1
        self.iterations = k
        return x_k, self.predict(self.A, x_k)


class Lasso(ProximalGradient):
    def predict(self, A, x):
        return A @ x

    def soft_threshold(self, z, threshold):
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
        return np.sign(z) * x


    def prox(self, z, lambda_, tau):
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
        x = abs(z) - tau*lambda_/2
        x[x < 0] = 0
        return np.sign(z) * x


    def grad(self, A, x, y):
        """
        The least squares gradient.

        :param A: The input measurement matrix.
        :param x: The point the gradient is taken at.
        :param y: The observations.
        :returns: The gradient of ||Ax - y||_2^2 at the point x_k.
        """
        return 2*A.T.dot(A@x - y)

def breast_cancer_data():
    """
    :returns: The breast cancer data as (X, y). y are the observations (-1 or 1
    depending on cancer/cancer free) and X are the level of gene expression for
    each patient.
    """
    from scipy.io import loadmat
    data = loadmat('./BreastCancer.mat')
    return data['X'], data['y'].flat[:]

if __name__ == "__main__":
    A, y = breast_cancer_data()

    A_train, A_test, y_train, y_test = train_test_split(A, y,
                                            train_size=0.8, random_state=42)

    lambda_ = 70
    lasso = Lasso(A_train, y_train, lambda_=lambda_)

    # changing our prediction function
    lasso.predict = lambda A, x: np.sign(A @ x)
    x_hat, y_hat = lasso.run()
    y_test_hat = lasso.predict(A_test, x_hat)

    error = np.sum(np.sign(np.abs(y_test_hat - y_test))) / len(y_test)
    print("{0:0.2f}% accuracy rate on test data for lambda={1}".format(
            ((1-error)*100),                        lambda_))

    sns.set_context('talk')
    fig = plt.figure()
    plt.plot(x_hat)
    plt.xlabel('gene index')
    plt.ylabel('weight')
    plt.title('Gene importance in breast cancer')
    plt.show()
