import numpy as np
import jax.numpy as jnp
from jax.config import config
import jax.scipy as jsp
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)


class GprJac():
    def __init__(self, X_train, y_train, hyper_param_init):
        self.X_train = X_train
        self.y_train = y_train

        # Hyperparameter
        self.theta = jnp.array([1.0, 1.0, 1e-8])

    def update_data(self, X, y):
        self.X_train.append(X)
        self.y_train.append(y)

    def kernel(self, X1, X2, theta):
        '''
        Isotropic squared exponential kernel
        :param X1: Array of m points (m x d)
        :param X2: Array of n points (n x d)
        :param theta: kernel parameters (2, )
        :return: kernel (m x n)
        '''


        # Compute squared distance
        sq_dist = jnp.sum(X1**2, axis=1).reshape(-1,1) + \
                  jnp.sum(X2**2, axis=1) - \
                  2*jnp.dot(X1, X2.T)

        # Compute Kernel with hyper parameter
        Kernel = theta[1]**2 * jnp.exp(-0.5 / theta[0] **2 *sq_dist)

        return Kernel

    def dkernel(self, X1, X2, theta):
        '''
        Jacobian of kernel
        :param X1:
        :param X2:
        :param theta:
        :return:
        '''


        n = X1.shape[0]
        m = X2.shape[0]

        diff = np.zeros((n,m))

        # Compute squared distance
        sq_dist = jnp.sum(X1**2, axis=1).reshape(-1, 1) + \
                  jnp.sum(X2**2, axis=1) - \
                  2*jnp.dot(X1, X2.T)

        for i in range(n):
            for j in range(m):
                diff[i,j] = X1[i,:] - X2[j,:]

        dKernel = 1/theta[0]**2 * diff * jnp.exp(-0.5 / theta[0] **2 * sq_dist)

        return dKernel


    def posterior(self, X_s):
        '''
        Computes the posterior distribution from m training m training data X_train, y_train
        and n new inputs X_s
        :param X_s: Test points (n x d)
        :param X_train: trained input (m x d)
        :param y_train: (m x 1)
        :param l: characteristic length
        :param sigma_f: vertical variation parameter
        :param sigma_n: noise parameters
        :return: Posterior mean vector (n x d) and covariance matrix (n x n)
        '''

        # Get kernel
        K = self.kernel(self.X_train, self.X_train, self.theta) +\
            self.theta[2]**2 * jnp.eye(self.X_train.shape[0])

        K_s = self.kernel(self.X_train, X_s, self.theta)

        dK_s = self.dkernel(self.X_train, X_s, self.theta)

        K_ss = self.kernel(X_s, X_s, self.theta) +\
            self.theta[2]**2 * jnp.eye(X_s.shape[0])

        # dK_s = self.dKernel(X_train, X_s, self.theta)

        # Get lower triangular matrix from cholesky
        L = jnp.linalg.cholesky(K)

        # Compute mean given data set and a new input point
        z = jnp.linalg.solve(L, self.y_train)
        alpha = jnp.linalg.solve(L.T, z)
        mu_new = K_s.T.dot(alpha)
        dmu_new = dK_s.T.dot(alpha)

        # Compute covariance
        v = jnp.linalg.solve(L, K_s)
        cov_new = K_ss - v.T.dot(v)

        return mu_new, cov_new, dmu_new

    def nll_func(self, X_train, Y_train, noise):
        '''
        Compute the Negative Log Likelihood of training data X_train and y_train
        :param X_train: training input (m x d)
        :param Y_train: training output (n x d)
        :param noise: known noise level
        :return: Objective to minimize
        '''
        y_train_ = self.y_train.ravel()

        def nll_stable(theta):

            # Kernel
            K = self.kernel(self.X_train, self.X_train, theta) +\
                theta[0]**2 * jnp.eye(self.X_train.shape[0])

            L = jnp.linalg.cholesky(K)

            S1 = jnp.linalg.solve(L, self.y_train)
            S2 = jnp.linalg.solve(L.T, S1)

            return jnp.sum(jnp.log(jnp.diagonal(L))) +\
                0.5 * self.y_train.dot(S2) +\
                0.5 * self.X_train.shape[0] * jnp.log(2*jnp.pi)

        return nll_stable

if __name__ == '__main__':

    X = jnp.arange(-5, 5, 0.1).reshape(-1, 1)
    Y = jnp.cos(X)

    X_train = jnp.array([-4, -3, -2, -1, 0, 1, 2, 3]).reshape(-1, 1)
    Y_train = jnp.cos(X_train)

    hyper_param = jnp.array([1.0, 1.0, 0.0])

    GP = GprJac(X_train, Y_train, hyper_param)

    mu_new, cov_new, dmu_new = GP.posterior(X)

    plt.plot(X, mu_new)
    plt.plot(X, dmu_new)
    plt.grid('True')
    plt.show()