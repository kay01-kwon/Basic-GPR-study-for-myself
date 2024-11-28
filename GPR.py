import numpy as np
import seaborn as sn
from scipy.spatial import distance
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D
def kernel(X1, X2, l = 1.0, sigma_f = 1.0):
    """
    Istropic squared exponential kernel

    :param X1: Array of m points (m x d)
    :param X2: Array of n points (n x d)
    :param sigma_f: vertical variance

    :return: (m x n) kernel matrix
    """

    sqdist = distance.cdist(X1, X2, metric = 'sqeuclidean')
    return sigma_f**2 * np.exp(-0.5/l**2*sqdist)

def plot_gp(mu, cov, X, Y, X_train = None, Y_train = None, samples=[]):
    X_ = X.ravel()
    mu_ = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X_, mu_ + uncertainty, mu_ - uncertainty, alpha = 0.2, color = 'blue')
    plt.plot(X_, mu_, label = 'Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw = 1,
                 ls = '--', label =f'Sample {i+1}')

    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
        plt.plot(X, Y, label='True Values', color='red')

    plt.legend()

    plt.show()

    sn.heatmap(cov, annot = False, fmt = 'g')

    plt.show()

def plot_gp_2d(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1,2,i,projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm,
                    linewidth=0, alpha = 0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)

def posterior(X_new, X_train, Y_train, l = 1.0, sigma_f = 1.0, sigma_y = 1e-8):
    """
    Computes the sufficient statistics of the posterior distribution
    from m training data X_train, Y_train, and n new inputs X_s

    :param X_new: New input location (n x d)
    :param X_train: Trained input (m x d)
    :param Y_train: Trained output (m x 1)
    :param l: characteristic length
    :param sigma_f: vertical variation parameter
    :param sigma_y: Noise parameter

    :return: Posterior mean vector (n x d) and covariance matrix (n x n)
    """

    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(X_train.shape[0])
    K_s = kernel(X_train, X_new, l, sigma_f)
    K_ss = kernel(X_new, X_new, l, sigma_f) + sigma_y**2 * np.eye(X_new.shape[0])

    # Get Lower triangular matrix from cholsky
    L = np.linalg.cholesky(K)

    # Compute mean given data set and a new input
    z = np.linalg.solve(L, Y_train)
    alpha = np.linalg.solve(L.T, z)
    mu_new = K_s.T.dot(alpha)

    # Compute Covariance
    v = np.linalg.solve(L, K_s)
    cov_new = K_ss - v.T.dot(v)

    return mu_new, cov_new

def nll_fn(X_train, Y_train, noise):
    """
    Returns a function that computes the negative log marginal
    likelihood of training data X_train, Y_train
    at given noise level
    :param X_train: training input (m x d)
    :param Y_train: training output (m x 1)
    :param noise: known noise level

    :return: Minimization objective
    """

    Y_train = Y_train.ravel()

    def nll_stable(theta):
        K = kernel(X_train, X_train, l = theta[0], sigma_f = theta[1]) + \
            noise**2 * np.eye(X_train.shape[0])

        L = np.linalg.cholesky(K)

        S1 = np.linalg.solve(L, Y_train)
        S2 = np.linalg.solve(L.T, S1)

        return np.sum(np.log(np.diagonal(L))) + \
            0.5 * Y_train.dot(S2) + \
            0.5 * X_train.shape[0] * np.log(2 * np.pi)

    return nll_stable


def main_Prior():

    # Generate input data
    X = np.arange(-5, 5, 0.2).reshape(-1,1)
    Y = np.sin(X)

    # Mean and covariance set for Gaussian process prior
    # The kernel for covariance follows Radial basis function (RBF)
    mu =  np.zeros(X.shape)
    cov = kernel(X, X)

    samples = np.random.multivariate_normal(mu.ravel(), cov, 3)

    plot_gp(mu, cov, X, Y, samples = samples)
    plt.show()

def main_Posterior():
    X = np.arange(-5, 5, 0.2).reshape(-1, 1)
    Y = np.sin(X)
    X_train = np.array([-4, -3, -2, -1, 1, 3]).reshape(-1, 1)
    Y_train = np.sin(X_train)

    mu_new, cov_new = posterior(X, X_train, Y_train)

    samples = np.random.multivariate_normal(mu_new.ravel(), cov_new, 3)
    plot_gp(mu_new, cov_new, X, Y, X_train=X_train, Y_train=Y_train, samples=samples)

def main_Posterior_from_noisy_train_data():

    X = np.arange(-5, 5, 0.2).reshape(-1, 1)
    Y = np.sin(X)

    # Generate noisy data
    noise = 0.1
    X_train = np.arange(-3, 4, 1).reshape(-1,1)
    Y_train = np.sin(X_train) + noise*np.random.randn(*X_train.shape)


    res = minimize(nll_fn(X_train, Y_train, noise), [1, 1],
                                  bounds=((1e-5, None), (1e-5, None)),
                                  method='L-BFGS-B')

    l_opt, sigma_f_opt = res.x

    nll_res = res.fun

    print('l_opt = ', l_opt, ', sigma_f_opt = ', sigma_f_opt)
    print('nll_res = ', nll_res)

    mu_new, cov_new = posterior(X, X_train, Y_train,
                                sigma_y=noise, sigma_f=sigma_f_opt, l=l_opt)

    samples = np.random.multivariate_normal(mu_new.ravel(), cov_new, 3)
    plot_gp(mu_new, cov_new, X, Y, X_train=X_train, Y_train=Y_train, samples=samples)

def main_high_dim():

    noise_2D = 0.1

    rx, ry = np.arange(-5, 5, 0.3), np.arange(-5, 5, 0.3)
    gx, gy = np.meshgrid(rx, ry)

    X_2d = np.c_[gx.ravel(), gy.ravel()]

    X_2d_train = np.random.uniform(-4, 4, (50, 2))
    Y_2d_train = np.sin(0.5*np.linalg.norm(X_2d_train, axis = 1) + \
                        noise_2D*np.random.randn(X_2d_train.shape[0]))

    plt.figure(figsize=(14,7))

    mu_new, cov_new = posterior(X_2d, X_2d_train, Y_2d_train, sigma_y = noise_2D)

    plot_gp_2d(gx, gy, mu_new, X_2d_train, Y_2d_train, f'Before parameter optimization:'
                                                       f'l={1.00} sigma_f={1.00}', 1)

    res = minimize(nll_fn(X_2d_train, Y_2d_train, noise_2D), [1,1],
                   bounds=((1e-5, None), (1e-5, None)),
                   method='L-BFGS-B')

    l_opt, sigma_f_opt = res.x

    mu_new_opt, cov_new_opt = posterior(X_2d, X_2d_train, Y_2d_train,
                                        sigma_y = noise_2D, l=l_opt,
                                        sigma_f=sigma_f_opt)

    plot_gp_2d(gx, gy, mu_new_opt, X_2d_train, Y_2d_train, f'After parameter optimization:'
                                                       f'l={l_opt:2f} sigma_f={sigma_f_opt:2f}', 2)

    plt.show()


if __name__ == '__main__':
    # main_Prior()
    # main_Posterior()
    # main_Posterior_from_noisy_train_data()
    main_high_dim()