import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt

from jax import random, jit, value_and_grad
from jax.config import config
from scipy.optimize import minimize
from matplotlib import animation
from IPython.display import HTML


config.update("jax_enable_x64", True)



def func(x):
    """Latent function"""
    return 1.0 * jnp.sin(x * 3 * jnp.pi) + \
            0.3 * jnp.cos(x * 9 * jnp.pi) + \
            0.5 * jnp.sin(x * 7 * jnp.pi)

def kernel(X1, X2, theta):
    """ Isotropic squared exponential kernel

    X1: Array of m points (m x d)
    X2: Array of n points (n x d)
    theta: kernel parameters (2,)
    """

    sqdist = jnp.sum(X1 ** 2, axis=1).reshape(-1, 1) +\
             jnp.sum(X2 ** 2, axis=1) +\
             -2*jnp.dot(X1, X2.T)

    return theta[1]**2 * jnp.exp(-0.5 / theta[0]**2 * sqdist)

def kernel_diag(d,theta):
    """
    Isotropic squared exponential kernel (computes diagonal elements only)
    :param d:
    :param theta:
    :return:
    """
    return jnp.full(shape=d, fill_value=theta[1]**2)

def jitter(d, value=1e-6):
    return jnp.eye(d) * value

def softplus(X):
    return jnp.log(1 + jnp.exp(X))

def softplus_inv(X):
    return jnp.log(jnp.exp(X) - 1)

def pack_params(theta, X_m):
    return jnp.concatenate([softplus_inv(theta), X_m.ravel()])

def unpack_params(params):
    return softplus(params[:2]), jnp.array(params[2:].reshape(-1,1))

def nlb_fn(X, y, sigma_y):
    n = X.shape[0]

    def nlb(params):
        """
        Negative lower bound on log marginal likelihood
        :param params: kernel parameters 'theta' and inducing points 'X_m'
        :return:
        """

        theta, X_m = unpack_params(params)

        K_mm = kernel(X_m, X_m, theta) + jitter(X_m.shape[0])
        K_mn = kernel(X_m, X, theta)

        L = jnp.linalg.cholesky(K_mm)
        A = jsp.linalg.solve_triangular(L, K_mn, lower=True) / sigma_y
        AAT = A @ A.T   # m x m
        B = jnp.eye(X_m.shape[0]) + AAT
        LB = jnp.linalg.cholesky(B)
        c = jsp.linalg.solve_triangular(LB, A.dot(y), lower=True) / sigma_y

        # Lower bound
        lb = - n/2 * jnp.log(2*jnp.pi)
        lb -= jnp.sum(jnp.log(jnp.diag(LB)))
        lb -= n/2 * jnp.log(sigma_y**2)
        lb -= 0.5/sigma_y**2 * y.T.dot(y)
        lb += 0.5 * c.T.dot(c)
        lb -= 0.5/sigma_y**2 * jnp.sum(kernel_diag(n,theta))
        lb += 0.5*jnp.trace(AAT)

        return -lb[0, 0]

    nlb_grad = jit(value_and_grad(nlb))

    def nlb_grad_wrapper(params):
        value, grads = nlb_grad(params)
        return np.array(value), np.array(grads)

    return nlb_grad_wrapper

def phi_opt(theta, X_m, X, y, sigma_y):
    """Optimize mu_m and A_m"""
    precision = (1.0/sigma_y**2)

    K_mm = kernel(X_m, X_m, theta) + jitter(X_m.shape[0])
    K_mm_inv = jnp.linalg.inv(K_mm)
    K_nm = kernel(X, X_m, theta)
    K_mn = K_nm.T

    Sigma = jnp.linalg.inv(K_mm + precision * K_mn @ K_nm)

    mu_m = precision*(K_mm @ Sigma @ K_mn).dot(y)
    A_m = K_mm @ Sigma @ K_mm
    return mu_m, A_m, K_mm_inv

def q(X_test, theta, X_m, mu_m, A_m, K_mm_inv):
    """
    Approximate posterial
    :param X_test:
    :param theta:
    :param X_m:
    :param mu_m:
    :param A_m:
    :param K_mm_inv:
    :return:
    """

    K_ss = kernel(X_test, X_test, theta)
    K_sm = kernel(X_test, X_m, theta)
    K_ms = K_sm.T

    f_q = (K_sm @ K_mm_inv).dot(mu_m)
    f_q_cov = K_ss - K_sm @ K_mm_inv @ K_ms + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms

    return f_q, f_q_cov

def generate_animation(theta_steps, X_m_steps, X_test, f_true, X, y, sigma_y, phi_opt, q, interval=100):
    fig, ax = plt.subplots()

    def plot_step(i):
        theta = theta_steps[i]
        X_m = X_m_steps[i]

        mu_m, A_m, K_mm_inv = phi_opt(theta, X_m, X, y, sigma_y)
        f_test, f_test_cov = q(X_test, theta, X_m, mu_m, A_m, K_mm_inv)
        f_test_var = np.diag(f_test_cov)
        f_test_std = np.sqrt(f_test_var)
        
        ax.clear()
        
        line_func, = ax.plot(X_test, f_true, label='Latent function', c='k', lw=0.5)
        line_pred, = ax.plot([], [], label='Prediction', c='b')       
        pnts_ind = ax.scatter(X_m, mu_m, label='Inducing variables', c='m')
        line_pred.set_data(X_test, f_test.ravel())
        area_pred = ax.fill_between(X_test.ravel(),
                                    f_test.ravel() + 1.96 * f_test_std,
                                    f_test.ravel() - 1.96 * f_test_std,
                                    color='r', alpha=0.1,
                                    label='Epistemic uncertainty')
        
        ax.set_title('Optimization of a sparse Gaussian process')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-3, 3.5)
        ax.legend(loc='upper right')

        return line_func, pnts_ind, line_pred, area_pred

    result = animation.FuncAnimation(fig, plot_step, frames=len(theta_steps), interval=interval, repeat=False)

    # plt.show()
    
    result.save('sgpr_example.gif',writer ='pillow', fps = 30)

    # Prevent output of last frame as additional plot
    # plt.close()

    return result

# Number of training data
n = 1000

# Number of inducing variables
m = 30

# Noise
sigma_y = 0.2

# Noisy training data
X = jnp.linspace(-1, 1, n).reshape(-1, 1)
y = func(X) + sigma_y * random.normal(random.PRNGKey(0), (n, 1))

# Test data
X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1,1)
f_true = func(X_test)

# Inducing points
X_m = jnp.linspace(-0.4, 0.4, m).reshape(-1,1)


print(nlb_fn(X, y, sigma_y))

theta_0 = jnp.array([1.0, 1.0])
theta_steps = [theta_0]

X_m_0 = X_m
X_m_steps = [X_m_0]

def callback(xk):
    theta, X_m = unpack_params(xk)
    theta_steps.append(theta)
    X_m_steps.append(X_m)
    return False


# # Run optimization
# res = minimize(fun=nlb_fn(X, y, sigma_y),
#                x0=pack_params(jnp.array([1.0, 1.0]), X_m),
#                method='L-BFGS-B',
#                jac=True,
#                callback=callback)

res = minimize(fun=nlb_fn(X, y, sigma_y),
               x0=pack_params(theta_0, X_m_0),
               method='L-BFGS-B',
               jac=True,
               callback=callback)

anim = generate_animation(theta_steps, X_m_steps, X_test, f_true, X, y, sigma_y, phi_opt, q)


# Show animation widget
HTML(anim.to_jshtml())

# # Optimized kernel parameters and inducing points
# theta_opt, X_m_opt = unpack_params(res.x)
#
# mu_m_opt, A_m_opt, K_mm_inv = phi_opt(theta_opt, X_m_opt, X, y, sigma_y)
#
# f_test, f_test_cov = q(X_test, theta_opt, X_m_opt, mu_m_opt, A_m_opt, K_mm_inv)
# f_test_var = np.diag(f_test_cov)
# f_test_std = np.sqrt(f_test_var)


# plt.scatter(X, y, label='Training examples', marker='x', color='blue', alpha=0.1)
# plt.plot(X_test, f_true, label='Latent function', c='k', lw=0.5)
# plt.plot(X_test, f_test, label='Prediction', c='b')
# plt.scatter(X_m_opt, mu_m_opt, label='Inducing variables', c='m')
# plt.fill_between(X_test.ravel(),
#                  f_test.ravel() + 1.96*f_test_std,
#                  f_test.ravel() - 1.96*f_test_std,
#                  label='Epistemic uncertainty',
#                  color='r',
#                  alpha=0.1)
#
# plt.title('Approximate posterior')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.xlim(-1.5, 1.5)
# plt.ylim(None, 3.0)
# plt.legend()
#
# plt.show()