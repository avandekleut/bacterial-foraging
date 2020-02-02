import numpy as np

def simulate(J, p=2, N=10, iter=100, rng=(-5.12, 5.12), mu=0.4, c_1=0.8, c_2=1.0):
    """
    Implements particle swarm optimization.
    J: np.ndarray -> float          the function to minimize
    p: int                          the dimensionality of the problem
    N: int                          the number of particles
    mu: float                       the momentum
    c_1: float                      global best influence
    c_2: float                      personal best influence
    """

    v = np.zeros((N, p))
    theta = rng[0] + np.random.rand(N, p)*(rng[1] - rng[0])

    p_best = theta.copy()
    J_p_best = np.apply_along_axis(J, 1, p_best)

    g_best = theta[np.argmin(J_p_best)].copy()
    J_g_best = J_p_best.min()

    J_histories = np.zeros((N, iter))
    theta_histories = np.zeros((N, iter, p))

    for t in range(iter):
        v = mu*v + c_1*np.random.rand(N, p)*(g_best-theta) + c_2*np.random.rand(N, p)*(p_best-theta)
        theta = theta + v

        J_theta = np.apply_along_axis(J, 1, theta)
        improved_p_best = np.where(J_theta < J_p_best)

        p_best[improved_p_best] = theta[improved_p_best].copy()
        J_p_best[improved_p_best] = J_theta[improved_p_best]

        g_best = theta[np.argmin(J_p_best)].copy()
        J_g_best = J_p_best.min()

        J_histories[:, t] = J_theta.copy()
        theta_histories[:, t, :] = theta.copy()

    return J_histories, theta_histories
