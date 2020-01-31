import numpy as np

def simulate(J, p=2, c=0.1, N_c=100, rng=(-10,10)):

    theta = rng[0] + np.random.rand(p)*(rng[1] - rng[0])

    J_history = np.zeros((N_c,))
    theta_history = np.zeros((N_c, p))

    for j in range(N_c):
        phi = np.random.uniform(low=-1, high=1, size=p)
        phi /= np.linalg.norm(phi)
        theta = theta + c*phi

        theta_history[j] = theta.copy()
        J_history[j] = J(theta)

        while J(theta + c*phi) < J(theta):
            theta = theta + c*phi

    return J_history, theta_history
