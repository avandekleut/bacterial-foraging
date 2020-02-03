import numpy as np

def simulate(J, p=2, c=0.1, N_c=100, rng=(-5.12,5.12)):

    theta = rng[0] + np.random.rand(p)*(rng[1] - rng[0])

    J_history = np.zeros((N_c,))
    theta_history = np.zeros((N_c, p))

    for j in range(N_c):
        phi = np.random.uniform(low=-1, high=1, size=p)
        phi /= np.linalg.norm(phi)
        J_last = J(theta)

        theta = theta + c*phi

        J_new = J(theta)

        while (J_new < J_last):
            J_last = J_new
            theta = theta + c*phi
            J_new = J(theta)

        theta_history[j] = theta.copy()
        J_history[j] = J(theta)

    return J_history, theta_history
