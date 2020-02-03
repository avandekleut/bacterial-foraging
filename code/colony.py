import numpy as np

def simulate(J, S=10, p=2, c=0.1, N_c=100, rng=(-5.12,5.12), d_attract=0.1, w_attract=0.2, h_repellant=0.1, w_repellant=10):

    theta = rng[0] + np.random.rand(S, p)*(rng[1] - rng[0])

    def J_cc(x):
        result = 0
        for theta_i in theta:
            result += -d_attract*np.exp(-w_attract*((x-theta_i)**2).sum()) + h_repellant*np.exp(-w_repellant*((x-theta_i)**2).sum())
        return result

    J_histories = np.zeros((S, N_c))
    theta_histories = np.zeros((S, N_c, p))

    for j in range(N_c):
        for i in range(S):
            phi = np.random.uniform(low=-1, high=1, size=p)
            phi /= np.linalg.norm(phi)
            J_last = J(theta[i]) + J_cc(theta[i])
            theta[i] = theta[i] + c*phi
            while J(theta[i]) + J_cc(theta[i]) < J_last:
                J_last = J(theta[i]) + J_cc(theta[i])
                theta[i] = theta[i] + c*phi

            theta_histories[i, j] = theta[i].copy()
            J_histories[i, j] = J(theta[i])

    return J_histories, theta_histories
