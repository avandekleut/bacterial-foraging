import numpy as np

def simulate(J, S=10, p=2, c=0.1, N_c=100, rng=(-10,10), d_plus=0.1, w_plus=10, d_minus=0.1, w_minus=0.2):
    theta = rng[0] + np.random.rand(S, p)*(rng[1] - rng[0])

    def J_cc(x):
        result = 0
        for theta_i in theta:
            result += -d_plus*np.exp(-w_plus*(x-theta_i).T @ (x-theta_i)) + d_minus*np.exp(-w_minus*(x-theta_i).T @ (x - theta_i))
        return result

    J_histories = [[] for _ in range(S)]
    theta_histories = [[] for _ in range(S)]

    for j in range(N_c):
        for i in range(S):
            phi = np.random.uniform(low=-1, high=1, size=p)
            phi /= np.linalg.norm(phi)
            theta[i] = theta[i] + c*phi

            theta_histories[i].append(theta.copy())
            J_histories[i].append(J(theta[i]))

            while J(theta[i] + c*phi) + J_cc(theta[i] + c*phi) < J(theta[i]) + J_cc(theta[i]):
                theta[i] = theta[i] + c*phi

    return J_histories, theta_histories
