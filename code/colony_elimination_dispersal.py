import numpy as np

def simulate(J, S=10, p=2, c=0.1, N_ed=2, p_ed=0.25, N_re=4, N_c=100, rng=(-10,10), d_attract=0.1, w_attract=0.2, h_repellant=0.1, w_repellant=10):

    theta = rng[0] + np.random.rand(S, p)*(rng[1] - rng[0])

    def J_cc(x):
        result = 0
        for theta_i in theta:
            result += -d_attract*np.exp(-w_attract*((x-theta_i)**2).sum()) + h_repellant*np.exp(-w_repellant*((x-theta_i)**2).sum())
        return result

    J_histories = np.zeros((N_ed, N_re, S, N_c))
    theta_histories = np.zeros((N_ed, N_re, S, N_c, p))

    for l in range(N_ed):
        for k in range(N_re):
            for j in range(N_c):
                for i in range(S):
                    phi = np.random.uniform(low=-1, high=1, size=p)
                    phi /= np.linalg.norm(phi)
                    theta[i] = theta[i] + c*phi

                    theta_histories[l, k, i, j] = theta[i].copy()
                    J_histories[l, k, i, j] = J(theta[i])

                    while J(theta[i] + c*phi) + J_cc(theta[i] + c*phi) < J(theta[i]) + J_cc(theta[i]):
                        theta[i] = theta[i] + c*phi

            I = np.argsort(J_histories[l, k].sum(axis=1)) # sort
            theta = np.concatenate((theta[I[:S//2]].copy(), theta[I[:S//2]].copy()))
        for i in range(S):
            if np.random.rand() < p_ed:
                theta[i] = rng[0] + np.random.rand( p)*(rng[1] - rng[0])

    return J_histories, theta_histories
