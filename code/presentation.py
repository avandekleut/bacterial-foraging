import numpy as np

def simulate(J, S=10, p=2, rng=(-5.12,5.12), c=0.1,
                N_ed=2, p_ed=0.25,
                N_re=4,
                d_attract=0.1, w_attract=0.2, h_repellant=0.1, w_repellant=10,
                N_c=100):

    theta = rng[0] + np.random.rand(S, p)*(rng[1] - rng[0])

    def J_cc(x):
        result = 0
        for theta_i in theta:
            result += -d_attract*np.exp(-w_attract*((x-theta_i)**2).sum()) \
            + h_repellant*np.exp(-w_repellant*((x-theta_i)**2).sum())
        return result

    for l in range(N_ed):
        for k in range(N_re):
            for j in range(N_c):
                for i in range(S):
                    phi = np.random.uniform(low=-1, high=1, size=p)
                    phi /= np.linalg.norm(phi)
                    J_last = J(theta[i]) + J_cc(theta[i])
                    theta[i] = theta[i] + c*phi
                    while J(theta[i]) + J_cc(theta[i]) < J_last:
                        J_last = J(theta[i]) + J_cc(theta[i])
                        theta[i] = theta[i] + c*phi

            I = np.argsort(J_histories[l, k].sum(axis=1))
            theta = np.concatenate((theta[I[:S//2]].copy(), theta[I[:S//2]].copy()))
        for i in range(S):
            if np.random.rand() < p_ed:
                theta[i] = rng[0] + np.random.rand(p)*(rng[1] - rng[0])
