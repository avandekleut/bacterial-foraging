import numpy as np
import sklearn.metrics

"""
This code optimizes some of the functionality of colony_elimination_dispersal.py
using vectorized operations. It essentially avoids looping over the colony itself (index i).
"""

def simulate(J, S=10, p=2, c=0.1, N_ed=2, p_ed=0.25, N_re=4, N_c=100, rng=(-5.12,5.12), d_attract=0.1, w_attract=0.2, h_repellant=0.1, w_repellant=10):

    theta = rng[0] + np.random.rand(S, p)*(rng[1] - rng[0])

    def J_cc(x, theta):
        diff = sklearn.metrics.pairwise.euclidean_distances(theta, x, squared=True)
        return (-d_attract*np.exp(-w_attract*diff) + h_repellant*np.exp(-w_repellant*diff)).sum(axis=0)

    J_histories = np.zeros((N_ed, N_re, S, N_c))
    theta_histories = np.zeros((N_ed, N_re, S, N_c, p))

    for l in range(N_ed):
        for k in range(N_re):
            for j in range(N_c):
                phi = np.random.uniform(low=-1, high=1, size=(S,p))
                phi /= np.linalg.norm(phi)
                theta = theta + c*phi
                J_theta = np.apply_along_axis(J, 1, theta) + J_cc(theta, theta)

                theta_new = theta + c*phi
                J_theta_new = np.apply_along_axis(J, 1, theta_new) + J_cc(theta_new, theta_new)

                to_run = np.where(J_theta_new < J_theta)[0]
                while len(to_run) > 0:
                    theta_old = theta_new.copy()
                    J_theta_old = J_theta_new.copy()
                    theta_new[to_run] = theta_old[to_run] + c*phi[to_run]
                    J_theta_new[to_run] = np.apply_along_axis(J, 1, theta_new[to_run]) + J_cc(theta_new[to_run], theta_new)
                    to_run = np.where(J_theta_new < J_theta_old)[0]

                theta = theta_new

                J_histories[l, k, :, j] = np.apply_along_axis(J, 1, theta)
                theta_histories[l, k, :, j, :] = theta.copy()

            I = np.argsort(J_histories[l, k].sum(axis=1)) # sort
            theta = np.concatenate((theta[I[:S//2]].copy(), theta[I[:S//2]].copy()))

        disperse = np.random.rand(S) < p_ed
        theta[disperse] = rng[0] + np.random.rand(disperse.sum(), p)*(rng[1] - rng[0])

    return J_histories, theta_histories

import itertools
from losses import rastrigin

d_attract=[0.1, 1, 10]
w_attract=[0.1, 1, 10]
h_repellant=[0.1, 1, 10]
w_repellant=[0.1, 1, 10]

hyperparameters = list(itertools.product(d_attract, w_attract, h_repellant, w_repellant))

i = 1
best_hyperparameters = hyperparameters[0]
best_J = np.inf
for d_attract, w_attract, h_repellant, w_repellant in hyperparameters:
    np.random.seed(17)
    print(f'{i}/{len(hyperparameters)}')
    J_history, _ = simulate(rastrigin, d_attract=d_attract, w_attract=w_attract, h_repellant=h_repellant, w_repellant=w_repellant)

    J_min = np.min(J_history)
    if J_min < best_J:
        best_J = J_min
        best_hyperparameters = (d_attract, w_attract, h_repellant, w_repellant)
        print('\t',J_min)
        print('\t',best_hyperparameters)
    i += 1
