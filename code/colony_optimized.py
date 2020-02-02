import numpy as np
import sklearn.metrics

"""
This code optimizes some of the functionality of colony_elimination_dispersal.py
using vectorized operations. It essentially avoids looping over the colony itself (index i).
"""

def simulate(J, S=10, p=2, c=0.1, N_ed=2, p_ed=0.25, N_re=4, N_c=100, rng=(-5.12,5.12), d_attract=0.1, w_attract=0.2, h_repellant=0.1, w_repellant=10, verbose=False):

    theta = rng[0] + np.random.rand(S, p)*(rng[1] - rng[0])

    def J_cc(x, theta):
        diff = sklearn.metrics.pairwise.euclidean_distances(theta, x, squared=True)
        return (-d_attract*np.exp(-w_attract*diff) + h_repellant*np.exp(-w_repellant*diff)).sum(axis=0)

    J_histories = np.zeros((N_ed, N_re, S, N_c))
    theta_histories = np.zeros((N_ed, N_re, S, N_c, p))

    for l in range(N_ed):
        if(verbose):
            print(f'{l}/{N_ed}')
        for k in range(N_re):
            if(verbose):
                print(f' {k}/{N_re}')
            for j in range(N_c):
                if(verbose):
                    print(f'   {j}/{N_c}')
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

import pso
import time
import losses

t = time.time()
J_history, theta_history = pso.simulate(losses.rastrigin)
print(time.time()-t)
print(np.min(J_history))

t = time.time()
J_history, theta_history = simulate(losses.rastrigin)
print(time.time()-t)
print(np.min(J_history))
