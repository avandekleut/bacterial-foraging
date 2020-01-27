from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib import cbook
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def rastrigin(theta, A = 10):
    """
    Highly nonconvex function with minimum at 0.
    https://en.wikipedia.org/wiki/Rastrigin_function
    """
    n = theta.size
    return A*n + (theta**2 - A*np.cos(2*np.pi*theta)).sum()

def simple_quadratic(theta):
    return (theta**2).sum()

def plot(J, rng=(-5.12, 5.12), num=100, cm=cm.rainbow):
    # generating data
    x = np.linspace(*rng, num=num)
    y = np.linspace(*rng, num=num)
    X, Y = np.meshgrid(x, y)
    theta = np.array([X, Y])
    Z = np.apply_along_axis(J, 0, theta)

    # creating figures
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # contour plot
    contour = ax.contour(X, Y, Z, cmap=cm, offset=np.min(Z), levels=5)

    # lighting and shading
    ls = LightSource(270, 45)
    rgb = ls.shade(Z, cmap=cm, vert_exag=0.1, blend_mode='soft')

    # surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm, alpha=1,
                       linewidth=1, antialiased=False, facecolors=rgb)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

class Bacterium:
    def __init__(self, p, C, rng=(-1, 1)):
        """
        p: int, number of dimensions
        C: positive float, step-size
        """
        self.p = p
        self.rng = rng
        self.theta = self.rng[0] + np.random.rand(self.p)*(self.rng[1] - self.rng[0])
        self.C = C
        self.J_last = None
        self.J_history = []
        self.theta_history = []

    def clone(self):
        """
        Creates a deep copy of the current bacterium
        """
        c = Bacterium(self.p, self.C)
        c.theta = self.theta.copy()
        return c

class Colony:
    """
    Contains information about the function J to be optimized.
    Contains an iterable of bacterium to do the optimization.
    """
    def __init__(self, f, S, p, C, rng):
        """
        f: callable(theta), function to optimize
        S: int, size of colony
        p: int, number of dimensions
        C: callable(i), returns positive float, step-size
        """
        self.S = [Bacterium(p, C(i), rng) for i in range(S)]
        self.J = self.J(f)

    def J_cc(self, theta, d_attract, w_attract, h_repellant, w_repellant):
        """
        Cell-to-cell attraction/repellant
        theta: p-dimensional numpy array, arbitrary position
        (P): ommitted. Only included in paper for mathematical reasons.
        d_attract: positive float, depth of attraction
        w_attract: positive float, width of attraction
        h_reppelant: positive float, depth of repellant
        w_repellant: positive float, width of repellant
        """
        result = 0

        for i in range(len(self.S)):
            theta_i = self.S[i].theta
            squared_distance = ((theta - theta_i)**2).sum()
            result += -d_attract*np.exp(-w_attract*squared_distance)  + h_repellant*np.exp(-w_repellant*squared_distance)
        return result

    def J(self, f):
        def inner(i, j, k, l):
            theta_i = self.S[i].theta
            return f(theta_i)
        return inner

    def simulate(self, p_ed, N_ed, N_re, N_c, N_s, d_attract=0.1, w_attract=0.2, h_repellant=0.1, w_repellant=10):
        for l in range(N_ed):
            # elimination-dispersal
            print('ed')
            for k in range(N_re):
                # reproduction
                print('rep')
                for j in range(N_c):
                    # chemotaxis

                    for i in range(len(self.S)):
                        bacterium = self.S[i]

                        # compute loss
                        J = self.J(i, j, k, l) + self.J_cc(bacterium.theta, d_attract, w_attract, h_repellant, w_repellant)

                        bacterium.J_last = J
                        bacterium.J_history.append(J)
                        bacterium.theta_history.append(bacterium.theta.copy())

                        # tumble
                        delta_i = -1 + 2*np.random.rand(bacterium.p)
                        bacterium.theta = bacterium.theta + bacterium.C*delta_i/np.sqrt(delta_i.T @ delta_i)

                        # run
                        for m in range(N_s):
                            J = self.J(i, j, k, l) + self.J_cc(bacterium.theta, d_attract, w_attract, h_repellant, w_repellant)

                            if J < bacterium.J_last:
                                bacterium.J_last = J
                                bacterium.theta = bacterium.theta + bacterium.C*delta_i/np.sqrt(delta_i.T @ delta_i)
                            else:
                                m = N_s

                            bacterium.J_history.append(J)
                            bacterium.theta_history.append(bacterium.theta.copy())

                # reproduction

                self.S = sorted(self.S, key=lambda bacterium: sum(bacterium.J_history), reverse=True) # sort by increasing total cost


                ######
                fig = plt.figure()
                ax = fig.gca()
                for bacterium in self.S:
                    ax.plot(bacterium.J_history)
                plt.show()

                print(sum([self.J(i, j, k, l) + self.J_cc(self.S[i].theta, d_attract, w_attract, h_repellant, w_repellant) for i in range(len(self.S))]))
                ######

                for i in range(len(self.S)):
                    bacterium = self.S[i]
                    bacterium.J_history.clear() # reset the history of each bacterium
                    bacterium.theta_history.clear()

                self.S = self.S[:len(self.S)//2] # take best half of offspring
                offspring = [bacterium.clone() for bacterium in self.S] # reproduce these offspring
                self.S += offspring # add them to the current colony

            # elimination-dispersal
            for i in range(len(self.S)):
                bacterium = self.S[i]
                if np.random.rand() < p_ed:
                    bacterium.theta = bacterium.rng[0] + np.random.rand(bacterium.p)*(bacterium.rng[1] - bacterium.rng[0]) # randomly transplant position of bacterium


c = Colony(simple_quadratic, 10, 2, lambda i: 0.1, (-20, 20))
# c = Colony(rastrigin, 100, 2, lambda i: 0.1, (-2, 2))
# c.simulate(0.05, 3, 5, 200, 4)
c.simulate(0.05, 3, 10, 20, 4)
