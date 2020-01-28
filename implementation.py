from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib import cbook
from matplotlib import cm
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


def rastrigin(theta, A = 10):
    n = theta.size
    return A*n + (theta**2 - A*np.cos(2*np.pi*theta)).sum()

def sphere(theta):
    return (theta**2).sum()

def rosenbrock(theta):
    return (100*(theta[1:] - theta[:-1]**2)**2 + (1-theta)**2).sum()

def easom(theta):
    assert theta.size == 2
    x, y = theta[0], theta[1]
    return -np.cos(x)*np.cos(y)*np.exp(-((x-np.pi)**2 + (y-np.pi)**2))

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
    def __init__(self, J, S, p, C, rng, d_attract=0.1, w_attract=0.2, h_repellant=0.1, w_repellant=10):
        """
        J: callable(theta), function to optimize
        S: int, size of colony
        p: int, number of dimensions
        C: callable(i), returns positive float, step-size
        d_attract: positive float, depth of attraction
        w_attract: positive float, width of attraction
        h_reppelant: positive float, depth of repellant
        w_repellant: positive float, width of repellant
        """
        self.S   = [Bacterium(p, C(i), rng) for i in range(S)]
        self.J   = J
        self.rng = rng

        self.d_attract   = d_attract
        self.w_attract   = w_attract
        self.h_repellant = h_repellant
        self.w_repellant = w_repellant

    def J_cc(self, theta, P=None):
        """
        Cell-to-cell attraction/repellant
        theta: p-dimensional numpy array, arbitrary position
        """
        result = 0

        if P is None:
            P = [bacterium.theta for bacterium in self.S]

        for i in range(len(P)):
            theta_i = P[i]
            squared_distance = ((theta - theta_i)**2).sum()
            result += -self.d_attract*np.exp(-self.w_attract*squared_distance)  + self.h_repellant*np.exp(-self.w_repellant*squared_distance)
        return result

    def simulate(self, p_ed, N_ed, N_re, N_c, N_s):
        """
        p_ed: float in [0, 1] probability of elimination-dispersal
        N_ed: number of elimination-dispersal iterations
        N_re: number of reproduction iterations
        N_c:  number of chemotaxis steps
        N_s:  maximum length for a run
        d_attract: positive float, depth of attraction
        w_attract: positive float, width of attraction
        h_reppelant: positive float, depth of repellant
        w_repellant: positive float, width of repellant
        """
        for l in range(N_ed):
            # elimination-dispersal
            print(f'Elimination dispersal {l+1}/{N_ed}')
            for k in range(N_re):
                # reproduction
                print(f'\t Reproduction {k+1}/{N_re}')
                for j in range(N_c):
                    # chemotaxis
                    print(f'\t\t Chemotaxis {j+1}/{N_c}')
                    for i in range(len(self.S)):
                        print(f'\t\t\t Bacterium {i+1}/{len(self.S)}\r', end='')
                        bacterium = self.S[i]

                        # compute loss
                        J = self.J(bacterium.theta)
                        J_cc = self.J_cc(bacterium.theta)

                        bacterium.J_last = J + J_cc
                        bacterium.J_history.append(J)
                        bacterium.theta_history.append(bacterium.theta.copy())

                        # tumble
                        delta_i = -1 + 2*np.random.rand(bacterium.p)
                        bacterium.theta = bacterium.theta + bacterium.C*delta_i/np.sqrt(delta_i.T @ delta_i)

                        # run
                        for m in range(N_s):
                            J = self.J(bacterium.theta)
                            J_cc = self.J_cc(bacterium.theta)

                            if (J + J_cc) < bacterium.J_last:
                                bacterium.J_last = J + J_cc
                                bacterium.theta = bacterium.theta + bacterium.C*delta_i/np.sqrt(delta_i.T @ delta_i)
                            else:
                                m = N_s

                            bacterium.J_history.append(J)
                            bacterium.theta_history.append(bacterium.theta.copy())
                    print()
                # reproduction

                self.S = sorted(self.S, key=lambda bacterium: sum(bacterium.J_history), reverse=True) # sort by increasing total cost

                # for i in range(len(self.S)):
                #     bacterium = self.S[i]
                #     bacterium.J_history.clear() # reset the history of each bacterium
                    # bacterium.theta_history.clear()

                self.S = self.S[:len(self.S)//2] # take best half of offspring
                offspring = [bacterium.clone() for bacterium in self.S] # reproduce these offspring
                self.S += offspring # add them to the current colony

            # elimination-dispersal
            for i in range(len(self.S)):
                bacterium = self.S[i]
                if np.random.rand() < p_ed:
                    bacterium.theta = bacterium.rng[0] + np.random.rand(bacterium.p)*(bacterium.rng[1] - bacterium.rng[0]) # randomly transplant position of bacterium

    def plot_paths2(self, speed=10):
        # generating data
        x = np.linspace(*self.rng, num=100)
        y = np.linspace(*self.rng, num=100)
        X, Y = np.meshgrid(x, y)
        theta = np.array([X, Y])

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        def surfaces(i):
            print(f'frame {i}')
            ax.cla()

            P = []
            for bacterium in self.S:
                try:
                    P.append(bacterium.theta_history[i])
                except:
                    pass

            Z = np.apply_along_axis(lambda theta: self.J(theta) + self.J_cc(theta, P=P), 0, theta)
            ls = LightSource(270, 45)
            rgb = ls.shade(Z, cmap=cm.rainbow, vert_exag=0.1, blend_mode='soft')
            surf = ax.plot_surface(X, Y, Z, cmap=cm.rainbow, alpha=1,
                               linewidth=1, facecolors=rgb, antialiased=False)

            contour = ax.contour(X, Y, Z, cmap=cm.rainbow, offset=np.min(Z))

            return ax,

        max_path_length = max([len(bacterium.theta_history) for bacterium in self.S])

        print(max_path_length//speed)

        ani = animation.FuncAnimation(fig, surfaces, frames=range(max_path_length//speed), interval=1, repeat=False)

        plt.show()
        ani.save('', fps=60)


    def plot_paths(self, speed=10):
        x = np.linspace(*self.rng, num=100)
        y = np.linspace(*self.rng, num=100)
        X, Y = np.meshgrid(x, y)
        theta = np.array([X, Y])
        Z = np.apply_along_axis(self.J, 0, theta)

        fig = plt.figure()
        ax = fig.gca()
        ax.set_ylim(*self.rng)
        ax.set_xlim(*self.rng)
        contour = ax.contour(X, Y, Z, cmap=cm.rainbow)

        def paths(i):
            for bacterium in self.S:
                ax.plot(*np.array(bacterium.theta_history[:i*speed]).T, color='gray') # paths
            return ax,

        max_path_length = max([len(bacterium.theta_history) for bacterium in self.S])
        ani = animation.FuncAnimation(fig, paths, frames=range(max_path_length//speed), interval=1, repeat=False)

        plt.show()

colony_args = dict(
    J=lambda theta:0,
    S=6,
    p=2,
    C=lambda i: 0.5,
    rng=(-5.12, 5.12),
    h_repellant=3
)

simulation_args = dict(
    p_ed=0.05,
    N_ed=1,
    N_re=1,
    N_c=10,
    N_s=10
)

c = Colony(**colony_args)
c.simulate(**simulation_args)
c.plot_paths(speed=1)
