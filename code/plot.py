from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib import cbook
from matplotlib import cm
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

def plot_surface(J, rng=(-5.12, 5.12), num=100, cm=cm.rainbow, filename=None):
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
    contour = ax.contour(X, Y, Z, cmap=cm, offset=np.min(Z))

    # lighting and shading
    ls = LightSource(270, 45)
    rgb = ls.shade(Z, cmap=cm, vert_exag=0.1, blend_mode='soft')

    # surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm, alpha=1,
                       linewidth=1, antialiased=False, facecolors=rgb)
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def plot_J_cc(S=3, rng=(-5.12, 5.12), num=100, cm=cm.rainbow, d_plus=0.1, w_plus=2, d_minus=0.2, w_minus=1, filename=None, seed=17):
    # generating data
    x = np.linspace(*rng, num=num)
    y = np.linspace(*rng, num=num)
    X, Y = np.meshgrid(x, y)
    theta = np.array([X, Y])

    np.random.seed(seed)
    thetas = rng[0] + np.random.rand(S, 2)*(rng[1] - rng[0])

    def J_cc(theta):
        result = 0
        for theta_i in thetas:
            result += -d_plus*np.exp(-w_plus*((theta-theta_i)**2).sum()) + d_minus*np.exp(-w_minus*((theta-theta_i)**2).sum())
        return result

    np.random.seed(None)
    Z = np.apply_along_axis(J_cc, 0, theta)

    # creating figures
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # contour plot
    contour = ax.contour(X, Y, Z, cmap=cm, offset=np.min(Z))

    # lighting and shading
    ls = LightSource(270, 45)
    rgb = ls.shade(Z, cmap=cm, vert_exag=0.1, blend_mode='soft')

    # surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm, alpha=1,
                       linewidth=1, antialiased=False, facecolors=rgb)
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def plot_J(J_histories, filename=None):
    fig = plt.figure()
    ax = fig.gca()
    for J_history in J_histories:
        ax.plot(J_history, alpha=0.2, color='r')
    if filename is None:
        plt.show()
    else:
        plt.savefig(f'{filename}_J.pdf')

def plot_paths(J, theta_histories, num=100, cm=cm.rainbow, filename=None):
    """
    Theta_histories: an iterable of iterables of 2D np.ndarrays.
    """
    rng = (np.min(theta_histories), np.max(theta_histories))
    fig = plt.figure()
    ax = fig.gca()
    x = np.linspace(*rng, num=num)
    y = np.linspace(*rng, num=num)
    X, Y = np.meshgrid(x, y)
    theta = np.array([X, Y])
    Z = np.apply_along_axis(J, 0, theta)
    contour = ax.contour(X, Y, Z, cmap=cm)

    for theta_history in theta_histories:
        theta_history = np.asarray(theta_history).T
        ax.plot(*theta_history, color='black', alpha=0.5)

    if filename is None:
        plt.show()
    else:
        plt.savefig(f'{filename}_theta.pdf')


def evaluate_sb(J, filename=None):
    from single_bacterium import simulate as sb
    J_histories = []
    theta_histories = []
    for _ in range(10):
        J_history, theta_history = sb(J)
        J_histories.append(J_history)
        theta_histories.append(theta_history)

    plot_J(J_histories, filename=filename)
    plot_paths(J, theta_histories, filename=filename)

if __name__ == "__main__":
    from losses import sphere, rastrigin
    evaluate_sb(sphere, filename='presentation/assets/sphere')
    evaluate_sb(rastrigin, filename='presentation/assets/rastrigin')
    plot_J_cc(d_plus=0.1, w_plus=0.2, d_minus=0.1, w_minus=10, filename='presentation/assets/swarming.pdf')
