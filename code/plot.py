import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib import cbook
from matplotlib import cm
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(17)

def plot_surface(J, rng=(-10, 10), num=100, cm=cm.rainbow, filename=None):
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

def plot_J_cc(title=None, filename=None, S=3, rng=(-10, 10), num=100, cm=cm.rainbow, d_attract=0.1, w_attract=0.2, h_repellant=0.1, w_repellant=10):

    x = np.linspace(*rng, num=num)
    y = np.linspace(*rng, num=num)
    X, Y = np.meshgrid(x, y)
    theta = np.array([X, Y])

    thetas = rng[0] + np.random.rand(S, 2)*(rng[1] - rng[0])

    def J_cc(theta):
        result = 0
        for theta_i in thetas:
            result += -d_attract*np.exp(-w_attract*((theta-theta_i)**2).sum()) + h_repellant*np.exp(-w_repellant*((theta-theta_i)**2).sum())
        return result

    np.random.seed(None)
    Z = np.apply_along_axis(J_cc, 0, theta)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(title)

    contour = ax.contour(X, Y, Z, cmap=cm, offset=np.min(Z))

    ls = LightSource(270, 45)
    rgb = ls.shade(Z, cmap=cm, vert_exag=0.1, blend_mode='soft')

    surf = ax.plot_surface(X, Y, Z, cmap=cm, alpha=1,
                       linewidth=1, antialiased=False, facecolors=rgb)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def plot_J(J_histories, title=None, filename=None):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel(r'Iteration ($j$)')
    ax.set_ylabel(r'$J(\theta)$')
    ax.set_title(title)
    for J_history in J_histories:
        ax.plot(J_history, alpha=0.2, color='r')
    ax.text(0.5, 0.5, r'$J^*$ =' f'{np.min(J_histories)}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    if filename is None:
        plt.show()
    else:
        plt.savefig(f'{filename}_J.pdf')

def plot_paths(J, theta_histories, title=None,  num=100, cm=cm.rainbow, filename=None):
    """
    Theta_histories: an iterable of iterables of 2D np.ndarrays.
    """
    rng = (np.min(theta_histories), np.max(theta_histories))
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(title)
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

def evaluate_sb(J, title=None, filename=None, **simulation_args):
    from single_bacterium import simulate as sb
    J_histories = []
    theta_histories = []
    for _ in range(10):
        J_history, theta_history = sb(J, **simulation_args)
        J_histories.append(J_history)
        theta_histories.append(theta_history)

    plot_J(J_histories, title=title, filename=filename)
    plot_paths(J, theta_histories, title=title, filename=filename)

def evaluate_col(J, title=None, filename=None, **simulation_args):
    from colony import simulate as col
    J_histories, theta_histories = col(J, **simulation_args)
    plot_J(J_histories, title=title, filename=filename)
    plot_paths(J, theta_histories, title=title, filename=filename)

if __name__ == "__main__":
    from losses import sphere, rastrigin

    evaluate_sb(sphere,
        title=r'$p$=2, $c$=0.1, $N_c$=100',
        filename='presentation/assets/sphere',
        p=2,
        c=0.1,
        N_c=100,
        rng=(-10,10))

    evaluate_sb(rastrigin,
        title=r'$p$=2, $c$=0.1, $N_c$=100',
        filename='presentation/assets/rastrigin',
        p=2,
        c=0.1,
        N_c=100,
        rng=(-10,10))

    plot_J_cc(title=r'$S$=3, $p$=2, $d_\mathregular{attract}$=0.1, $w_\mathregular{attract}$=0.2, $h_\mathregular{repellant}$=0.1, $w_\mathregular{repellant}$=10',
        filename='presentation/assets/swarming.pdf',
        S=3,
        rng=(-10, 10),
        num=100,
        d_attract=0.1,
        w_attract=0.2,
        h_repellant=0.1,
        w_repellant=10)

    evaluate_col(sphere,
        title=r'$S$=10, $p$=2, $c$=0.1, $N_c$=100' '\n' r'$d_\mathregular{attract}$=0.1, $w_\mathregular{attract}$=0.2, $h_\mathregular{repellant}$=0.1, $w_\mathregular{repellant}$=10',
        filename='presentation/assets/sphere_colony',
        S=10,
        p=2,
        c=0.1,
        N_c=100,
        rng=(-10,10),
        d_attract=0.1,
        w_attract=0.2,
        h_repellant=0.1,
        w_repellant=10)

    evaluate_col(rastrigin,
        title=r'$S$=10, $p$=2, $c$=0.1, $N_c$=100' '\n' r'$d_\mathregular{attract}$=0.1, $w_\mathregular{attract}$=0.2, $h_\mathregular{repellant}$=0.1, $w_\mathregular{repellant}$=10',
        filename='presentation/assets/rastrigin_colony',
        S=10,
        p=2,
        c=0.1,
        N_c=100,
        rng=(-10,10),
        d_attract=0.1,
        w_attract=0.2,
        h_repellant=0.1,
        w_repellant=10)
