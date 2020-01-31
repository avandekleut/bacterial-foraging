import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix' # set fonts to latex fonts
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib import cbook
from matplotlib import cm
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

def plot_surface(J,  filename=None, rng=(-5.12, 5.12), num=100, cm=cm.rainbow):
    """
    Makes a surface plot of the function J.
    J: np.ndarray -> float      function to minimize
    filename: str               filename to save as
    rng: tuple(low, high)       range of values for theta_1 and theta_2
    num: int                    how many points to use for surface plot grid
    cm: matplotlib.cm           colormap to use for plotting
    """
    x = np.linspace(*rng, num=num)
    y = np.linspace(*rng, num=num)
    X, Y = np.meshgrid(x, y)
    theta = np.array([X, Y])
    Z = np.apply_along_axis(J, 0, theta)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    contour = ax.contour(X, Y, Z, cmap=cm, offset=np.min(Z))

    ls = LightSource(270, 45)
    rgb = ls.shade(Z, cmap=cm, vert_exag=0.1, blend_mode='soft')

    surf = ax.plot_surface(X, Y, Z, cmap=cm, alpha=1,
                       linewidth=1, antialiased=False, facecolors=rgb)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close('all')

def plot_J_cc(title=None, filename=None, S=3, rng=(-5.12, 5.12), num=100, cm=cm.rainbow, d_attract=0.1, w_attract=0.2, h_repellant=0.1, w_repellant=10):
    """
    Plots cell-to-cell interaction function.
    title: str                  title of plot
    filename: str               filename to save as
    S: interaction              number of bacterium to include
    rng: tuple(low, high)       range of values for theta_1 and theta_2
    num: int                    how many points to use for surface plot grid
    cm: matplotlib.cm           colormap to use for plotting
    d_attract: float > 0        depth of attraction
    w_attract: float > 0        reciprocal width of attraction
    h_repellant: float > 0      height of repellant
    w_repellant: float > 0      reciprocal width of repellant
    Default values chosen to replicate plots from paper.
    """

    x = np.linspace(*rng, num=num)
    y = np.linspace(*rng, num=num)
    X, Y = np.meshgrid(x, y)
    theta = np.array([X, Y])

    np.random.seed(17)
    thetas = rng[0] + np.random.rand(S, 2)*(rng[1] - rng[0])

    def J_cc(theta):
        result = 0
        for theta_i in thetas:
            result += -d_attract*np.exp(-w_attract*((theta-theta_i)**2).sum()) + h_repellant*np.exp(-w_repellant*((theta-theta_i)**2).sum())
        return result

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
    plt.close('all')

def plot_J(J_histories, title=None, filename=None):
    """
    Plots the values of J achieved during optimization.
    J_histories: np.ndarray(S, N_c)
    title: str                  title of plot
    filename: str               filename to save as
    """
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel(r'Iteration ($j$)')
    ax.set_ylabel(r'$J(\theta)$')
    ax.set_title(title)
    for J_history in J_histories:
        ax.plot(J_history, alpha=0.2, color='r')
    ax.text(0.5, 0.5, r'$J^*$ =' f'{np.min(J_histories):.2}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='large')
    if filename is None:
        plt.show()
    else:
        plt.savefig(f'{filename}_J.pdf')
    plt.close('all')

def plot_J_re(J_histories, title=None, filename=None):
    """
    Plots the values of J achieved during optimization.
    J_histories: np.ndarray(N_re, S, N_c)
    title: str                  title of plot
    filename: str               filename to save as
    """
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel(r'Iteration ($j$)')
    ax.set_ylabel(r'$J(\theta)$')
    ax.set_title(title)
    N_re = J_histories.shape[0]
    N_c  = J_histories.shape[2]
    colours = ['r', 'g', 'b', 'c']*N_re
    for k in range(N_re):
        generation = J_histories[k]
        colour = colours[k]
        for J_history in generation:
            ax.plot(range(k*N_c, (k+1)*N_c), J_history, alpha=0.2, color=colour)
    ax.text(0.5, 0.5, r'$J^*$ =' f'{np.min(J_histories):.2}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='large')
    if filename is None:
        plt.show()
    else:
        plt.savefig(f'{filename}_J.pdf')
    plt.close('all')

def plot_paths(J, theta_histories, title=None, filename=None, rng=(-5.12, 5.12), num=100, cm=cm.rainbow, ):
    """
    Plots the paths taken during optimization.
    J: np.ndarray -> float      function to minimize
    theta_histories: J_histories: np.ndarray(S, N_c, p)
    title: str                  title of plot
    filename: str               filename to save as
    num: int                    how many points to use for contour plot grid
    cm: matplotlib.cm           colormap to use for plotting
    """

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
        ax.plot(*theta_history.T, color='black', alpha=0.5)

    if filename is None:
        plt.show()
    else:
        plt.savefig(f'{filename}_theta.pdf')
    plt.close('all')

def plot_paths_re(J, theta_histories, title=None, filename=None, rng=(-5.12, 5.12), num=100, cm=cm.rainbow, ):
    """
    Plots the paths taken during optimization.
    J: np.ndarray -> float      function to minimize
    theta_histories: J_histories: np.ndarray(S, N_c, p)
    title: str                  title of plot
    filename: str               filename to save as
    num: int                    how many points to use for contour plot grid
    cm: matplotlib.cm           colormap to use for plotting
    """

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(title)
    x = np.linspace(*rng, num=num)
    y = np.linspace(*rng, num=num)
    X, Y = np.meshgrid(x, y)
    theta = np.array([X, Y])
    Z = np.apply_along_axis(J, 0, theta)
    contour = ax.contour(X, Y, Z, cmap=cm)

    N_re = theta_histories.shape[0]
    N_c  = theta_histories.shape[2]
    colours = ['r', 'g', 'b', 'c']*N_re
    for k in range(N_re):
        generation = theta_histories[k]
        colour = colours[k]
        for theta_history in generation:
            ax.plot(*theta_history.T, alpha=0.5, color=colour)

    if filename is None:
        plt.show()
    else:
        plt.savefig(f'{filename}_theta.pdf')
    plt.close('all')

def evaluate_sb(J, title=None, filename=None, **simulation_args):
    """
    Used to run and generate plots for single-bacterium-optimization.
    J: np.ndarray -> float      function to minimize
    title: str                  title of plot
    filename: str               filename to save as
    simulation_args: kwargs     keyword arguments to be passed to
                                single_bacterium.simulate
    """

    from single_bacterium import simulate
    J_histories = []
    theta_histories = []
    for _ in range(10):
        J_history, theta_history = simulate(J, **simulation_args)
        J_histories.append(J_history)
        theta_histories.append(theta_history)

    plot_J(J_histories, title=title, filename=filename)
    plot_paths(J, theta_histories, title=title, filename=filename)

def evaluate_col(J, title=None, filename=None, **simulation_args):
    """
    Used to run and generate plots for colony-optimization.
    J: np.ndarray -> float      function to minimize
    title: str                  title of plot
    filename: str               filename to save as
    simulation_args: kwargs     keyword arguments to be passed to
                                colony.simulate
    """

    from colony import simulate
    J_histories, theta_histories = simulate(J, **simulation_args)
    plot_J(J_histories, title=title, filename=filename)
    plot_paths(J, theta_histories, title=title, filename=filename)

def evaluate_col_re(J, title=None, filename=None, **simulation_args):
    """
    Used to run and generate plots for colony-optimization with reproduction.
    J: np.ndarray -> float      function to minimize
    title: str                  title of plot
    filename: str               filename to save as
    simulation_args: kwargs     keyword arguments to be passed to
                                colony.simulate
    """

    from colony_reproduction import simulate
    J_histories, theta_histories = simulate(J, **simulation_args)
    plot_J_re(J_histories, title=title, filename=filename)
    plot_paths_re(J, theta_histories, title=title, filename=filename)

def evaluate_col_ed(J, title=None, filename=None, **simulation_args):
    """
    Used to run and generate plots for colony-optimization with reproduction.
    J: np.ndarray -> float      function to minimize
    title: str                  title of plot
    filename: str               filename to save as
    simulation_args: kwargs     keyword arguments to be passed to
                                colony.simulate
    """

    from colony_elimination_dispersal import simulate
    J_histories, theta_histories = simulate(J, **simulation_args)
    N_ed = J_histories.shape[0]
    for l in range(N_ed):
        J_history = J_histories[l]
        plot_J_re(J_history, title=title, filename=f'{filename}_{l}')

        theta_history = theta_histories[l]
        plot_paths_re(J, theta_history, title=title, filename=f'{filename}_{l}')

if __name__ == "__main__":
    """
    Used to produce the figures from the presentation.
    """
    from losses import rastrigin as J

    plot_surface(J,
        filename='presentation/assets/rastrigin',
        rng=(-5.12,5.12),
        num=100,
        cm=cm.rainbow)

    np.random.seed(17)
    evaluate_sb(J,
        title=r'$p$=2, $c$=0.1, $N_c$=100',
        filename='presentation/assets/rastrigin',
        p=2,
        c=0.1,
        N_c=100,
        rng=(-5.12,5.12))

    np.random.seed(17)
    plot_J_cc(title=r'$d_\mathregular{attract}$=0.1, $w_\mathregular{attract}$=0.2, $h_\mathregular{repellant}$=0.1, $w_\mathregular{repellant}$=10',
        filename='presentation/assets/swarming',
        S=3,
        rng=(-5.12, 5.12),
        num=100,
        d_attract=0.1,
        w_attract=0.2,
        h_repellant=0.1,
        w_repellant=10)

    np.random.seed(17)
    evaluate_col(J,
        title=r'$S$=10, $p$=2, $c$=0.1, $N_c$=100' '\n' r'$d_\mathregular{attract}$=0.1, $w_\mathregular{attract}$=0.2, $h_\mathregular{repellant}$=0.1, $w_\mathregular{repellant}$=10',
        filename='presentation/assets/rastrigin_colony',
        S=10,
        p=2,
        c=0.1,
        N_c=100,
        rng=(-5.12,5.12),
        d_attract=0.1,
        w_attract=0.2,
        h_repellant=0.1,
        w_repellant=10)

    np.random.seed(17)
    evaluate_col(J,
        title=r'$S$=10, $p$=2, $c$=0.1, $N_c$=100' '\n' r'$d_\mathregular{attract}$=100, $w_\mathregular{attract}$=0.01, $h_\mathregular{repellant}$=0.4, $w_\mathregular{repellant}$=0.01',
        filename='presentation/assets/rastrigin_colony_tuned',
        S=10,
        p=2,
        c=0.1,
        N_c=100,
        rng=(-5.12,5.12),
        d_attract=100,
        w_attract=0.01,
        h_repellant=0.4,
        w_repellant=0.01)

    np.random.seed(17)
    plot_J_cc(title=r'$S$=3, $p$=2, $d_\mathregular{attract}$=0.1, $w_\mathregular{attract}$=0.2, $h_\mathregular{repellant}$=0.1, $w_\mathregular{repellant}$=10',
        filename='presentation/assets/swarming_tuned',
        S=3,
        rng=(-5.12, 5.12),
        num=100,
        d_attract=100,
        w_attract=0.01,
        h_repellant=0.4,
        w_repellant=0.01)

    np.random.seed(17)
    evaluate_col_re(J,
        title=r'$N_{re}$=4, $S$=10, $p$=2, $c$=0.1, $N_c$=100' '\n' r'$d_\mathregular{attract}$=100, $w_\mathregular{attract}$=0.01, $h_\mathregular{repellant}$=0.4, $w_\mathregular{repellant}$=0.01',
        filename='presentation/assets/rastrigin_colony_re',
        N_re=4,
        S=10,
        p=2,
        c=0.1,
        N_c=100,
        rng=(-5.12,5.12),
        d_attract=100,
        w_attract=0.01,
        h_repellant=0.4,
        w_repellant=0.01)

    np.random.seed(17)
    evaluate_col(J,
        title=r'$S$=10, $p$=2, $c$=0.1, $N_c$=100' '\n' r'$d_\mathregular{attract}$=100, $w_\mathregular{attract}$=0.01, $h_\mathregular{repellant}$=0.4, $w_\mathregular{repellant}$=0.01',
        filename='presentation/assets/rastrigin_colony_400',
        S=10,
        p=2,
        c=0.1,
        N_c=400,
        rng=(-5.12,5.12),
        d_attract=100,
        w_attract=0.01,
        h_repellant=0.4,
        w_repellant=0.01)

    np.random.seed(17)
    evaluate_col_ed(J,
        title=r'$N_{ed}$=2, $p_{ed}$=0.25, $N_{re}$=4, $S$=10, $p$=2, $c$=0.1, $N_c$=100' '\n' r'$d_\mathregular{attract}$=100, $w_\mathregular{attract}$=0.01, $h_\mathregular{repellant}$=0.4, $w_\mathregular{repellant}$=0.01',
        filename='presentation/assets/rastrigin_colony_ed',
        N_ed=4,
        p_ed=0.25,
        N_re=4,
        S=10,
        p=2,
        c=0.1,
        N_c=100,
        rng=(-5.12,5.12),
        d_attract=100,
        w_attract=0.01,
        h_repellant=0.4,
        w_repellant=0.01)
