from colony_reproduction import simulate as col_re
from colony import simulate as col
from losses import rastrigin
from plot import plot_J, plot_paths, plot_J_re, plot_paths_re
import numpy as np

# np.random.seed(17)
J_history_col_re, theta_history_col_re = col_re(rastrigin)

# np.random.seed(17)
J_history_col, theta_history_col = col(rastrigin, N_c=400)

J_col_plot = plot_J(J_history_col)
path_col_plot = plot_paths(rastrigin, theta_history_col)

J_col_re_plot = plot_J_re(J_history_col_re)
path_col_re_plot = plot_paths_re(rastrigin, theta_history_col_re)


J_diff = J_history_col.flatten() - J_history_col_re.flatten()
