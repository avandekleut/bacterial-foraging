import numpy as np

def colony(f, p=2, S=10, N_ed=2, N_re=3, N_c=5, N_s=10, p_ed=0.05, c=lambda i: 0.1, rng=(-1, 1), d_attract=0.1, w_attract=0.2, h_repellant=0.1, w_repellant=10):
    """
    f: np.ndarray -> float          the function to minimize
    p: int > 0                      the number of dimensions
    S: int > 0                      the size of the colony
    N_ed: int > 0                   the number of elimination-dispersal events
    N_re: int > 0                   the number of reproduction events
    N_s: int > 0                    the number of chemotactic steps
    p_ed: float in [0, 1]           the probability of an e-d event
    c: int -> float                 maps the bacterium index to step size
    rng: tuple(float, float)        the range of values used for initialization
    d_attract: float >= 0           the depth of the attraction between
                                    bacterium
    w_attract: float >= 0           the width of the attraction between
                                    bacterium
    h_repellant: float >= 0         the height of the repellant force between
                                    bacterium
    w_repellant: float >= 0         the width of the repellant force between
                                    bacterium
    """

    # The authors index into a tensor J using i, j, k, l.
    # i = bacterium index
    # j = chemtactic step index
    # k = reproductive step index
    # l = elimination-dispersal step index
    J = np.zeros((S, N_c, N_re, N_ed)) #J(i, j, k, l)

    # Theta is indexed similarly, except that it has an extra dimension to hold
    # the information about positions.
    theta = np.zeros((p, S, N_c, N_re, N_ed))

    # We want to balance how the authors index with how numpy indexing works.
    # We really would want to index like J(l, k, j) to get the last dimension
    # for the ith bacterium. Instead, we will have to do J(:, j, k, l) to get
    # it which is not a memory efficient slice.

    # This observation similarly holds for theta.

    # initialize theta to random values in rng
    theta[:, :, 0, 0, 0] = rng[0] + np.random.rand(p, S)*(rng[1] - rng[0])

    # initialize step sizes
    C = np.vectorize(c)(np.arange(S))

    # define a function for cell-to-cell interactions
    def J_cc(theta_i, j, k, l):
        result = 0
        for i in range(S):
            squared_distance = ((theta_i - theta[:, i, j, k, l])**2).sum()
            result += -d_attract*np.exp(-w_attract*squared_distance) \
                + h_repellant*np.exp(-w_repellant*squared_distance)
        return result

    for l in range(N_ed-1):
        # print(f"l: {l}")
        # Elimination-dispersal
        for k in range(N_re-1):
            # print(f"\t k: {k}")
            # Reproduction
            for j in range(N_c-1):
                # print(f"\t\t j: {j}")
                # Chemotaxis
                for i in range(S):
                    # Compute the loss
                    J[i, j, k, l] = f(theta[:, i, j, k, l])
                    # print(f"\t\t\t theta[:, {i}, {j}, {k}, {l}] = {theta[:, i, j, k, l]}")
                    # print(f"\t\t\t J[{i}, {j}, {k}, {l}] = {J[i, j, k, l]}")

                    # Save it
                    J_last = J[i, j, k, l]

                    # Move in a random direction (tumble)
                    delta_i = 2*np.random.rand(p) - 1

                    theta[:, i, j+1, k, l] = theta[:, i, j, k, l] \
                                + C[i] * delta_i / np.sqrt(delta_i.T @ delta_i)

                    # Compute the loss for the new position
                    J[i, j+1, k, l] = f(theta[:, i, j+1, k, l]) + J_cc(theta[:, i, j+1, k, l], j, k, l)

                    # For each swim step (run)
                    for m in range(N_s):
                        # If moving in the direction delta_i improves loss
                        if J[i, j+1, k, l] < J_last:
                            # Update best loss
                            J_last = J[i, j+1, k, l]

                            # Continue to move in direction of improvement
                            theta[:, i, j+1, k, l] = theta[:, i, j+1, k, l] \
                                    + C[i] * delta_i / np.sqrt(delta_i.T @ delta_i)

                            # Compute the new loss
                            J[i, j+1, k, l] = f(theta[:, i, j+1, k, l]) + J_cc(theta[:, i, j+1, k, l], j, k, l)
                        else:
                            # Stop iterating
                            m = N_s
                print(J[:, j, k, l].mean())
                theta[:, :, j, k+1, l] = theta[:, :, j+1, k, l]
            theta[:, :, :, k, l+1] = theta[:, :, :, k+1, l]




colony(lambda x:(x**2).sum(), N_s=100)
