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
