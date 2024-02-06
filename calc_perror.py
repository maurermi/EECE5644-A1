from scipy.integrate import nquad
from scipy.stats import multivariate_normal
import numpy as np
import random


N_SAMPLES = 10000

mu1 = np.array([-1, -1, -1, -1])
mu2 = -1 * mu1

sigma1 = np.array([[5, 3, 1, -1], [3, 5, -2, -2], [1, -2, 6, 3], [-1, -2, 3, 4]])
sigma2 = np.array([[1.6, -0.5, -1.5, -1.2], [-0.5, 8, 6, -1.7], [-1.5, 6, 6, 0], [-1.2, -1.7, 0, 1.8]])

pL0 = 0.35
pL1 = 0.651

g0 = multivariate_normal(mu1, sigma1)
g1 = multivariate_normal(mu2, sigma2)

def myfun(x1, x2, x3, x4):
    x = [x1, x2, x3, x4]
    return min(pL0*g0.pdf(x), pL1*g1.pdf(x))

def find_min_perror():
    opts = {'epsabs':1.e-2}
    calc = nquad(myfun,ranges=[[-5,5],[-5,5],[-5,5],[-5,5]],opts=opts)
    return calc

if __name__ == '__main__':
    np.random.seed(42)
    print(find_min_perror())


