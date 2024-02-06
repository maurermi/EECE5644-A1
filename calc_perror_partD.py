from scipy.integrate import nquad
from scipy.stats import multivariate_normal
import numpy as np
import random


N_SAMPLES = 10000

mu1 = np.array([-1, -1, -1, -1])
mu2 = -1 * mu1

sigma1 = np.array([[2.8257724, 0.77577852, -0.62412386, -1.16485146], [0.77577852, 6.97845812, 3.1279159, -1.82801788], [-0.62412386, 3.1279159, 6.00617411, 1.06367039], [-1.16485146, -1.82801788, 1.06367039, 2.59658631]])

pL0 = 0.35
pL1 = 0.651

g0 = multivariate_normal(mu1, sigma1)
g1 = multivariate_normal(mu2, sigma1)

def myfun(x1, x2, x3, x4):
    x = [x1, x2, x3, x4]
    return min(pL0*g0.pdf(x), pL1*g1.pdf(x))

def find_min_perror():
    opts = {'epsabs':1.e-2}
    calc = nquad(myfun,ranges=[[-5,5],[-5,5],[-5,5],[-5,5]],opts=opts)
    return calc

if __name__ == '__main__':
    np.random.seed(3)
    print(find_min_perror())


