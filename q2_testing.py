from scipy.integrate import nquad
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt


N_SAMPLES = 10000

# Passing in gaussians for speed
def gen_samples(pl0, g0, g1):
    res = []
    dist = np.random.rand(N_SAMPLES)
    for i in range(N_SAMPLES):
        if(dist[i] < pl0):
            res.append([g0.rvs(size = 1), 0])
        else:
            res.append([g1.rvs(size = 1), 1])
    return res

def decision_maker(x, g0, g1, gamma):
    return (g1.pdf(x) / g0.pdf(x)) > gamma

def classifier(samples, g0, g1, gamma):
    # This matrix is defined as follows (guessed:actual)
    # [ 0:0, 0:1 ]
    # [ 1:0, 1:1 ]
    classifications = [[0, 0], [0, 0]]

    for sample in samples:
        guess = decision_maker(sample[0], g0, g1, gamma)
        # Since we defined classifications as such, we can do this
        classifications[guess][sample[1]] = classifications[guess][sample[1]] + 1
        # Case 1: L = 1 guessed, L = 1 actual
        # Case 2: L = 0 guessed, L = 0 actual
        # Case 3: L = 1 guessed, L = 0 actual
        # Case 4: L = 0 guessed, L = 1 actual
    return classifications

def myfun(x1, x2, x3, x4):
    x = [x1, x2, x3, x4]
    return min(pL0*g0.pdf(x), pL1*g1.pdf(x))

def find_min_perror():
    opts = {'epsabs':1.e-2}
    calc = nquad(myfun,ranges=[[-5,5],[-5,5],[-5,5],[-5,5]],opts=opts)
    return calc


def k(beta, mu1, mu2, sigma1, sigma2):
    res = beta*(1 - beta)/2
    res = res * (mu1 - mu2)
    # Putting the transpose here because mu1, mu2 are defined as row vectors
    res = np.matmul(res , np.matmul(np.linalg.inv((1 - beta)*sigma1 + beta*sigma2), np.transpose(mu1 - mu2)))
    res = res + 0.5*np.log( np.linalg.det(((1 - beta) * sigma1 + beta * sigma2)) / ((np.linalg.det(sigma1)**(1-beta)) * (np.linalg.det(sigma2))**beta))
    return res


def estimate_cov(samples, mu1, mu2):
    total = np.zeros(shape=(4, 4))
    for sample in samples:
        if(sample[1] == 0):
            total = total + (np.outer((sample[0] - mu1), (sample[0] - mu1)))
        else:
            total = total + (np.outer((sample[0] - mu2), (sample[0] - mu2)))
    return total / N_SAMPLES

if __name__ == '__main__':
    np.random.seed(1)

    mu1 = np.array([-1, -1, -1, -1])
    mu2 = -1 * mu1

    sigma1 = np.array([[5, 3, 1, -1], [3, 5, -2, -2], [1, -2, 6, 3], [-1, -2, 3, 4]])
    sigma2 = np.array([[1.6, -0.5, -1.5, -1.2], [-0.5, 8, 6, -1.7], [-1.5, 6, 6, 0], [-1.2, -1.7, 0, 1.8]])

    pL0 = 0.35
    pL1 = 0.65

    g0 = multivariate_normal(mu1, sigma1)
    g1 = multivariate_normal(mu2, sigma2)

    samples = gen_samples(pL0, g0, g1)

    # Q2a

    # gammas = np.linspace(0.25, 2, 50)
    gammas = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.33, 0.4, 0.5, 0.54, 0.6, 0.66, 0.75, 0.8, 0.9, 1, 1.2, 1.5, 1.8, 2, 2.5, 2.8, 3, 4, 6, 9, 12, 500, np.inf]
    max = 0
    max_gam = -1
    fpr_samps = []
    tpr_samps = []
    p_errors = []
    min_error = np.inf
    min_gamma_error = -1
    index = 0
    mindex = -1
    for gamma in gammas:
        classification = classifier(samples, g0, g1, gamma)
        positives = classification[0][1] + classification[1][1]

        true_positives = 0
        if(positives != 0):
            true_positives = classification[1][1] / positives

        false_positives = 0
        negatives = classification[1][0] + classification[0][0]
        if(negatives != 0):
            false_positives = classification[1][0] / negatives

        if(true_positives + (1 - false_positives) > max):
            max = true_positives + 1 - false_positives
            max_gam = gamma
        tpr_samps.append(true_positives)
        fpr_samps.append(false_positives)
        error = (classification[1][0] + classification[0][1]) / N_SAMPLES
        if(error < min_error):
            min_error = error
            min_gamma_error = gamma
            mindex = index

        index = index + 1
        p_errors.append(error)

    print("Q2A")
    print("Q2A Min observed error", min_gamma_error, min_error)
    plt.title("ROC Curve for gamma based decision rule")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr_samps, tpr_samps)
    plt.plot(fpr_samps[mindex], tpr_samps[mindex], 'r*')
    plt.savefig("ROC_Curve_Q2a_alt.png")

    plt.clf()
    plt.title("Empirical Error vs. Gamma")
    plt.xlabel("P(Error)")
    plt.ylabel("Gamma")
    plt.plot(p_errors, gammas)
    plt.savefig("P_Error_vs_Gamma_Q2a_alt.png")
    print()

    # Q2b

    best_error = np.inf
    beta_mindex = -1
    betas = np.linspace(0, 1, 50)
    results = []
    for beta in betas:
        res = np.exp(-1 * k(beta, mu1, mu2, sigma1, sigma2))
        if(res < best_error):
            best_error = res
            beta_mindex = beta
        results.append(res)

    plt.clf()
    plt.title("Chernoff Bound Curve")
    xlab = "Beta (opt = {best:.2f})".format(best = beta_mindex)
    plt.xlabel(xlab)
    plt.ylabel("P(Error)")
    plt.plot(betas, results)
    plt.plot(beta_mindex, best_error, 'r*')
    plt.savefig("Q2b.png")


    print("Chernoff:", beta_mindex, best_error)
    print("Bhattacharyya", 0.5, np.exp(-1 * k(0.5, mu1, mu2, sigma1, sigma2)))


    # Q2c
    sigma1 = np.array([[5, 0, 0, 0],
                       [0, 5, 0, 0],
                       [0, 0, 6, 0],
                       [0, 0, 0, 4]])
    sigma2 = np.array([[1.6, 0, 0, 0],
                       [0, 8, 0, 0],
                       [0, 0, 6, 0],
                       [0, 0, 0, 1.8]])
    g0 = multivariate_normal(mu1, sigma1)
    g1 = multivariate_normal(mu2, sigma2)

    #gammas = np.linspace(0.5, 1.5, 50)
    gammas = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.33, 0.4, 0.5, 0.6, 0.66, 0.75, 0.8, 0.9, 1, 1.2, 1.5, 1.8, 2, 2.5, 2.8, 3, 4, 6, 9, 12]
    max = 0
    max_gam = -1
    index = 0
    mindex = -1
    fpr_samps = []
    tpr_samps = []
    p_errors = []
    min_error = np.inf
    min_gamma_error = -1
    for gamma in gammas:
        classification = classifier(samples, g0, g1, gamma)
        positives = classification[0][1] + classification[1][1]

        true_positives = 0
        if(positives != 0):
            true_positives = classification[1][1] / positives

        false_positives = 0
        negatives = classification[1][0] + classification[0][0]
        if(negatives != 0):
            false_positives = classification[1][0] / negatives

        if(true_positives + (1 - false_positives) > max):
            max = true_positives + 1 - false_positives
            max_gam = gamma
        tpr_samps.append(true_positives)
        fpr_samps.append(false_positives)
        error = (classification[1][0] + classification[0][1]) / N_SAMPLES
        if(error < min_error):
            min_error = error
            min_gamma_error = gamma
            mindex = index

        index = index + 1

        p_errors.append(error)

    print("Q2C")

    print("Q2C Min observed error", min_gamma_error, min_error)
    plt.clf()
    plt.title("ROC Curve Part C")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr_samps, tpr_samps)
    plt.plot(fpr_samps[mindex], tpr_samps[mindex], 'r*')
    plt.savefig("ROC_Curve_Q2c.png")

    plt.clf()
    plt.title("Empirical Error vs. Gamma (Part C)")
    plt.xlabel("P(Error)")
    plt.ylabel("Gamma")
    plt.plot(p_errors, gammas)
    plt.savefig("P_Error_vs_Gamma_Q2c.png")
    print()

    print()


    # Q2D

    print("Q2D")
    sigma1 = estimate_cov(samples, mu1, mu2)
    print("Covariance Matrix:", sigma1)

    g0 = multivariate_normal(mu1, sigma1)
    g1 = multivariate_normal(mu2, sigma1)

    #gammas = np.linspace(2, 4, 20)
    gammas = [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.33, 0.4, 0.5, 0.6, 0.66, 0.75, 0.8, 0.9, 1, 1.2, 1.5, 1.8, 2, 2.5, 2.8, 3, 3.2, 3.5, 3.8, 4, 6, 9, 12]
    max = 0
    max_gam = -1
    index = 0
    mindex = -1
    fpr_samps = []
    tpr_samps = []
    p_errors = []
    min_error = np.inf
    min_gamma_error = -1
    for gamma in gammas:
        classification = classifier(samples, g0, g1, gamma)
        positives = classification[0][1] + classification[1][1]

        true_positives = 0
        if(positives != 0):
            true_positives = classification[1][1] / positives

        false_positives = 0
        negatives = classification[1][0] + classification[0][0]
        if(negatives != 0):
            false_positives = classification[1][0] / negatives

        if(true_positives + (1 - false_positives) > max):
            max = true_positives + 1 - false_positives
            max_gam = gamma
        tpr_samps.append(true_positives)
        fpr_samps.append(false_positives)
        error = (classification[1][0] + classification[0][1]) / N_SAMPLES
        if(error < min_error):
            min_error = error
            min_gamma_error = gamma
            mindex = index

        index = index + 1
        p_errors.append(error)
        #print("Gamma:", gamma, "TPR, FPR", true_positives, 1 - false_positives)

    print("Q2D Min observed error", min_gamma_error, min_error)
    # plt.plot(fpr_samps, tpr_samps)
    # plt.show()

    # plt.plot(p_errors, gammas)
    # plt.show()
    plt.clf()
    plt.title("ROC Curve Part D")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr_samps, tpr_samps)
    plt.plot(fpr_samps[mindex], tpr_samps[mindex], 'r*')
    plt.savefig("ROC_Curve_Q2d.png")

    plt.clf()
    plt.title("Empirical Error vs. Gamma (Part D)")
    plt.xlabel("P(Error)")
    plt.ylabel("Gamma")
    plt.plot(p_errors, gammas)
    plt.savefig("P_Error_vs_Gamma_Q2d.png")
    print()


    # Q2E
    betas = np.linspace(0.01, 20, 100)
    gammas = [(pL0)/(pL1) * (1 / beta) for beta in betas]
    false_positives_list = []
    false_negatives_list = []

    sigma1 = np.array([[5, 3, 1, -1], [3, 5, -2, -2], [1, -2, 6, 3], [-1, -2, 3, 4]])
    sigma2 = np.array([[1.6, -0.5, -1.5, -1.2], [-0.5, 8, 6, -1.7], [-1.5, 6, 6, 0], [-1.2, -1.7, 0, 1.8]])

    g0 = multivariate_normal(mu1, sigma1)
    g1 = multivariate_normal(mu2, sigma2)

    for gamma in gammas:
        classification = classifier(samples, g0, g1, gamma)
        false_positives_list.append(classification[0][1])
        false_negatives_list.append(classification[1][0])
    incurred_cost = [false_positives_list[i] + false_negatives_list[i]*betas[i] for i in range(len(betas))]
    print("Minimizing beta", betas[np.argmin(incurred_cost)])

    plt.clf()
    plt.title("Minimum Risk Decision Boundary")
    plt.xlabel("B")
    plt.ylabel("Gamma")
    plt.plot(betas, gammas, label="Gamma")
    plt.plot(betas, incurred_cost, label="Cost")
    plt.legend()
    plt.savefig("Q2E.png")
