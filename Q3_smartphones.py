import numpy as np
from scipy.stats import multivariate_normal
import sys
import matplotlib.pyplot as plt

N_FEATURES = 561
N_LABELS = 6 # 1 - 6

def read_in_smartphone_data(x_files, y_files):
    x_data = []
    y_data = []

    for fname in x_files:
        file = open(fname)
        lines = file.readlines()
        for line in lines:
            d = line.split()
            data = [float(i) for i in d]
            x_data.append(data)
        file.close()
    for fname in y_files:
        file = open(fname)
        lines = file.readlines()
        for line in lines:
            y_data.append(int(line.strip('\n')))
        file.close()

    return np.array(x_data), np.array(y_data)

def get_prob(distribution, prior, sample):
    if(distribution == None):
        return 0
    return prior*distribution.pdf(sample)

def classify(distributions, priors, sample):
    results = []
    for i in range(len(distributions)):
        results.append(get_prob(distributions[i], priors[i], sample))
    return np.argmax(results) # This is the label we assign

def classifier(samples, distributions, priors):
    features = samples[:, :-1]
    labels = samples[:, -1]

    predicted = np.zeros(shape=priors.shape)

    confusion_matrix = np.zeros(shape=(N_LABELS, N_LABELS))

    for i in range(len(features)):
        res = classify(distributions, priors, features[i])
        predicted[res] = predicted[res] + 1
        confusion_matrix[int(labels[i]) - 1][res] = confusion_matrix[int(labels[i]) - 1][res] + 1
    return confusion_matrix



if __name__ == '__main__':
    # This must be modified to find the samples
    x_files = ["/home/michael/Documents/NEU/EECE5644/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt",
               "/home/michael/Documents/NEU/EECE5644/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt"]
    y_files = ["/home/michael/Documents/NEU/EECE5644/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt",
               "/home/michael/Documents/NEU/EECE5644/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt"]

    [data, labels] = read_in_smartphone_data(x_files, y_files)
    print(data.shape)
    print(labels.shape)


    class_samples = [[] for i in range(N_LABELS)]
    print(class_samples)
    for i in range(len(data)):
        class_samples[labels[i] - 1].append(data[i])

    n = len(data)

    for class_i in class_samples:
        print(len(class_i))

    class_conditioned_means = []
    for class_i in class_samples:
        if(len(class_i) == 0):
            class_conditioned_means.append(np.array([0]*N_LABELS))
        else:
            class_conditioned_means.append(np.mean(class_i, axis=0))
    print("Means shape:", np.shape(class_conditioned_means))

    priors = np.array([len(x) for x in class_samples])
    priors = priors / n

    print("Priors", priors)
    print(np.sum(priors))

    l = 0.1
    class_conditioned_covariances = []
    for i in range(len(class_samples)):
        class_conditioned_covariances.append(np.zeros(shape=(N_FEATURES, N_FEATURES)))
        if(len(class_samples[i]) != 0):
            for feature_row in class_samples[i]:
                op = np.outer((feature_row - class_conditioned_means[i]), (feature_row - class_conditioned_means[i]))
                class_conditioned_covariances[i] = class_conditioned_covariances[i] + op
            # Regularize the covariances
            class_conditioned_covariances[i] = (1/len(class_samples[i])) * class_conditioned_covariances[i] + l*np.identity(class_conditioned_covariances[i].shape[0])



    class_params = []
    for i in range(len(class_conditioned_means)):
        any_samples = True
        if(len(class_samples[i]) == 0):
            any_samples = False
        class_params.append([class_conditioned_means[i], class_conditioned_covariances[i], any_samples])

    distributions = []
    for params in class_params:
        if(not params[-1]):
            distributions.append(None)
        else:
            distributions.append(multivariate_normal(params[0], params[1]))

    labels = labels.reshape((len(labels), 1))
    rows = np.append(data, labels, axis=1)
    res = classifier(rows, distributions, priors)
    np.set_printoptions(threshold=sys.maxsize)
    for i in res:
        for j in i:
            print(int(j), ',\t', end="")
        print()

    n_errors = 0
    for i in range(len(res)):
        for j in range(len(res[i])):
            if(i != j):
                n_errors = n_errors + res[i][j]
    print("Number of errors:", n_errors)
    print("p_error", n_errors / len(data))

    plt.scatter(rows[:, 0], rows[:, 1])
    plt.show()


    plt.scatter(rows[:,1], rows[:, 2])
    plt.show()
