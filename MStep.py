import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, x):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussian
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussian
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).
    gamma = np.array(gamma)
    n, k = gamma.shape
    d = x.shape[-1]

    loglikelihood = 0.0
    means = np.zeros((k, d))
    weights = np.zeros((1, k))
    covariances = np.zeros((d, d, k))

    for j in range(k):
        w_sum = np.sum(gamma[:, j][:, np.newaxis] * x, axis=0)
        total_resp = np.sum(gamma[:, j])
        means[j] = w_sum / total_resp

    for j in range(k):
        w_sum = np.sum(gamma[:, j] @ (x - means[j]) @ (x - means[j]).T, axis=0)
        total_resp = np.sum(gamma[:, j])
        covariances[:, :, j] = w_sum / total_resp

    updated_weights = np.sum(gamma, axis=0) / n

    return updated_weights, means, covariances, loglikelihood
