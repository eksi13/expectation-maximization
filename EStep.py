import numpy as np
from getLogLikelihood import multi_gauss


def EStep(means, covariances, weights, x):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussian
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussian
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussian.
    n, d = x.shape
    k = len(means)

    gamma = np.zeros((n, k))

    for i in range(n):
        for j in range(k):
            component_pdf = multi_gauss(x[i], means[j], covariances[:, :, j])
            gamma[i][j] = weights[j] * component_pdf

    gamma /= np.sum(gamma, axis=1, keepdims=True)

    loglikelihood = np.sum(np.log(np.sum(gamma * weights, axis=1)))

    return [loglikelihood, gamma]
