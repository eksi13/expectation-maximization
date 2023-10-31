import numpy as np


def regularize_cov(covariance, epsilon):
    x, y = covariance.shape
    id_matrix = np.identity(x) * epsilon

    regularized_cov = covariance + id_matrix
    # regularize a covariance matrix, by enforcing a minimum
    # value on its singular values. Explanation see exercise sheet.
    #
    # INPUT:
    #  covariance: matrix
    #  epsilon:    minimum value for singular values
    #
    # OUTPUT:
    # regularized_cov: reconstructed matrix
    return regularized_cov
