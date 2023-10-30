import numpy as np
import scipy.stats


def multi_gauss(x, means, covariances):
    d = len(means)
    norm = 1 / (((2 * np.pi) ** (d / 2)) * (np.linalg.det(covariances)) ** (1 / 2))
    exponent = (-1 / 2) * (((x - means).T.dot(np.linalg.inv(covariances))).dot((x - means)))
    return norm * np.exp(exponent)


def getLogLikelihood(means, weights, covariances, x):
    n, d = x.shape  # nbr of datapoints, dimension
    k = len(means)  # nbr of gaussian-components

    log_likelihood = 0.0

    for point in x:
        point_log_likelihood = 0.0
        for i in range(k):
            mean = means[i]
            cov = covariances[:, :, i]
            weight = weights[i]

            # component_pdf = scipy.stats.norm(mean, cov).cdf(point)
            component_pdf = multi_gauss(point, mean, cov)
            weighted_likelihood = weight * component_pdf
            point_log_likelihood += weighted_likelihood

        log_likelihood += np.log(point_log_likelihood)

    return log_likelihood
