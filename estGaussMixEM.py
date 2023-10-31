import numpy as np
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def estGaussMixEM(data, k, n_iters, epsilon):
    # EM algorithm for estimation gaussian mixture mode
    #
    # INPUT:
    # data           : input data, N observations, D dimensional
    # K              : number of mixture components (modes)
    #
    # OUTPUT:
    # weights        : mixture weights - P(j) from lecture
    # means          : means of gaussian
    # covariances    : covariance matrices of gaussian

    weights = np.ones(k) / k

    kmeans = KMeans(n_clusters=k, n_init=10).fit(data)
    cluster_idx = kmeans.labels_

    means = kmeans.cluster_centers_
    covariances = np.zeros((2, 2, 3))
    # Create initial covariance matrices
    for j in range(k):
        data_cluster = data[cluster_idx == j]
        min_dist = np.inf
        for i in range(k):
            # compute sum of distances in cluster
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
            if dist < min_dist:
                min_dist = dist
                covariances[:, :, j] = np.eye(2) * min_dist

    for iteration in range(n_iters):
        loglikelihood, gamma = EStep(np.array(means), np.array(covariances), np.array(weights), np.array(data))

        weights, means, covariances, loglikelihood = MStep(gamma, data)

    return [weights, means, covariances]
