'''
k-means++

This module contains the k-means++ algorithm
and supporting functions.
'''
import numpy as np


def min_distance_update(obs_mat, cent, min_distances):
    """
    update the distances array in-place.
    :param obs_mat: Observation matrix
    :param cent: the last centroid that was found with probability func
    :param min_distances: cell i is the distance between point i and the nearest centroid.
    :return: NULL
    """

    # get the distance between data point  in each row and the current centroid
    temp_distance = np.sum((obs_mat - cent)**2, axis=1)

    # if this distance is minimal for i, set it as the new min distance for i
    smaller = np.argwhere(temp_distance < min_distances)
    min_distances[smaller] = temp_distance[smaller]


def probability(distances_array, N):
    """
    Select the next centroid randomly.
    the probability of selecting point j is proportional
    to its distance from the nearest centroid.

    :param distances_array:
    :param N: number of observations
    :return: an index of a new centroid
    """
    # an array with the possible selections
    obs_indexes = np.arange(0, N, 1)

    # an array representing the probability for each selection
    prob_array = distances_array / np.sum(distances_array)  # zero division not possible for out inputs

    # select an index with regards to the selection probabilities array
    return np.random.choice(obs_indexes, 1, p=prob_array)[0]


def kmeans_pp(K, N, obs_mat):
    """
    The k-means++ algorithm
    :param K: number of Clusters
    :param N: number of data points
    :param obs_mat: Observation matrix - datapoint
    :return: array where cluster[i] is an index 0-(N-1) indicates that obs_mat[i] is a centroid
    """
    np.random.seed(0)

    # step 1: select the first centroid randomly
    next_centroid = np.random.choice(N)

    # additional initializations
    min_distances = np.full(N, np.inf)
    clusters = np.zeros(K)

    # step 2: select the rest of the centroids
    for i in range(K):

        # update the clusters array and obtain the current centroid
        clusters[i] = next_centroid
        cent = obs_mat[next_centroid]

        # track the min distances between points & centroids
        min_distance_update(obs_mat, cent, min_distances)

        # select the next centroid with varying probabilities as required
        next_centroid = probability(min_distances, N)

    return clusters
