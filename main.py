"""
Main Module:

This program generates data points.
The points are then clustered using two algorithms:
Spectral Clustering and K-means Clustering.
Finally, the program outputs a performance comparison.
"""

import input_output_funcs as io
import spectral_funcs as sp
import kmeans_pp as kpp
import mykmeanssp as kc
import numpy as np


# maximum capacity properties
MAX_ITER = 300
N_2D = 550
K_2D = 11
N_3D = 500
K_3D = 10


def input_handling():
    """
    get parameters from user and generate data from sk_learn according to its inputs
    :return: Tuple of  (n, K, d, data, labels, Random)
    """
    # get program parameters from user
    k_in, n_in, Random = io.get_values()

    # Generation of n data points (2d or 3d)
    n, K, d, data, labels = io.sk_generator(n_in, k_in, Random)
    # Print description of Inputs and Random choices
    io.print_description(k_in, n_in, K, n, Random, d)

    return n, K, d, data, labels, Random


def spectral_clusterin(data, K, Random):
    """
    Spectral clustering techniques make use of the spectrum (eigenvalues) of the similarity
    matrix of the data to perform dimensionality reduction before clustering in fewer
    dimensions. The similarity matrix is provided as an input and consists of a quantitative
    assessment of the relative similarity of each pair of points in the dataset.

    :param data: Data of 3d or 2d
    :param K: number of clustering from user input
    :param Random:  if True determine k with eigen heuristic algoritm
    :return: Tuple of (T, k)
    """
    # Create the Weighted Adjacency Matrix
    W = sp.create_weighted_matrix(data)

    # Create D^(-1/2). D is the Diagonal Degree Matrix
    D_neg_sqrt = sp.create_diagonal_matrix(W)

    # Create the Laplacian Matrix
    L = sp.create_laplacian_matrix(D_neg_sqrt, W)

    # Iterative QR to get A' and Q'
    Atag, Qtag = sp.qr_iterator(L)

    #detemin k from user
    k=K

    # if Random = True, use the Eigengap Heuristic to select k
    if Random:
        k = sp.eigengap(Atag)

    # Get the eigenvector matrix U
    U = sp.get_spectral_matrix(Atag, Qtag, k)

    # Get T by Normalizing U's rows to unit length
    T = sp.normalize_matrix(U)

    return T, k


def kmeans_intialize_centroids(k, n, data, T):
    """
    Use k-means++ to get the initial clusters from T and from original data.

    :param k: number of Clusters
    :param n: number of Observations
    :param data: data of n Observations from 3 or 2 dimension
    :param T: Normalized matrix from spectral clustering
    :return: sp_initial, km_initial
    """
    # cast to a list to be fed to a c extension
    sp_initial = kpp.kmeans_pp(k, n, T).astype(int).tolist()

    # cast to a list to be fed to a c extension
    km_initial = kpp.kmeans_pp(k, n, data).astype(int).tolist()

    return sp_initial, km_initial


def calc_final_clusters(sp_initial, km_initial):
    """
    Calculating final clusters labels using K-means with C extention

    :param sp_initial: array of initialized centroids corresponding to T perform by Kmeans++
    :param km_initial: array of initialized centroids corresponding to Datapoints perform by Kmeans++
    :return: Tuple of (spectral_clusters, kmeans_clusters)
    """
    sp_final = kc.kmeans(T.tolist(), sp_initial, k, n, k, MAX_ITER)
    km_final = kc.kmeans(data.tolist(), km_initial, k, n, d, MAX_ITER)
    if km_final is None or sp_final is None:
        exit(1)
    spectral_clusters = np.array(sp_final)
    kmeans_clusters = np.array(km_final)

    return spectral_clusters, kmeans_clusters


def output_files(d, k, K, data, labels, spectral_clusters, kmeans_clusters):
    """
    Output 3 files of the analize data:

    datapoint.txt - will contain the generated data from Random Data Points Generation.

    clusters.txt - will contain the computed clusters from both algorithms. The first
    value will be the number of clusters k that was use.

    Clustering visualization - depending on the dimension of the data used, the plot will
    be either 2-dimensional (with X, Y axis) or 3-dimensional (with X, Y, Z axis). Each
    calculated cluster, resulting from the Normalized Spectral Clustering algorithm and
    the K-means algorithm, will be colored differently.

    :param d: dimension of 2 or 3
    :param k: number of clusters corresponding to eigengap herustic (if Random is True)
    :param K: number of clusters corresponding to user input
    :param data: matrix of d dimension
    :param labels: original cluster labels of data
    :param spectral_clusters: spectral cluster labels
    :param kmeans_clusters: k-means cluster labels
    :return:NULL
    """
    # output file of the generated data
    io.out_data_txt(d, data, labels)

    # output file of the calculated clusters
    io.out_clusters_txt(k, spectral_clusters, kmeans_clusters)

    # plot and compare the clusters from both algorithms
    io.out_clusters_pdf(data, d, spectral_clusters, kmeans_clusters, labels, k, K)


if __name__ == "__main__":

    n, K, d, data, labels, Random = input_handling()

    T, k = spectral_clusterin(data, K, Random)

    sp_initial, km_initial = kmeans_intialize_centroids(k, n, data, T)

    spectral_clusters, kmeans_clusters = calc_final_clusters(sp_initial, km_initial)

    output_files(d, k, K, data, labels, spectral_clusters, kmeans_clusters)



