"""
Input and Output Functions:

The input functions in this module receive input
from the user and generate datapoints accordingly.

The output functions create txt and pdf files which
present the outcomes of two clustering algorithms
and compare their performance.
"""
import argparse
import random
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import main

def print_description(k_in, n_in, K, n, Random, d):
    """
    print a description about algoritm capacity and description
    of Random choices that have been made for the specific run

    :param k_in: k by user
    :param n_in: n by user
    :param K: if !Random k=K ; else origin K
    :param n: number of generated rows in data
    :param d: dimension of data
    :return: NULL
    """
    # Maximum capacity prints
    print("\n" + '='*50)
    print(' '*12 +"Project Maximum Capacity\n")
    print("For 2D data: n="+str(main.N_2D)+" points, k="+str(main.K_2D)+" clusters")
    print("For 3D data: n=" + str(main.N_3D) + " points, k=" + str(main.K_3D) + " clusters")
    print('='*50)

    # Generated and input values prints
    if Random:
        print(f"Random values: k={K} n={n} d={d}")
        print('='*50)
        print("Note that final k of algorithm is chosen after Eigengap Heuristic\n")
    else:
        print(' '*19 + "Data Values\n")
        print(f"User inputs: k={k_in} n={n_in} Random={Random}")
        print(f"Random values: d={d}")
        print('='*50)


def get_values():
    """
    get the program parameters from the user
    K - Number of Clusters , N - Number of datapoint, RANDOM - default set to True.
    :return: tuple of(K, N, RANDOM)
    """
    # receive inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('K', type=int)
    parser.add_argument('N', type=int)
    parser.add_argument('--Random',dest='RANDOM',action='store_true',required=False)
    parser.add_argument('--no-Random',dest='RANDOM',action='store_false',required=False)
    parser.set_defaults(RANDOM=True) # set default value for Random to True
    args = parser.parse_args()

    K = args.K
    N = args.N
    RANDOM = args.RANDOM

    # assertions on the input
    if K >= N or K <= 0 or N <= 0:
        print("Incorrect arguments. Make sure K<N and 0<K,N.")
        exit(1)

    return K, N, RANDOM


def draw_variables(vector_dimension):

    """
    This is an assisting function for sk_generator()
    draw the actual program parameters n as number of
    datapoints and K as number of Clusters

    :param vector_dimension: 2 or 3
    :return: tuple of n and K
    """
    # if the vectors to be generated are 2D, use the 2D max capacities
    if vector_dimension == 2:
        # draw n randomly based on N_2D
        n = random.randint(main.N_2D//2, main.N_2D)
        # draw K randomly based on K_2D
        K = random.randint(main.K_2D//2, main.K_2D)

    # else if the vectors to be generated are 3D, use the 3D max capacities
    else:
        # draw n randomly based on N_3D
        n = random.randint(main.N_3D//2, main.N_3D)
        # draw K randomly based on K_3D
        K = random.randint(main.K_3D//2, main.K_3D)

    return n, K


def sk_generator(n_in, k_in, Random):

    """
    draw the program parameters
    then generate the data points
    its centers and n,k,d based on user or Random choice

    :param n_in: Number of datapoints from user
    :param k_in: Number of clusters by user
    :param Random: a Boolean typed variable with default value of True that indicates the way
    the data is to be generated.
    :return: tuple of (n, K, d, data, labels)
    """
    # randomly draw the vector dimensions (2D or 3D)
    d = random.randint(2, 3)

    # determine N, K:
    # if Random==True, draw randomly based on the maximum capacity
    if Random:
        n, K = draw_variables(d)
    # if Random==False, use the supplied parameters
    else:
        n, K = n_in, k_in

    # generate the data points using sklearn
    # obtain the actual cluster membership labels for the data
    data, labels = make_blobs(n_samples=n,  n_features=d, centers=K)

    return n, K, d, data, labels


def out_data_txt(d, datapoints, labels):
    """
    Output the generated data points and cluster membership to txt file

    :param d: dimension of data (2 or 3)
    :param datapoints: genarated datapoints
    :param labels: lables[i] corresponding to the cluster of datapoint[i]
    :return: NULL
    """
    # order the data and labels according to the display format
    points_and_labels = np.column_stack((datapoints, labels))

    # create the txt file
    np.savetxt('data.txt', points_and_labels, fmt=','.join(['%f']*d + ['%i']))


def out_clusters_txt(k, spectral_clusters, kmeans_clusters):
    """
    Output the calculated k and the cluster membership labels
    under spectral clustering and k-means clustering into txt file

    :param k:  Number of Clusters
    :param spectral_clusters:  First lables for clustering
    :param kmeans_clusters: Second lables for clustering
    :return: NULL
    """
    # create the txt file
    with open('clusters.txt', 'w') as outfile:

        # output k
        np.savetxt(outfile, np.array([k]), fmt='%i')

        # output the spectral cluster members
        # j is in row i <=> data point j is in cluster i
        for i in range(k):
            np.savetxt(outfile, np.where(spectral_clusters == i), fmt='%i', delimiter=',')

        # output the k-means cluster members
        # j is in row i <=> data point j is in cluster i
        for i in range(k):
            np.savetxt(outfile, np.where(kmeans_clusters == i), fmt='%i', delimiter=',')


def jaccard_measure(calculated_labels, true_labels):
    """
    This function is used by out_clusters_pdf() to present
    the performance of the spectral clustering and the
    k-means algorithms.

    calculate the jaccard score of a clustering algorithm's result
    this is the number of pairs that are in the same cluster
    in both the calculated and actual solutions
    divided by the number of pairs that are in the same cluster
    in at least one solution.

    :param calculated_labels: lables based on algorithm clustering
    :param true_labels: original makeblobs lables
    :return: jaccard score for clustering data.
    """
    # get dimensions
    n = true_labels.shape[0]

    # (i,j) is True iff vectors i,j are in the same cluster
    true_labels_bool = (true_labels == true_labels[:, None])
    calculated_labels_bool = (calculated_labels == calculated_labels[:, None])

    # (i,j) is true iff vectors i,j are in the same cluster in both solutions
    intersection = np.logical_and(true_labels_bool, calculated_labels_bool)

    # the number of pairs in the same cluster in both solutions
    numerator = (intersection.sum() - n)

    # (i,j) is true iff vectors i,j are in the same cluster in at least one solution
    union = np.logical_or(true_labels_bool, calculated_labels_bool)

    # the number of pairs in the same cluster in at least one solution
    denominator = (union.sum() - n)

    # score calculation - the intersection divided by the union
    jaccard_score = numerator / denominator

    return jaccard_score


def out_clusters_pdf(data, d, sp_labels, km_labels, true_labels, k, K):
    """
    output a visualization of the result of the
    spectral clustering and k-means clustering
    algorithms, and the jaccard score for each.

    this function sometimes uses separate commands
    for 2d and 3d data.

    :param data: Data of (n*d) generaterd from sklearn
    :param d: dimension of data (2 or 3)
    :param sp_labels: lables based on spectral clustering
    :param km_labels: lables based on k-means clustering
    :param true_labels: Original lables based on make-blobs
    :param k: if Random is false number of Clusters based on make-blobs
    :param K: number of Clusters by user
    :return: NULL
    """
    # figure setup
    fig = plt.figure(figsize=plt.figaspect(0.5))


    # prepare the plot data
    x = data[:, 0]
    y = data[:, 1]
    if d == 3:
        z = data[:, 2]

    # create a subplot: spectral clustering
    if d == 2:
        plt.subplot(221)
        plt.scatter(x, y, c=sp_labels)
    else:
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.scatter3D(x, y, z, c=sp_labels)
    plt.title('Normalized Spectral Clustering')

    # create a subplot: kmeans clustering
    if d == 2:
        plt.subplot(222)
        plt.scatter(x, y, c=km_labels)
    else:
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.scatter3D(x, y, z, c=km_labels)

    plt.title('K-means')

    # prepare the plot text
    # jaccard scores for each algorithm's result
    sp_jaccard = jaccard_measure(sp_labels, true_labels)
    sp_jaccard_str = np.format_float_positional(sp_jaccard, precision=2)
    km_jaccard = jaccard_measure(km_labels, true_labels)
    km_jaccard_str = np.format_float_positional(km_jaccard, precision=2)
    # program parameters
    n = str(data.shape[0])
    K = str(K)
    k = str(k)

    # text to be displayed
    text = ("Data was generated from the values:\n" + "n = " + n + " , k = " +
            K + "\n" + "The k that was used for both algorithms was " +
            k + "\n" + "The jaccard measure for Spectral Clustering: " +
            sp_jaccard_str + "\n" + "The jaccard measure for K-means: " +
            km_jaccard_str)

    # add the text in place of a subplot
    fig.add_subplot(2, 2, (3, 4), frameon=False)
    plt.text(0.5, 0.5, text, ha='center', wrap=False)
    plt.axis('off')

    # save to pdf
    plt.savefig("clusters.pdf")
