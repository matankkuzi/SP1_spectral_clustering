"""
Spetcral Clustering Functions:

This module contains the functions
that are the building blocks of
the Spectral Clustering algorithm.

"""

import numpy as np

EPSILON = 0.0001


def create_weighted_matrix(x):
    """
    Form the weighted adjacency matrix W from X
    :param x: data points
    :return: W
    """
    # create a matrix W where Wij = ||xi-xj|| (L2 norm)
    W = np.sqrt(np.sum((x[:, None, :] - x[None, :]) ** 2, -1))
    # apply the exp() function
    W = np.exp(-0.5 * W)
    # create the final matrix by inserting zeros in the diagonal
    np.fill_diagonal(W, 0)

    return W


def create_diagonal_matrix(W):
    """
    The diagonal degree matrix D ∈ R n×n is defined as D = (dij )i,j=1,...,n, such that:
    if i=j,
    We get that i-th element along the diagonal equals to the sum of the i-th row of W. In
    essence, D’s diagonal elements represent the sum of weights that lead to vertex vi.
    if i!=j
     dij=0
     This function calculates D and returns D^(-1/2) to be used to create W's laplacian.

    :param W: weighted adjacency matrix
    :return: D^(-1/2)  (D is W's diagonal degree matrix)
    """
    # This array is the diagonal of D
    D_diagonal = np.sum(W, axis=1)

    # This array is the diagonal of D^(-1/2)
    div = np.divide(1, np.sqrt(D_diagonal))

    # Handling zero division
    if np.any(div == 0):
        exit(0)

    D_neg_sqrt_diagonal = np.divide(1, np.sqrt(D_diagonal))

    # create D^(-1/2) from the diagonal
    D_neg_sqrt = np.diag(D_neg_sqrt_diagonal)

    return D_neg_sqrt


def create_laplacian_matrix(D_neg_sqrt, W):
    """
    The normalize graph Laplacian (n,n) os defined as (I- D^(0.5) @ W @ D^(0.5))
    :param D_neg_sqrt: diagonal degree matrix
    :param W: weighted adjacency matrix
    :return: Laplacian matrix
    """
    # get dimensions
    n = W.shape[0]

    # create an identity matrix
    Id = np.eye(n)

    # calculate laplacian
    L = Id - (D_neg_sqrt @ (W @ D_neg_sqrt))

    return L


def create_gramschmidt_matrix(L):
    """
    apply the gram-schmidt algorithm to get the QR decomposition of L
    this is an assisting function used by the QR iteration algorithm
    """
    # Matrix Initialization
    U = np.copy(L, order='F').astype(np.float32)
    R = np.zeros_like(U,  dtype=np.float32)
    Q = np.zeros_like(U,  dtype=np.float32)

    # get dimensions
    n = L.shape[0]

    # Modified Gram-Schmidt Algorithm
    for i in range(n):

        # First loop of the algorithm
        R[i][i] = np.linalg.norm(U[:, i], axis=0)
        Q[:, i] = np.divide(U[:, i], R[i][i])  # np.div print descriptive message if div by 0.

        # the second loop does not apply to the last vector
        if i == n-1:
            break

        # Second loop of the algorithm
        R[i, i+1:] = Q[:, i].T @ U[:, i+1:]
        U[:, i+1:] -= (R[i, i+1:][:, None] * Q[:, i]).T

    return Q, R


def qr_iterator(L):
    """
    the Q R iteration algorithm
    the returned Agag is an upper triangular matrix whose diagonal
    is comprised of L's eigenvalues.
    Qgag is a matrix whose columns are the eigenvalues of L
    ordered according to their respective eigenvalues
    in Agag's diagonal.

    :param L: Laplacian matrix
    :return: Agag and Qgag
    """

    # get dimensions
    n = L.shape[0]

    # Matrix Initialization
    Agag = L.copy().astype(np.float32)
    Qgag = np.identity(n, dtype=np.float32)

    # QR Iteration Algorithm
    for i in range(n):

        # get the QR decomposition of Agag
        Q, R = create_gramschmidt_matrix(Agag)

        Agag = R @ Q
        Qgag_matmul_Q = Qgag @ Q

        # an end condition
        if np.all(np.abs(np.abs(Qgag) - np.abs(Qgag_matmul_Q)) <= EPSILON):
            return Agag, Qgag

        else:
            Qgag = Qgag_matmul_Q

    return Agag, Qgag


def eigengap(Atag):
    """
    Use the eigengap heuristic to obtain k.
    by the first k eigenvectors according the smallest k eigenvalues.

    :param Atag: is an upper triangular matrix whose diagonal
    is comprised of L's eigenvalues.
    :return: k
    """
    eigenvalues = np.diagonal(Atag)

    # get ceil of n/2
    n_2 = int(np.ceil(Atag.shape[0] / 2))

    # get sorted eigen values from Atag diagonal
    eigenvalues_array = np.sort(eigenvalues)

    # calculate k using the eigengap heuristic
    k = np.argmax(np.diff(eigenvalues_array)[:n_2]) + 1

    return k


def get_spectral_matrix(Atag, Qtag, k):
    """
    Let U ∈ R n×k be the matrix containing the vectors u1, . . . , uk as columns
    corresponds to the smallest k eigenvalues

    :param Atag: matrix (n,n) whose diagonal elements approach the eigenvalues of L
    :param Qtag: orthogonal matrix (n,n) whose each col is j an eigenvector corresponds to eigen value A(j,j)
    :param k: number of Clusters
    :return: U
    """
    eigenvalues = np.diagonal(Atag)

    # sort the eigenvectors (Qtag's columns) with respect to the eigenvalues' sorting
    eigen_sorting = np.argsort(eigenvalues)[:k]
    U = Qtag[:, eigen_sorting]

    return U


def normalize_matrix(U):
    """
    Form the matrix T ∈ R n×k from U by renormalizing each of U’s rows to have unit length,
    that is set to  T(i,j) = U(i,j)/sum of col j (U(i,j)^2)

    :param U: normalize mateix
    :return:T
    """
    div = np.sqrt(np.sum(U*U, axis=1))

    # Handling zero division
    if np.any(div == 0):
        exit(0)

    T = np.divide(U, np.sqrt(np.sum(U*U, axis=1))[:, None])

    return T