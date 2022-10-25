# Handling arrays
from re import A
import numpy as np

# Functions for learning in Pytorch
import torch

# General initialization functions
from sklearn.decomposition import PCA as skPCA
from sklearn.metrics import pairwise_distances

# Functions to initialize t-SNE
from sklearn.manifold._t_sne import _joint_probabilities
from scipy.spatial.distance import squareform

# Functions to initialize UMAP
from umap.umap_ import find_ab_params
from umap.umap_ import fuzzy_simplicial_set


MACHINE_EPSILON = np.finfo(np.float).eps


def tsne_init(X, n_components, initial_components, perplexity, random_state=None):
    """
    Initialize t-SNE embedding with PCA.

    Parameters
    ----------
        X - high-dimensional data matrix
        n_components - required dimensionality of the embedding
        initial_components - dimensionality of the PCA projection space in which the neighbor probabilities are computed
        perplexity - guess about the number of close neighbors each point has
        random_state - used to set random seed for reproducibility
    """
    if initial_components < X.shape[1]:
        X = skPCA(n_components=initial_components,
                  random_state=random_state).fit_transform(X)
        init = torch.tensor(X[:, range(n_components)]).type(torch.float)
    else:
        init = skPCA(n_components=n_components,
                     random_state=random_state).fit_transform(X)
        init = torch.tensor(init).type(torch.float)
    P = _joint_probabilities(distances=pairwise_distances(
        X, squared=True), desired_perplexity=perplexity, verbose=0)
    P = torch.max(torch.tensor(squareform(P)).type(
        torch.float), torch.tensor([MACHINE_EPSILON]))

    return P, init


def umap_init(X, n_components, initial_components, n_neighbors, spread, min_dist, random_state=None):
    """
    Initialize UMAP embedding with PCA.

    Parameters
    ----------
        X - high-dimensional data matrix
        n_components - required dimensionality of the embedding
        initial_components - dimensionality of the PCA projection space in which the neighbor probabilities are computed
        n_neighbors - desired number of nearest neighbors
        spread - hyperparameter to control inter-cluster distance
        min_dist - hyperparameter to control cluster size
        random_state - used to set random seed for reproducibility
    """
    if initial_components < X.shape[1]:
        X = skPCA(n_components=initial_components,
                  random_state=random_state).fit_transform(X)
        init = torch.tensor(X[:, range(n_components)]).type(torch.float)
    else:
        init = skPCA(n_components=n_components,
                     random_state=random_state).fit_transform(X)
        init = torch.tensor(init).type(torch.float)
        
    dist = pairwise_distances(X, metric="euclidean")

    a, b = find_ab_params(spread, min_dist)
    P = fuzzy_simplicial_set(
        dist, n_neighbors, random_state=random_state, metric="precomputed")[0].tocoo()
    P = torch.sparse.FloatTensor(torch.LongTensor(np.vstack(
        (P.row, P.col))), torch.FloatTensor(P.data), torch.Size(P.shape)).to_dense()
    P = torch.max(P, torch.tensor([MACHINE_EPSILON]))

    return P, init, a, b


def random_projection(d):
    """Initialize embedding with a random 2-dimensional projection.

    1) Sampling a d-dimensional vector 'a' on the unit sphere
    2) Computing an orthonormal basis including 'a'
    3) Sampling d-1 dimensional coefficients 'c' on the unit sphere
    4) Obtain 'b' by combining the basis vectors (except 'a') using
    the coefficients 'c'


    Args:
        d (int): data dimension and size of projection vectors
    """
    a = np.random.multivariate_normal(mean=np.zeros((d,)), cov=np.eye(d))
    a = a / np.linalg.norm(a)

    A_random = np.random.normal(0, 1, size=(d, d))
    A_random = A_random/np.linalg.norm(A_random, ord=2, axis=1, keepdims=True)
    A_random[0, :] = a
    A_orthonormal = gram_schmidt(A_random)
    assert np.allclose(
        A_orthonormal[0, :], a), f"First basis vector is changed after applying Gram-Schmidt. {A_orthonormal} vs {a}"

    c = np.random.normal(0, 1, size=d-1)
    c = c / np.linalg.norm(c)

    b = np.sum(c[:, np.newaxis]*A_orthonormal[1:, :], axis=0)
    proj = np.stack((a, b), axis=1)
    return proj


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum(np.dot(v, b)*b for b in basis)
        if (w > 1e-10).any():
            basis.append(w/np.linalg.norm(w))
        else:
            raise ValueError(
                "The input vectors to Gram-Schmidt are linearly dependent.")
    return np.array(basis)
