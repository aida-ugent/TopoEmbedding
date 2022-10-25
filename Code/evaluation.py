import pandas as pd
import numpy as np

# Quantitative evaluation with sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import NearestNeighbors


def evaluate_embeddings(Ys, labels, model, scoring, params={}, stratify=None, ntimes=10, test_frac=0.1, random_state=None):
    # Obtain test performances over multiple train-test splits
    performances = pd.DataFrame(columns=Ys.keys())
    if random_state is not None:
        np.random.seed(random_state)
    for idx in range(ntimes):
        train, test = train_test_split(
            range(len(labels)), stratify=stratify, test_size=test_frac)

        # Obtain prediction performance per data embedding
        for key, Y in Ys.items():
            this_train = Y[tuple(np.meshgrid(train, train))] if hasattr(
                model, "metric") and model.metric == "precomputed" else Y[train, :]
            this_test = Y[tuple(np.meshgrid(train, test))] if hasattr(
                model, "metric") and model.metric == "precomputed" else Y[test, :]
            this_labels_train = np.array([labels[idx] for idx in train])
            this_labels_test = np.array([labels[idx] for idx in test])
            CV = GridSearchCV(model, params, scoring=scoring)
            CV.fit(this_train, this_labels_train)
            performances.loc["test" +
                             str(idx), key] = CV.score(this_test, this_labels_test)

    return performances


def getNeighbors(data, n_neigh=30):
    nbrs = NearestNeighbors(n_neighbors=n_neigh).fit(data)
    _, indices = nbrs.kneighbors(data)
    return indices


def average_jaccard_distance(data, embedding, n_neigh=30):
    # compute pairwise distances in the ambient space and embedding
    high_nbr = getNeighbors(data, n_neigh=n_neigh)
    low_nbr = getNeighbors(embedding, n_neigh=n_neigh)

    jacc = getJaccard(high_nbr, low_nbr)
    return np.mean(jacc)


def getJaccard(orig, new):
    """Compute Jaccard distance of neighbor indices for every observation.

    Args:
        nb1 : ndarray(dtype=float, ndim=2)
            Array containing the neighbor indices from space 1.  
        nb2 : ndarray(dtype=float, ndim=2)
            Array containing the neighbor indices from space 2.

    Returns:
        ndarray(dtype=float, ndim=1):
            Array with Jaccard distance for every observation.
    """
    frac = [0]*new.shape[0]
    for i in range(new.shape[0]):
        inter = set(orig[i, :]).intersection(new[i, :])
        frac[i] = 1 - len(inter)/len(set(orig[i, :]).union(new[i, :]))
    return frac
