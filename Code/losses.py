# Handling arrays
import numpy as np

# Functions for learning in Pytorch
import torch

# MSE loss for PCA
from torch.nn import MSELoss

# Function to preprocess persistence diagrams
from topologylayer.util.process import remove_zero_bars
from topologylayer.nn import AlphaLayer

# Working with graphs in Python
import networkx as nx

# For sampling loss
import random

MACHINE_EPSILON = np.finfo(np.float).eps
MSE = MSELoss()


def pca_loss(X, W, Y=None):
    """
    Pytorch compatible implementation of the ordinary PCA loss.

    Parameters
    ----------
        X - original high-dimensional data
        W - (approximately) linear projection matrix
        Y - projection of X matrix onto W (computed if missing)
    """
    # Projection of X onto subspace W
    if Y is None:
        Y = torch.matmul(X, W)
    # Low-rank reconstruction of X
    L = torch.matmul(Y, torch.transpose(W, 0, 1))

    # Compute loss
    loss = MSE(L, X)

    return loss


def tsne_loss(P, Y):
    """
    Pytorch compatible implementation of the PCA reconstruction loss.

    Parameters
    ----------
        P - high-dimensional neighbor probabilities
        Y - current embedding
    """
    # Compute pairwise affinities
    sum_Y = torch.sum(Y * Y, 1)
    num = -2 * torch.mm(Y, Y.t())
    num = 1 / (1 + torch.add(torch.add(num, sum_Y).t(), sum_Y))
    num[range(Y.shape[0]), range(Y.shape[0])] = 0
    Q = num / torch.sum(num)
    Q = torch.max(Q, torch.tensor([MACHINE_EPSILON]))

    # Compute loss
    loss = torch.sum(P * torch.log(P / Q))
    return loss


def umap_loss(P, Y, a, b):
    """
    Pytorch compatible implementation of the ordinary UMAP loss.

    Parameters
    ----------
        P - high-dimensional neighbor probabilities
        Y - current embedding
        a - UMAP hyperparameter used in pairwise affinities computation
        b - UMAP hyperparameter used in pairwise affinities computation
    """
    # Compute pairwise affinities
    Q = 1 / (1 + a * torch.cdist(Y, Y)**(2 * b))
    Q = torch.max(Q, torch.tensor([MACHINE_EPSILON]))
    oneminQ = torch.max(1 - Q, torch.tensor([MACHINE_EPSILON]))

    # Compute loss
    loss = torch.sum(- P * torch.log(Q) - (1 - P) * torch.log(oneminQ))

    return loss


def zero_loss(*args):
    """
    Pytorch compatible implementation zero loss function.
    """
    return torch.tensor(0, dtype=torch.float)


def RandomWalk(G, node, t):
    """
    Original source: https://github.com/dsgiitr/graph_nets

    Parameters
    ----------

    """
    walk = [node]  # Walk starts from this node
    for i in range(t - 1):
        if not nx.is_weighted(G):
            W = np.ones(len(G[node]))
        else:
            W = [G[node][n]["weight"] for n in G.neighbors(node)]
        node = np.random.choice(list(G.neighbors(node)), p=W / np.sum(W))
        walk.append(node)

    return walk


def deepwalk_loss(model, G, w, t):
    """
    Pytorch compatible implementation of the deepwalk loss.
    Original source: https://github.com/dsgiitr/graph_nets

    Parameters
    ----------

    """
    loss = torch.tensor(0).type(torch.float)
    for vi in list(G.nodes()):
        wvi = RandomWalk(G, vi, t)
        for j in range(len(wvi)):
            for k in range(max(0, j - w), min(j + w, len(wvi))):
                prob = model(wvi[j], wvi[k])
                loss = loss - torch.log(prob)

    return loss


class DiagramLoss(torch.nn.Module):
    """
    Applies function g over points in a diagram sorted by persistence.
    Original source: https://github.com/bruel-gabrielsson/TopologyLayer

    Parameters
    ----------
        dim - homology dimension to work over
        g - pytorch compatible function to evaluate on each diagram point
        i - start of summation over ordered diagram points
        j - end of summation over ordered diagram points
        remove_zero = Flag to remove zero-length bars (default=True)
    """

    def __init__(self, dim, g, i=1, j=np.inf, remove_zero=True):
        super(DiagramLoss, self).__init__()
        self.dim = dim
        self.g = g
        self.i = i - 1
        self.j = j
        self.remove_zero = remove_zero

    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        dgm = dgms[self.dim]
        if self.remove_zero:
            dgm = remove_zero_bars(dgm)

        lengths = dgm[:, 1] - dgm[:, 0]
        indl = torch.argsort(lengths, descending=True)
        dgm = dgm[indl[self.i:min(dgm.shape[0], self.j)]]
        if len(dgm) == 0:
            loss = torch.tensor(0, dtype=torch.float, requires_grad=False)
        else:
            loss = torch.sum(torch.stack([self.g(dgm[i])
                                          for i in range(dgm.shape[0])], dim=0))
        return loss


class TopologicalLoss(torch.nn.Module):
    def __init__(self,
                 dim=0,
                 i=1,
                 j=np.inf,
                 weight=1,
                 exponent=1,
                 decay=0,
                 sampling=False,
                 sampling_fn=None,
                 sampling_frac=0.2,
                 sampling_rep=5):
        super().__init__()
        self.dim = dim
        # weight is negative if we want to maximize
        self.weight = weight
        self.i = i
        self.j = j
        self.exponent = exponent
        self.decay = decay
        self.sampling = sampling
        self.sampling_fn = sampling_fn
        self.sampling_frac = sampling_frac
        self.sampling_rep = sampling_rep

        # function to compute topological loss
        def g(point):
            return (point[1] - point[0])**self.exponent * (0.5*(point[0] + point[1]))**self.decay

        self.dgmloss = DiagramLoss(self.dim, g,
                                   i=self.i,
                                   j=self.j,
                                   remove_zero=True)

    def forward(self, Y, *args):
        # Zero loss
        if self.weight == 0:
            loss = torch.tensor(0, dtype=torch.float)
        # standard loss without sampling
        elif not self.sampling:
            dgminfo = AlphaLayer(maxdim=self.dim)(Y)
            loss = self.weight * self.dgmloss(dgminfo)
        else:
            loss = torch.tensor(0, dtype=torch.float)
            rep = 0
            for _ in range(self.sampling_rep):
                if self.sampling_fn is not None:
                    sample = self.sampling_fn(Y)
                else:
                    sample = random.sample(
                        range(Y.shape[0]),
                        int(Y.shape[0] * self.sampling_frac))
                dgminfo = AlphaLayer(maxdim=self.dim)(Y[sample, :])
                rep_loss = self.dgmloss(dgminfo)

                if rep_loss != 0.0:
                    loss = loss + rep_loss
                    rep += 1

            if rep > 0:
                loss = self.weight * (loss / rep)
            else:
                print(
                    f"Warning: No d-dimensional non-zero feature. Ignoring topological loss.")
        return loss


def flare_sampling(Y, thres=0.25):
    f = torch.norm(Y - torch.mean(Y, dim=0), dim=1)
    f /= torch.max(f)
    sample = f > thres
    return sample


def get_topological_loss(loss, **kwargs):
    loss_args = {
        "one_circle": {"dim": 1,
                       "i": 1,
                       "j": 1,
                       "weight": -1,
                       "sampling": True,
                       },
        "two_circles": {"dim": 1,
                        "i": 1,
                        "j": 2,
                        "weight": -1,
                        "sampling": True,
                        },
        "two_clusters": {"dim": 0,
                         "i": 2,
                         "j": 2,
                         "weight": -1,
                         "sampling": False
                         },
        "connected_component": {"dim": 0,
                                "i": 2,
                                "j": np.inf,
                                "weight": 1,
                                "sampling": False
                                },
        "flare": {"dim": 0,
                  "i": 3,
                  "j": 3,
                  "weight": -1,
                  "sampling": True,
                  "sampling_fn": flare_sampling,
                  "sampling_rep": 1},
        "zero_loss": {"dim": 0,
                      "i": 1,
                      "j": 1,
                      "weight": 0,
                      "sampling": False
                      },
    }

    if loss not in loss_args.keys():
        args = {}
        # check if all parameters are specified
        for p in ["dim", "i", "j", "weight", "sampling"]:
            if p not in kwargs:
                raise ValueError(
                    f"Need to specify parameter value for {p} when using custom loss.")
    else:
        args = loss_args[loss]

    args.update(kwargs)
    return TopologicalLoss(**args)


def combine_topological_losses(loss_functions):
    if len(loss_functions) == 1:
        return loss_functions[0]
    else:
        def combined_loss(Y, *args):
            loss = torch.tensor(0, dtype=torch.float)
            for fn in loss_functions:
                loss = loss + fn(Y)
            return loss
        return combined_loss
