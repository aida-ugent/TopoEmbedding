import numpy as np
import time
import random

# Functions for learning in Pytorch
import torch
import pytorch_lightning as pl
from torch import sigmoid
from torch.nn import Embedding
from torch.nn.functional import binary_cross_entropy
from sklearn.metrics import roc_auc_score

# Embedding and topological loss functions
from Code.losses import umap_loss, tsne_loss, zero_loss, deepwalk_loss
from Code.base_model import EmbeddingModel

# Functions to initialize embeddings
from sklearn.decomposition import PCA as skPCA
from Code.embedding_init import tsne_init, umap_init

# Helper functions for network embedding
from Code.splitter import compute_tr_val_split, construct_adj_matrix
from Code.dataloader import config_network_loader, config_edge_loader
from Code.pymanopt_pca import PymanoptPCA


def str_to_method(method_name):
    if method_name == "PCA":
        from Debugging import TopologicallyRegularizedPCA
        return TopologicallyRegularizedPCA

    if method_name == "OrthoProj":
        from Debugging import TopologicallyRegularizedOrthogonalProjection
        return TopologicallyRegularizedOrthogonalProjection

    if method_name == "ManoptPCA":
        return PymanoptPCA

    if method_name == "TSNE":
        return TopologicallyRegularizedTSNE

    if method_name == "UMAP":
        return TopologicallyRegularizedUMAP

    if method_name == "TOPO":
        return TopologicallyRegularizedEmbedding

    else:
        raise ValueError(f"Method {method_name} not supported. " +
                         "Supported methods are PCA, TSNE, UMAP, TOPO, and OrthoProj, ManoptPCA.")


class TopologicallyRegularizedEmbedding(EmbeddingModel):
    def __init__(self, topo_loss, dim=2,
                 random_state=42, **kwargs):
        super().__init__()
        self.topo_loss = topo_loss
        self.dim = dim
        self.W = None
        self.random_state = random_state

    def initialize(self, X, emb_init=None):
        # Center the data
        X = X - X.mean(axis=0)
        self.X = torch.tensor(X).type(torch.float)

        # Initialize embedding with PCA or dataset itself
        if emb_init is None:
            pca = skPCA(n_components=self.dim,
                        random_state=self.random_state).fit(self.X)
            init_y = torch.tensor(
                pca.components_.transpose(), dtype=torch.float)
            init_y = np.matmul(self.X, init_y)
            #self.Y = torch.tensor(self.Y).type(torch.float)
        else:
            assert emb_init.shape[1] == 2, "Initial embedding must be two-dimensional"
            init_y = torch.tensor(emb_init)

        self.Y = torch.nn.parameter.Parameter(init_y, requires_grad=True)

        return X

    def forward(self, X):
        loss = self.topo_loss(self.Y, X)
        return (loss, {
            'total_loss': loss.item(),
            'emb_loss': 0.0,
            'topo_loss': loss.item(),
            'weighted_total_loss': loss.item()})

    def encode(self, X):
        return self.Y

    def decode(self, Y):
        raise NotImplementedError

    def get_span_matrix(self):
        return None


class TopologicallyRegularizedTSNE(EmbeddingModel):
    """Topologically regularized T-SNE."""

    def __init__(self, topo_loss, topo_weight, dim=2,
                 perplexity=30, initial_components=30,
                 random_state=None, emb_loss=True):
        super().__init__()
        self.dim = dim
        self.random_state = random_state
        self.topo_loss = topo_loss
        self.topo_weight = topo_weight
        self.perplexity = perplexity
        self.initial_components = initial_components
        self.emb_loss = emb_loss

    def initialize(self, X, emb_init=None):
        # Change X to #initial_components of PCA and return the first #dim as initialization
        self.P, init_Y = tsne_init(X, n_components=self.dim,
                                   initial_components=self.initial_components,
                                   perplexity=self.perplexity,
                                   random_state=self.random_state)
        if emb_init is not None:
            assert emb_init.shape == (X.shape[0], self.dim)
            emb_init = torch.tensor(emb_init, dtype=torch.float) if not torch.is_tensor(
                emb_init) else emb_init
            self.Y = torch.nn.parameter.Parameter(emb_init, requires_grad=True)
        else:
            self.Y = torch.nn.parameter.Parameter(init_Y, requires_grad=True)

        return X

    def forward(self, X):
        # Recenter embedding
        self.Y - torch.mean(self.Y, 0)

        loss_emb = tsne_loss(self.P, self.Y) if self.emb_loss else zero_loss()
        loss_topo = self.topo_loss(self.Y, X)
        loss = loss_emb + self.topo_weight * loss_topo

        loss_components = {
            'emb_loss': loss_emb.item(),
            'topo_loss': loss_topo.item(),
            'weighted_topo_loss': self.topo_weight * loss_topo.item(),
            'total_loss': loss_emb.item() + loss_topo.item(),
            'weighted_total_loss': loss.item()
        }
        return (loss, loss_components)

    def encode(self, X):
        return self.Y

    def decode(self, Y):
        raise NotImplementedError()

    def on_train_end(self):
        if not self.emb_loss:
            emb_loss = tsne_loss(self.P, self.Y)
            print(f"TSNE loss for final embedding is {emb_loss}")

    def get_span_matrix(self):
        return None


class TopologicallyRegularizedUMAP(EmbeddingModel):

    def __init__(self, topo_loss, topo_weight, dim=2, initial_components=30,
                 n_neighbors=15, spread=1.0, min_dist=0.1, random_state=None,
                 emb_loss=True, verbosity=1):
        """Conduct topologically regularized UMAP embedding.

        Parameters
        ----------
            topo_loss - topological loss function (= prior) for topological regularization
            dim - required dimensionality of the embedding
            initial_components - dimensionality of the PCA projection space in which the neighbor probabilities are computed
            n_neighbors - desired number of nearest neighbors
            spread - hyperparameter to control inter-cluster distance
            min_dist - hyperparameter to control cluster size
            random_state - used to set random seed for reproducibility
            emb_loss - whether to use the UMAP loss function, if false, the zero loss function is used insteads

        """
        super().__init__()
        self.topo_loss = topo_loss
        self.topo_weight = topo_weight
        self.dim = dim
        self.initial_components = initial_components
        self.n_neighbors = n_neighbors
        self.spread = spread
        self.min_dist = min_dist
        self.random_state = random_state
        self.emb_loss = emb_loss
        self.verbosity = verbosity

        # computed later in the embedding process
        self.P = None
        self.Y = None
        self.a = None
        self.b = None

    def initialize(self, X, emb_init=None):
        # Initialize embedding with PCA
        (self.P, Y_init,
         self.a, self.b) = umap_init(X, n_components=self.dim,
                                     initial_components=self.initial_components,
                                     n_neighbors=self.n_neighbors,
                                     spread=self.spread,
                                     min_dist=self.min_dist,
                                     random_state=self.random_state)  # numba code compilation now completed
        if emb_init is not None:
            assert emb_init.shape == (X.shape[0], self.dim)
            Y_init = torch.tensor(emb_init, dtype=torch.float) if not torch.is_tensor(
                emb_init) else emb_init
        self.Y = torch.nn.parameter.Parameter(Y_init, requires_grad=True)
        return X

    def forward(self, X):
        # Recenter embedding
        self.Y - torch.mean(self.Y, 0)

        loss_emb = umap_loss(self.P, self.Y, self.a,
                             self.b) if self.emb_loss else zero_loss()
        loss_topo = self.topo_loss(self.Y, X)
        loss = loss_emb + self.topo_weight * loss_topo

        loss_components = {
            'emb_loss': loss_emb.item(),
            'topo_loss': loss_topo.item(),
            'weighted_topo_loss': self.topo_weight * loss_topo.item(),
            'total_loss': loss_emb.item() + loss_topo.item(),
            'weighted_total_loss': loss.item()
        }
        return (loss, loss_components)

    def encode(self, X):
        return self.Y

    def decode(self, Y):
        raise NotImplementedError()

    def on_train_end(self):
        if not self.emb_loss:
            emb_loss = umap_loss(self.P, self.Y, self.a, self.b)
            print(f"UMAP loss for final embedding is {emb_loss}")

    def get_span_matrix(self):
        return None


class GraphInProdEmbeddingModel(pl.LightningModule):
    def __init__(self, init=None, **kwargs):
        super(GraphInProdEmbeddingModel, self).__init__()
        self.name = "GraphEmbedding"
        self.n = kwargs["n"]
        self.dim = kwargs["dim"]
        self.learning_rate = kwargs["learning_rate"]
        self.eps = kwargs["eps"]
        self.emb_loss = kwargs["emb_loss"]
        self.topo_loss = kwargs["topo_loss"]
        self.optimizer = getattr(torch.optim, "Adam")

        self.embedding = Embedding(self.n, self.dim)
        self.embedding.weight.data.normal_(0, 0.1)

        if init is not None:
            self.embedding.weight.data = torch.tensor(init)

        self.b_node = torch.nn.parameter.Parameter(torch.Tensor(self.n))
        torch.nn.init.normal_(self.b_node, std=0.1)

        self.b = torch.nn.parameter.Parameter(torch.Tensor(1))
        torch.nn.init.normal_(self.b, std=0.1)

    def forward(self, uids, iids):
        return sigmoid((self.embedding(uids) * self.embedding(iids)).sum(1) + self.b_node[uids] + self.b_node[iids] + self.b)

    def training_step(self, batch, batch_idx):
        uids, iids, target = batch
        pred = self(uids, iids)
        loss_emb = binary_cross_entropy(
            pred.view(-1, 1).float(), target.view(-1, 1).float()).sum() if self.emb_loss else zero_loss()
        loss_top = self.topo_loss(self.embedding)
        loss = loss_emb + loss_top

        # self.log("train_loss", loss, on_epoch=True)
        self.log("emb. loss", loss_emb, prog_bar=True)
        self.log("top. loss", loss_top, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, loader_idx):
        uids, iids, target = batch
        pred = self(uids, iids)
        auc = roc_auc_score(target, pred)
        self.log(f"val_auc_{loader_idx}", auc, prog_bar=True)

    def configure_optimizers(self):
        print(f"Config optimizer with learning rate {self.learning_rate}")
        optimizer = self.optimizer(
            self.parameters(), lr=self.learning_rate, eps=self.eps)
        return optimizer


def GraphInProdEmbed(G, dim=2, emb_loss=True, topo_loss=zero_loss,
                     train_frac=0.9, num_epochs=250, learning_rate=1e-1,
                     eps=1e-07, random_state=None, init=None):

    # Track total embedding time
    start_time = time.time()

    # Prepare the data for training
    if not random_state is None:
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    E = np.array(G.edges())
    A = construct_adj_matrix(E, E.max() + 1)
    tr_A, tr_E, val_E = compute_tr_val_split(A, train_frac)
    tr_A_loader = config_network_loader(tr_A)
    tr_E_loader = config_edge_loader(tr_E)
    val_E_loader = config_edge_loader(val_E)

    # Conduct the training
    trainer = pl.Trainer(num_sanity_val_steps=0,
                         checkpoint_callback=False, logger=False, max_epochs=num_epochs)
    model = GraphInProdEmbeddingModel(n=tr_A.shape[0], dim=dim,
                                      emb_loss=emb_loss,
                                      topo_loss=topo_loss,
                                      learning_rate=learning_rate,
                                      eps=eps,
                                      init=init)
    trainer.fit(model, tr_A_loader, val_dataloaders=[
                tr_E_loader, val_E_loader])

    # Print total embedding time
    elapsed_time = time.time() - start_time
    print("Time for embedding: " +
          time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # Return the embedded graph
    return model


def func_L(w):
    """
    Parameters
    ----------
    w: Leaf node.
    Returns
    -------
    count: The length of path from the root node to the given vertex.
    """
    count = 1
    while(w != 1):
        count += 1
        w //= 2

    return count


def func_n(w, j):
    """
    Original source: https://github.com/dsgiitr/graph_nets

    Parameters
    ----------


    Returns
    -------

    """
    li = [w]
    while(w != 1):
        w = w // 2
        li.append(w)

    li.reverse()

    return li[j]


class HierarchicalModel(torch.nn.Module):
    """
    Original source: https://github.com/dsgiitr/graph_nets

    Parameters
    ----------


    Returns
    -------

    """

    def __init__(self, size_vertex, dim, init=None):
        super(HierarchicalModel, self).__init__()
        self.size_vertex = size_vertex
        self.phi = torch.nn.Parameter(torch.rand(
            (size_vertex, dim)) if init is None else torch.tensor(init), requires_grad=True)
        self.prob_tensor = torch.nn.Parameter(
            torch.rand((2 * size_vertex, dim), requires_grad=True))

    def forward(self, wi, wo):
        one_hot = torch.zeros(self.size_vertex)
        one_hot[wi] = 1
        w = self.size_vertex + wo
        h = torch.matmul(one_hot, self.phi)
        p = torch.tensor([1.0])
        for j in range(1, func_L(w) - 1):
            mult = -1
            if(func_n(w, j + 1) == 2 * func_n(w, j)):  # Left child
                mult = 1

            p = p * \
                torch.sigmoid(
                    mult * torch.matmul(self.prob_tensor[func_n(w, j)], h))

        return p


def DeepWalk(G, dim=2, emb_loss=True, topo_loss=zero_loss, init=None,
             num_epochs=250, learning_rate=1e-2, w=3, t=6, random_state=None):
    """
    Original source: https://github.com/dsgiitr/graph_nets

    Parameters
    ----------


    Returns
    -------

    """
    # Track total embedding time
    start_time = time.time()

    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    model = HierarchicalModel(size_vertex=len(G.nodes()), dim=dim, init=init)

    for epoch in range(num_epochs):
        loss_emb = deepwalk_loss(model, G, w, t) if emb_loss else zero_loss()
        loss_top = topo_loss(model.phi)
        loss = loss_emb + loss_top
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                param.data.sub_(learning_rate * param.grad)
                param.grad.data.zero_()

        # Print losses according to epoch
        if epoch == 0 or (epoch + 1) % (int(num_epochs / 10)) == 0:
            print("[epoch %d] [emb. loss: %f, top. loss: %f, total loss: %f]" % (
                epoch + 1, loss_emb, loss_top, loss))

    # Print total embedding time
    elapsed_time = time.time() - start_time
    print("Time for embedding: " +
          time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    return model
