import numpy as np
import torch
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient
from pymanopt.optimizers.line_search import BackTrackingLineSearcher, AdaptiveLineSearcher
import numpy as np
import pandas as pd

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as skPCA

from Code.losses import get_topological_loss, zero_loss
import Data.datasets as datasets
import Code.visualization as viz
from Code.manopt_searcher import NonAdaptiveSearcher, UnnormalizedSearcher

try:
    import wandb
    if wandb.__file__ is not None:
        usewandb = True
    else:
        usewandb = False
except ImportError as err:
    usewandb = False
torch.set_default_dtype(torch.float32)
MSE = torch.nn.MSELoss()


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class PymanoptPCA():
    def __init__(self,
                 topo_loss,
                 topo_weight,
                 random_state,
                 dim=2,
                 emb_loss=True,
                 verbosity=1):
        self.topo_loss_fn = topo_loss
        self.topo_weight = topo_weight
        if emb_loss:
            self.emb_loss_fn = self.pca_loss
        else:
            self.emb_loss_fn = zero_loss
        self.X = None
        self.dim = dim
        self.random_state = random_state
        set_random_seed(random_state)
        self.span_matrix = None

        # Store intermediate cost
        self.cost_df = pd.DataFrame(columns=["epoch", "emb_loss", "topo_loss",
                                             "weighted_topo_loss", "total_loss",
                                             "weighted_total_loss"])

        # cost function is called twice each iteration
        self.gradient_step = False
        self.verbosity = verbosity

    def initialize(self, X, emb_init=None):
        self.X = X - X.mean(axis=0)
        self.X = torch.tensor(self.X).type(torch.float)
        
        if emb_init is None:
            pca = skPCA(n_components=self.dim,
                        random_state=self.random_state).fit(self.X)
            self.span_matrix = pca.components_.transpose()
        else:
            self.span_matrix = emb_init

        initial_topo_loss = self.topo_loss_fn(
            torch.matmul(self.X, torch.tensor(self.span_matrix).type(
                torch.float)), self.X
        )
        initial_pca_loss = self.emb_loss_fn(self.get_reconstruction(
            torch.tensor(self.span_matrix).type(torch.float)))

        print(
            f"\nInit PCA loss {initial_pca_loss:.4f}, topo loss {initial_topo_loss:.4f}")

        init_loss_components = {
            'emb_loss': initial_pca_loss,
            'topo_loss': initial_topo_loss,
            'weighted_topo_loss': self.topo_weight * initial_topo_loss,
            'total_loss': initial_pca_loss + initial_topo_loss,
            'weighted_total_loss': initial_pca_loss + self.topo_weight * initial_topo_loss,
            'epoch': 0
        }
        self.cost_df.loc[len(self.cost_df)] = init_loss_components
        if usewandb:
            wandb.log(init_loss_components)

        return X

    def get_span_matrix(self):
        return self.span_matrix

    def get_reconstruction(self, span_matrix):
        projector = torch.matmul(
            span_matrix, torch.transpose(span_matrix, 1, 0))
        embedding = torch.matmul(self.X, projector)
        return embedding

    def pca_loss(self, reconstruction):
        loss = MSE(self.X, reconstruction)
        return loss

    def create_cost_function(self, manifold, smooth_len1, smooth_len2, loss_tol):
        @pymanopt.function.pytorch(manifold)
        def cost(w):
            w = w.type(torch.float)
            topo_loss = self.topo_loss_fn(torch.matmul(self.X, w), self.X)
            emb_loss = self.emb_loss_fn(self.get_reconstruction(w))
            loss = emb_loss + self.topo_weight * topo_loss

            if self.gradient_step:
                self.gradient_step = False
                self.cost_df.loc[len(self.cost_df)] = {
                    'emb_loss': emb_loss.item(),
                    'topo_loss': topo_loss.item(),
                    'weighted_topo_loss': self.topo_weight * topo_loss.item(),
                    'total_loss': emb_loss.item() + topo_loss.item(),
                    'weighted_total_loss': loss.item(),
                    'epoch': len(self.cost_df)
                }
                if self.verbosity > 1:
                    print((f"step {len(self.cost_df):4d}, emb loss {emb_loss.item():.4f}, " +
                           f"topo loss {topo_loss.item():.4f}, weighted_total_loss {loss.item():.4f}"))

                if len(self.cost_df) > smooth_len1:
                    obj_smooth = abs(np.mean(self.cost_df.topo_loss[-smooth_len1:]) /
                                     np.mean(self.cost_df.topo_loss[-smooth_len2:]) - 1)
                    if obj_smooth < loss_tol:
                        print(
                            f'Stopping optimization because (obj_smooth) {obj_smooth:.5f} < {loss_tol:.5f} (ftol)')
                        # return loss with gradient below min_grad
                        return 1e-15*torch.matmul(w[:, 0], w[:, 1])
            else:
                self.gradient_step = True

            return loss
        return cost

    def train(self,
              max_iterations=100,
              min_step_size=1e-6,
              line_search="custom",
              sufficient_decrease=0.1,
              contraction_factor=0.5,
              max_searcher_iterations=25,
              initial_step_size=1,
              smooth_len1=100,
              smooth_len2=50,
              loss_tol=1e-4,
              verbosity=1):

        manifold = Stiefel(self.X.size()[1], self.dim)
        problem = pymanopt.Problem(
            manifold,
            self.create_cost_function(manifold,
                                      smooth_len1,
                                      smooth_len2,
                                      loss_tol),
            euclidean_gradient=None,
            euclidean_hessian=None,
        )

        if line_search == "adaptive":
            print("Running adaptive line search.")
            line_searcher = AdaptiveLineSearcher(contraction_factor=contraction_factor,
                                                 sufficient_decrease=sufficient_decrease,
                                                 max_iterations=max_searcher_iterations,
                                                 initial_step_size=initial_step_size)
        elif line_search == "backtracking":
            print("Running backtracking line search.")
            # Backtracking line search reduces the step size until the approximated
            # gradient direction (slope will be decreased such that the line lies above f)
            # leads to a point above the function f
            line_searcher = BackTrackingLineSearcher(contraction_factor=contraction_factor,
                                                     optimism=2,
                                                     sufficient_decrease=sufficient_decrease,
                                                     max_iterations=max_searcher_iterations,
                                                     initial_step_size=initial_step_size)
        elif line_search == "custom":
            print("Running custom non adaptive update.")
            line_searcher = NonAdaptiveSearcher(
                initial_step_size=initial_step_size)
        elif line_search == "UnnormalizedSearcher":
            print(
                f"Running optimization with constant step size {initial_step_size}")
            line_searcher = UnnormalizedSearcher(
                initial_step_size=initial_step_size)
        else:
            raise ValueError(f"line_search must be one of['adaptive', 'backtrack', 'custom', 'UnnormalizedSearcher']")

        optimizer = ConjugateGradient(
            beta_rule="HestenesStiefel",
            line_searcher=line_searcher,
            min_gradient_norm=1e-06,
            max_iterations=max_iterations,
            min_step_size=min_step_size,
            verbosity=1,
            log_verbosity=1
        )

        optimizer_result = optimizer.run(problem,
                                         initial_point=self.span_matrix)
        self.span_matrix = optimizer_result.point
        embedding = (self.X @ self.span_matrix).type(torch.float)

        # Log intermediate losses
        log = optimizer_result.log["iterations"]
        log = pd.DataFrame.from_dict(log)
        log.rename(columns={'iteration': 'epoch'}, inplace=True)
        self.cost_df = pd.merge(self.cost_df, log[["epoch", "gradient_norm"]],
                                on="epoch")
        if usewandb:
            for row in self.cost_df.itertuples():
                wandb.log({
                    'emb_loss': row.emb_loss,
                    'topo_loss': row.topo_loss,
                    'weighted_topo_loss': row.weighted_topo_loss,
                    'total_loss': row.total_loss,
                    'weighted_total_loss': row.weighted_total_loss,
                    'epoch': row.epoch,
                    'gradient_norm': row.gradient_norm})
        if usewandb:
            wandb.log({'embedding_time': optimizer_result.time})
        return embedding.detach().numpy(), self.cost_df


if __name__ == "__main__":
    topo_loss_fn = get_topological_loss("one_circle", sampling=True,
                                        sampling_frac=0.4,
                                        sampling_rep=5)

    data, labels = datasets.get_circle_dataset(n=50, ndim=500,
                                               variance=0.0675, seed=42,
                                               noise="uniform")

    pyman = PymanoptPCA(topo_loss=topo_loss_fn,
                        random_state=42,
                        topo_weight=0.1,
                        emb_loss=True,
                        )
    pyman.initialize(data)
    emb, _ = pyman.train(
                        max_iterations=500,
                        line_search="custom",
                        initial_step_size=0.01)

    p1 = viz.plot_paper(emb, colors=labels)
    plt.show()