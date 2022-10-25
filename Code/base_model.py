import abc
from typing import Dict, Tuple

import torch.nn as nn


class EmbeddingModel(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for embedding models."""

    @abc.abstractmethod
    def initialize(self, x, emb_init):
        """Initialize embedding with e.g. PCA

        Args:
            x: Tensor with data
            emb_init: potential initialization
        Returns:
            Tensor with possibly normalized data
        """

    @abc.abstractmethod
    def forward(self, x) -> Tuple[float, Dict[str, float]]:
        """Compute loss for model.
        Args:
            x: Tensor with data
        Returns:
            Tuple[loss, dict(loss_component_name -> loss_component)]
        """

    @abc.abstractmethod
    def encode(self, x):
        """Compute latent representation."""

    @abc.abstractmethod
    def decode(self, z):
        """Compute reconstruction."""
        
    def on_optimizer_step(self): 
        """Called after every optimization step to 
        potentially normalize or orthogonalize weights.
        """
        pass

    def on_train_end(self):
        """Possiblity to store final weights."""
        pass
    
    @abc.abstractmethod
    def get_span_matrix(self):
        """Return the projection vectors."""
