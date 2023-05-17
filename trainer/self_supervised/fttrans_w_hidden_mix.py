import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .base import BaseSSLTrainer

logger = logging.getLogger(__name__)


class FTTransHiddenMixSSLTrainer(BaseSSLTrainer):
    def __init__(
        self,
        alpha: float = 0.5,
        label_mix=None,
        **kwargs,
    ) -> None:
        """
        Args:
            alpha (float): the parameter used in the hidden-mix augmentation
            label_mix (bool): whether to apply label mixing
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    def get_lambda(self) -> float:
        """Return lambda"""
        if self.alpha > 0.0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        torch.tensor([lam]).float().to(self.device)
        return lam

    def forward(self, cont: Optional[Tensor] = None, cate: Optional[Tensor] = None) -> Tuple[None, float]:
        """
        Args:
            cont (tensor): the input of shape (b, num_cont) representing the continuous values
            cate (tensor): the input of shape (b, num_cate) representing the categorical values

        Returns:
            loss (float): the model's contrastive loss
        """
        z_0 = self.model(cont, cate)
        lam = self.get_lambda()
        lam = max(lam, 1 - lam)
        z_1, _ = self.model(cont, cate, lam=lam)
        loss = self.forward_loss(z_0, z_1)
        return None, loss
