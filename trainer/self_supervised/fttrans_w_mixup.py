import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .base import BaseSSLTrainer

logger = logging.getLogger(__name__)


class FTTransMixupSSLTrainer(BaseSSLTrainer):
    def __init__(
        self,
        alpha: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Args:
            alpha (float): the parameter used in the mixup augmentation
        """
        super().__init__(**kwargs)
        logger.info(f"Set mixup alpha to {alpha}.")
        self.alpha = alpha

    def get_lambda(self) -> float:
        """Return lambda"""
        if self.alpha > 0.0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 0.0
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
        lam = min(lam, 1 - lam)
        z_1 = self.model.forward_no_labelmix(cont, cate, alpha=lam)
        loss = self.forward_loss(z_0, z_1)
        return None, loss
