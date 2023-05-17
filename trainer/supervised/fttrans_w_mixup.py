import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .base_da import BaseDATrainer

logger = logging.getLogger(__name__)


class FTTransWithMixupTraniner(BaseDATrainer):
    def __init__(self, alpha: float = 0.1, **kwargs) -> None:
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

    def forward_w_da(
        self,
        cont: Optional[Tensor] = None,
        cate: Optional[Tensor] = None,
        target: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            cont (tensor): the input of shape (b, num_cont) representing the continuous values
            cate (tensor): the input of shape (b, num_cate) representing the categorical values
            target (tensor): the target labels of shape (b, num_classes)

        Returns:
            out (tensor): the output of shape (b, num_classes) representing the model's prediction with data augmentation
            target (tensor): the target labels of shape (b, num_classes)
        """
        lam = self.get_lambda()
        if self.datamodule.d_out > 1:
            target = F.one_hot(target, self.datamodule.d_out)
        out, target = self.model(cont, cate, target, lam)
        return out, target
