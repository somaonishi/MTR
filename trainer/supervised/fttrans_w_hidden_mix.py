import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .base_da import BaseDATrainer

logger = logging.getLogger(__name__)


class FTTransWithHiddenMixTraniner(BaseDATrainer):
    def __init__(self, alpha: float = 0.5, label_mix: bool = True, **kwargs) -> None:
        """
        Args:
            alpha (float): the parameter used in the hidden-mix augmentation
            label_mix (bool): whether to apply label mixing
        """
        super().__init__(**kwargs)
        logger.info(f"Set hidden mix alpha to {alpha}.")
        self.alpha = alpha
        self.label_mix = label_mix

    def get_lambda(self) -> float:
        """Return lambda"""
        if self.alpha > 0.0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
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
        if self.label_mix:
            out, target = self.model(cont, cate, target, lam)
        else:
            lam = max(lam, 1 - lam)
            out, _ = self.model(cont, cate, target, lam)
            if self.datamodule.d_out > 1:
                target = torch.argmax(target, dim=1)
        return out, target
