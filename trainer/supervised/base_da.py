import logging
from typing import Optional, Tuple

import numpy as np
from torch import Tensor

from .base import BaseSupervisedTrainer

logger = logging.getLogger(__name__)


class BaseDATrainer(BaseSupervisedTrainer):
    def __init__(self, p: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p = p

    def forward_w_da(
        self,
        cont: Optional[Tensor] = None,
        cate: Optional[Tensor] = None,
        target: Optional[Tensor] = None,
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
        raise NotImplementedError()

    def forward(
        self,
        cont: Optional[Tensor] = None,
        cate: Optional[Tensor] = None,
        target: Tensor = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            cont (tensor): the input of shape (b, num_cont) representing the continuous values
            cate (tensor): the input of shape (b, num_cate) representing the categorical values
            target (tensor): the target labels of shape (b, num_classes)

        Returns:
            out (tensor): the output of shape (b, num_classes) representing the model's prediction
            loss (float): the model's supervised loss
        """
        if self.p > np.random.rand():
            out, target = self.forward_w_da(cont, cate, target)
        else:
            out = self.model(cont, cate)

        if target is not None:
            loss = self.criterion(out, target)
            return out, loss
        else:
            return out
