import logging
from typing import Optional, Tuple

from torch import Tensor

from .base import BaseSupervisedTrainer

logger = logging.getLogger(__name__)


class FTTransTraniner(BaseSupervisedTrainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

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
            loss (tensor): the model's loss
        """
        out = self.model(cont, cate)
        if target is not None:
            loss = self.criterion(out, target)
            return out, loss
        else:
            return out
