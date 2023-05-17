import logging
from typing import Optional, Tuple

from torch import Tensor

from .base import BaseSSLTrainer

logger = logging.getLogger(__name__)


class FTTransMaskTokenSSLTrainer(BaseSSLTrainer):
    def __init__(
        self,
        mask_ratio: float = 0.1,
        bias_after_mask: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            mask_ratio (float): the ratio of data points to be masked
            bias_after_mask (bool): whether to add the positional embedding before or after masking
        """
        super().__init__(**kwargs)

        self.mask_ratio = mask_ratio
        self.bias_after_mask = bias_after_mask

    def forward(self, cont: Optional[Tensor] = None, cate: Optional[Tensor] = None) -> Tuple[None, float]:
        """
        Args:
            cont (tensor): the input of shape (b, num_cont) representing the continuous values
            cate (tensor): the input of shape (b, num_cate) representing the categorical values

        Returns:
            loss (float): the model's contrastive loss
        """
        z_0 = self.model(cont, cate)
        z_1 = self.model(
            cont,
            cate,
            mask_ratio=self.mask_ratio,
            bias_after_mask=self.bias_after_mask,
        )
        loss = self.forward_loss(z_0, z_1)
        return None, loss
