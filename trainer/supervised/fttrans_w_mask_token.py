import logging
from typing import Optional, Tuple

from torch import Tensor

from .base_da import BaseDATrainer

logger = logging.getLogger(__name__)


class FTTransWithMaskTokenTraniner(BaseDATrainer):
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
        out = self.model(
            cont,
            cate,
            mask_ratio=self.mask_ratio,
            bias_after_mask=self.bias_after_mask,
        )
        return out, target
