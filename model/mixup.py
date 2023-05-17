from typing import Optional, Tuple

import torch
from torch import Tensor

from .core import FTTransformer


def mixup_process(x: Tensor, lam: float, target_reweighted: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Args:
        x (tensor): the input of shape (b, n, d)
        lam (float): the mixing coefficient
        target_reweighted (tensor): the target labels of shape (b, num_classes)

    Returns:
        new_x (tensor): the output of shape (b, n, d) representing the mixed input data
        target_reweighted (tensor): the mixed target labels of shape (b, num_classes)
    """
    indices = torch.randperm(x.shape[0])
    new_x = x * (1 - lam) + x[indices] * lam
    if target_reweighted is not None:
        target_shuffled_onehot = target_reweighted[indices]
        target_reweighted = target_reweighted * (1 - lam) + target_shuffled_onehot * lam
        return new_x, target_reweighted
    else:
        return new_x


class FTTransformerWithMixup(FTTransformer):
    def forward(
        self,
        x_num: Optional[Tensor],
        x_cat: Optional[Tensor],
        target: Tensor = None,
        lam: Tensor = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x_num (tensor): the input of shape (b, n) representing the numerical values
            x_cat (tensor): the input of shape (b, n) representing the categorical values
            target (tensor): the target labels of shape (b, num_classes)
            lam (tensor): the mixing coefficient

        Returns:
            x (tensor): the output of shape (b, num_classes) representing the model's prediction
            new_target (tensor): the mixed target labels of shape (b, num_classes)
        """
        """
        Supervised
        """
        x = self.feature_tokenizer(x_num, x_cat)
        x = x + self.pos_embedding
        x = self.cls_token(x)
        x = self.transformer(x)
        x = x[:, -1]
        x = self.normalization(x)
        x = self.activation(x)

        if target is not None:
            assert lam is not None
            x, new_target = mixup_process(x, lam, target)
            x = self.head(x)
            return x, new_target
        else:
            return self.head(x)

    def forward_no_labelmix(
        self,
        x_num: Optional[Tensor],
        x_cat: Optional[Tensor],
        alpha: float = 0.0,
    ) -> Tensor:
        """
        Args:
            x_num (tensor): the input of shape (b, n) representing the numerical values
            x_cat (tensor): the input of shape (b, n) representing the categorical values
            alpha (float): the mixing coefficient

        Returns:
            tensor: the output of shape (b, num_classes) representing the model's prediction
        """
        """
        Self-SL
        0.0      <= alpha <= 1.0
        original <=   x   <= other
        """
        x = self.feature_tokenizer(x_num, x_cat)
        x = x + self.pos_embedding
        x = self.cls_token(x)
        x = self.transformer(x)
        x = x[:, -1]
        x = self.normalization(x)
        x = self.activation(x)
        x = mixup_process(x, alpha)
        return self.head(x)
