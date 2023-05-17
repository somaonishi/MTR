from typing import Optional, Tuple

import torch
from torch import Tensor

from .core import FTTransformer


def mask_generator(x: Tensor, lam: float) -> Tensor:
    """
    Args:
        x (tensor): the input of shape (b, n, d)
        lam (float): the scalar coefficient to keep unmasked.

    Returns:
        mask (tensor): the binary mask of shape (b, n, d)
    """
    b, n, d = x.shape
    ids_noise = torch.rand(b, d, device=x.device)
    ids_shuffle = torch.argsort(ids_noise, dim=1)
    len_unmask = int(lam * d)
    ids_unmask = ids_shuffle[:, :len_unmask]
    mask = torch.zeros(b, d, device=x.device)
    mask[torch.arange(b)[:, None], ids_unmask] = 1
    mask = mask.unsqueeze(1)
    mask = mask.repeat(1, n, 1)
    return mask


def hidden_mix(x: Tensor, target: Optional[Tensor], lam: float) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Args:
        x (tensor): the input of shape (b, n, d)
        target (tensor): the target labels of shape (b, num_classes)
        lam (float): the scalar coefficient to keep unmasked.

    Returns:
        new_x (tensor): the output of shape (b, n, d) representing the mixed input data
        label (tensor): the mixed target labels of shape (b, num_classes)
    """
    mask = mask_generator(x, lam)
    indices = torch.randperm(x.shape[0])
    new_x = x * mask + x[indices] * (1 - mask)
    if target is not None:
        label = lam * target + (1 - lam) * target[indices]
        return new_x, label
    else:
        return new_x, None


class FTTransformerWithHiddenMix(FTTransformer):
    def forward(
        self,
        x_num: Optional[Tensor],
        x_cat: Optional[Tensor],
        target: Tensor = None,
        lam: Tensor = None,
    ) -> Tensor:
        """
        Args:
            x_num (tensor): the input of shape (b, n) representing the numerical values
            x_cat (tensor): the input of shape (b, n) representing the categorical values
            target (tensor): the target labels of shape (b, num_classes)
            lam (tensor): the scalar coefficient to keep unmasked.

        Returns:
            tensor: the output of shape (b, num_classes) representing the model's prediction
        """
        x = self.feature_tokenizer(x_num, x_cat)
        x = x + self.pos_embedding
        if target is not None or lam is not None:
            assert lam is not None
            x, new_target = hidden_mix(x, target, lam)
            x = self.cls_token(x)
            x = self.transformer(x)
            x = x[:, -1]
            x = self.normalization(x)
            x = self.activation(x)
            return self.head(x), new_target
        else:
            x = self.cls_token(x)
            x = self.transformer(x)
            x = x[:, -1]
            x = self.normalization(x)
            x = self.activation(x)
            return self.head(x)
