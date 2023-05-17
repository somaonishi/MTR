import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .base_da import BaseDATrainer

logger = logging.getLogger(__name__)


class FTTransWithCutmixTraniner(BaseDATrainer):
    def __init__(
        self,
        alpha: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Args:
            alpha (float): the parameter used in the cutmix augmentation
        """
        super().__init__(**kwargs)
        logger.info(f"Set mixup alpha to {alpha}.")
        self.alpha = alpha

    def get_lambda(self) -> float:
        """Return lambda"""
        if self.alpha > 0.0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        torch.tensor([lam]).float().to(self.device)
        return lam

    def mask_generator(self, x: Tensor, lam: float) -> Tensor:
        """
        Args:
            x (tensor): the input of shape (b, n)
            lam (float): The ratio to mask

        Returns:
            mask (tensor): the binary mask of shape (b, n)
        """
        b, n = x.shape
        ids_noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(ids_noise, dim=1)
        len_keep = int(n * lam)
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones(b, n, device=x.device)
        mask[torch.arange(b)[:, None], ids_keep] = mask[torch.arange(b)[:, None], ids_keep] * 0.0
        return mask

    def cutmix(self, x: Tensor, target: Tensor, lam: float) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (tensor): the input of shape (b, n)
            lam (float): The ratio to mask

        Returns:
            new_x (tensor): the output of shape (b, n) representing the new data after cutmix augmentation
            label (tensor): the mixed target labels of shape (b, num_classes)
        """
        indices = torch.randperm(x.shape[0])
        mask = self.mask_generator(x, lam)
        new_x = x * (1 - mask) + x[indices] * mask
        new_mask = (x == new_x).to(torch.float)
        new_lam = new_mask.mean(1).unsqueeze(1)
        label = target * new_lam + target[indices] * (1 - new_lam)
        return new_x, label

    def concat_data(self, cont: Optional[Tensor], cate: Optional[Tensor]) -> Tensor:
        """
        Args:
            cont (tensor): the input of shape (b, n) representing the continuous values
            cate (tensor): the input of shape (b, n) representing the categorical values

        Returns:
            x (tensor): A output of shape (b, n) representing the concatenated input values
        """
        if cont is not None and cate is not None:
            x = torch.cat([cont, cate], dim=1)
        elif cate is None:
            x = cont
        else:
            x = cate
        return x

    def cutmix_process(
        self, cont: Optional[Tensor], cate: Optional[Tensor], target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            cont (Tensor): The input tensor of shape (b, num_cont) representing the continuous values.
            cate (Tensor): The input tensor of shape (b, num_cate) representing the categorical values.
            target (Tensor): the target labels of shape (b, num_classes)

        Returns:
            cont (Tensor): the augmented continuous values
            cate (Tensor): the augmented categorical values
            target (Tensor): the mixed target labels of shape (b, num_classes)
        """
        lam = self.get_lambda()
        if self.datamodule.d_out > 1:
            target = F.one_hot(target, self.datamodule.d_out)
        x = self.concat_data(cont, cate)
        if cont is not None:
            n_cont = cont.shape[1]
        else:
            n_cont = 0
        x_new, target = self.cutmix(x, target, lam)
        if cont is not None:
            cont = x_new[:, :n_cont].to(torch.float)
        if cate is not None:
            cate = x_new[:, n_cont:].to(torch.long)
        return cont, cate, target

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
        cont, cate, target = self.cutmix_process(cont, cate, target)
        out = self.model(cont, cate)
        return out, target
