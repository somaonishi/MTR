import logging
from typing import Optional, Tuple

import torch
from torch import Tensor

from ..supervised.fttrans_w_scarf import SCARFDA
from .base import BaseSSLTrainer

logger = logging.getLogger(__name__)


class FTTransSCARFSSLTrainer(BaseSSLTrainer):
    def __init__(
        self,
        da_mode: str = "scarf",
        mask_ratio: float = 0.2,
        **kwargs,
    ) -> None:
        """
        Args:
            da_mode (str): the data augmentation mode
            mask_ratio (float): the ratio of data points to be masked
        """
        super().__init__(**kwargs)
        logger.info(f"DA mode is {da_mode}.")

        self.mask_ratio = mask_ratio
        self.da_mode = da_mode

        self.scarf_da_train = SCARFDA(
            self.datamodule.train,
            self.datamodule.continuous_columns,
            self.datamodule.categorical_columns,
            mask_ratio,
            self.device,
        )
        self.scarf_da_val = SCARFDA(
            self.datamodule.val,
            self.datamodule.continuous_columns,
            self.datamodule.categorical_columns,
            mask_ratio,
            self.device,
        )

    def vime_augument(self, x) -> Tensor:
        """
        Args:
            x (tensor): the input of shape (b, n)
        Returns:
            x_tilde (tensor): the output of shape (b, n) representing the new data after VIME augmentation
        """
        mask = torch.bernoulli(torch.ones(x.shape) * self.mask_ratio)
        mask = mask.to(torch.float).to(self.device)

        no, dim = x.shape
        x_bar = torch.zeros([no, dim]).to(self.device)
        for i in range(dim):
            idx = torch.randperm(no)
            x_bar[:, i] = x[idx, i]
        x_tilde = x * (1 - mask) + x_bar * mask
        return x_tilde

    def apply_data_augmentation(
        self, cont: Optional[Tensor] = None, cate: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            cont (tensor): the input of shape (b, num_cont) representing the continuous values
            cate (tensor): the input of shape (b, num_cate) representing the categorical values
        Returns:
            cont (tensor): the output of shape (b, num_cont) representing the new data after SCARF augmentation
            cate (tensor): the output of shape (b, num_cate) representing the new data after SCARF augmentation
        """
        if self.da_mode == "vime":
            if cont is not None:
                cont = self.vime_augument(cont).to(torch.float)
            if cate is not None:
                cate = self.vime_augument(cate).to(torch.long)
        elif self.da_mode == "scarf":
            if self.model.train:
                scarf_da = self.scarf_da_train
            else:
                scarf_da = self.scarf_da_val

            if cont is not None:
                cont = scarf_da(cont, "cont").to(torch.float)
            if cate is not None:
                cate = scarf_da(cate, "cate").to(torch.long)
        return cont, cate

    def forward(self, cont: Optional[Tensor] = None, cate: Optional[Tensor] = None) -> Tuple[None, float]:
        """
        Args:
            cont (tensor): the input of shape (b, num_cont) representing the continuous values
            cate (tensor): the input of shape (b, num_cate) representing the categorical values

        Returns:
            loss (float): the model's contrastive loss
        """
        cont_da, cate_da = self.apply_data_augmentation(cont, cate)
        z_0 = self.model(cont, cate)
        z_1 = self.model(cont_da, cate_da)
        loss = self.forward_loss(z_0, z_1)
        return None, loss
