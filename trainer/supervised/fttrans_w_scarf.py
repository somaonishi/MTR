import logging
from typing import Optional, Tuple

import torch
from torch import Tensor

from .base_da import BaseDATrainer

logger = logging.getLogger(__name__)


class SCARFDA:
    def __init__(self, df, cont_cols: list, cate_cols: list, mask_ratio: float, device: str) -> None:
        """
        Args:
            df (DataFrame): the original dataset
            cont_cols (List): the column names of continuous values
            cate_cals (List): the column names of categorica values
            mask_ratio (float): the ratio of data points to be masked
            device (str): the device to be used for computations (e.g., "cpu", "cuda")
        """
        self.cont = torch.tensor(df[cont_cols].values, device=device)
        self.cate = torch.tensor(df[cate_cols].values, device=device)
        self.device = device
        self.mask_ratio = mask_ratio

    def scarf_augument(self, x: Tensor, all_data: Tensor) -> Tensor:
        """
        Args:
            x (tensor): the input of shape (b, n)
            all_data (tensor): the entire dataset

        Returns:
            x_tilde (tensor): the output of shape (b, n) representing the masked data
        """
        mask = torch.bernoulli(torch.ones(x.shape) * self.mask_ratio)
        mask = mask.to(torch.float).to(self.device)
        batch_size = x.shape[0]
        no, dim = all_data.shape
        x_bar = torch.zeros([batch_size, dim]).to(self.device)
        for i in range(dim):
            idx = torch.randint(0, no, (batch_size,))
            x_bar[:, i] = all_data[idx, i]
        x_tilde = x * (1 - mask) + x_bar * mask
        return x_tilde

    def __call__(self, x: Tensor, mode: str) -> Tensor:
        """
        Args:
            x (tensor): the input of shape (b, n)
            mode (str): the mode of data should be either "cont" or "cate"

        Returns:
            tensor: the output of shape (b, n) representing the new data after SCARF augmentation
        """
        if mode == "cont":
            data = self.cont
        elif mode == "cate":
            data = self.cate
        else:
            raise ValueError(f"unexpected values: {mode}")

        return self.scarf_augument(x, data)


class FTTransWithSCARFTraniner(BaseDATrainer):
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

    def vime_augument(self, x: Tensor) -> Tensor:
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
        cont, cate = self.apply_data_augmentation(cont, cate)
        out = self.model(cont, cate)
        return out, target
