import logging
from statistics import mean
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from torch import Tensor
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base import BaseTrainer

logger = logging.getLogger(__name__)


class BaseSupervisedTrainer(BaseTrainer):
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
            loss (float): the model's supervised loss
        """
        raise NotImplementedError()

    def train_per_epoch(self, dataloader: DataLoader, pbar_epoch: tqdm, epoch: int) -> dict:
        self.model.train()
        all_loss = []
        if self.scheduler is not None:
            self.scheduler.step()
        for batch in dataloader:
            pbar_epoch.update(1)
            self.optimizer.zero_grad()
            with autocast(enabled=self.scaler is not None):
                cont, cate, target = self.apply_device(batch)
                _, loss = self.forward(cont, cate, target)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            all_loss.append(loss.item())
            scores = {"train/loss": mean(all_loss)}

            pbar_epoch.set_description(f"epoch[{epoch} / {self.epochs}]")
            pbar_epoch.set_postfix(scores)
        return scores

    @torch.no_grad()
    def eval(self, mode: str = "test"):
        self.model.eval()
        all_target = []
        all_pred = []
        all_loss = []
        for batch in self.datamodule.dataloader(mode, self.eval_batch_size):
            with autocast(enabled=self.scaler is not None):
                cont, cate, target = self.apply_device(batch)
                out = self.model(cont, cate)
                loss = self.criterion(out, target)

            all_target.append(target.cpu())
            all_pred.append(out.cpu())
            all_loss.append(loss.item())

        all_target = torch.cat(all_target, dim=0)
        all_pred = torch.cat(all_pred, dim=0)
        mean_loss = mean(all_loss)

        score = {f"{mode}/loss": mean_loss}
        if self.datamodule.task == "binary":
            label = (all_pred.numpy() > 0.5).astype(np.int)
            all_pred = torch.sigmoid(all_pred.float()).numpy()
            score.update(
                {
                    f"{mode}/acc": accuracy_score(all_target, label),
                    f"{mode}/auc": roc_auc_score(all_target, all_pred),
                }
            )
        elif self.datamodule.task == "multiclass":
            label = all_pred.argmax(1).numpy()
            score.update({f"{mode}/acc": accuracy_score(all_target, label)})
        else:
            assert self.datamodule.task == "regression"
            score.update(
                {f"{mode}/rmse": mean_squared_error(all_target, all_pred.numpy()) ** 0.5 * self.datamodule.y_std}
            )
        return score
