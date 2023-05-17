import logging
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
from timm.scheduler.scheduler import Scheduler
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import TabularDatamodule
from model.core.fttrans import FTTransformer

from .utils import EarlyStopping, auto_batch_size, save_json

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(
        self,
        datamodule: TabularDatamodule,
        batch_size: Union[int, str],
        eval_batch_size: int,
        model: FTTransformer,
        optimizer: Optimizer,
        criterion: _Loss,
        epochs: int,
        device: torch._C.device,
        patience: int = 16,
        eval_metric: str = "val/loss",
        eval_less_is_better: bool = True,
        scheduler: Scheduler = None,
        mixed_fp16: bool = False,
        tensorbord_dir: str = "./supervised",
        save_model: bool = False,
    ) -> None:
        """
        Args:
            datamodule (TabularDatamodule): providing the dataset and dataloaders
            batch_size (Union[int, str]): the batch size used for training
            eval_batch_size (int): the batch size used for evaluation
            model (FTTransformer): the FTTransformer model used for training
            optimizer (Optimizer): used for updating the model parameters during training
            criterion (_Loss): the loss function
            epochs (int): the total number of training epochs
            device (torch._C.device): the device to be used for computations (e.g., "cpu", "cuda")
            patience (int): the number of epochs to wait for improvement
            eval_metric (str): the evaluation metric
            eval_less_is_better (bool): the flag representing whether the lower value indicates better performance
            scheduler (Scheduler): used for adjusting the learning rate during training
            mixed_fp16 (bool): whether to use mixed precision training with FP16
            tensorbord_dir (str): the directory path to save TensorBoard logs
            save_model (bool): whether to save the best model during training
        """
        self.datamodule = datamodule

        if batch_size == "auto":
            batch_size = auto_batch_size(len(datamodule.train))
            logger.info(f"Use auto batch size; choose {batch_size}")

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.epochs = epochs
        self.eval_metric = eval_metric
        self.eval_less_is_better = eval_less_is_better

        self.early_stopping = EarlyStopping(patience)

        self.scheduler = scheduler
        if mixed_fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.save_model = save_model

        self.device = device

        os.makedirs(tensorbord_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorbord_dir)

    def apply_device(self, data: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            data (tensor): the input of shape (b, n)

        Returns:
            cont (tensor): the output of shape (b, num_cont) representing the continuous values
            cate (tensor): the output of shape (b, num_cate) representing the categorical values
            target (tensor): the target labels of shape (b, num_classes)
        """
        cont = data["continuous"]
        cate = data["categorical"]
        if "target" in data:
            target = data["target"].to(self.device)
        else:
            target = None

        if self.datamodule.task == "multiclass" and target is not None:
            target = target.squeeze(1)

        if cont != []:
            cont = cont.to(self.device)
        else:
            cont = None
        if cate != []:
            cate = cate.to(self.device)
        else:
            cate = None
        return cont, cate, target

    def forward(self, cont: Optional[Tensor] = None, cate: Optional[Tensor] = None):
        raise NotImplementedError()

    def train_dataloader(self):
        return self.datamodule.dataloader("train", batch_size=self.batch_size)

    def train_per_epoch(self, dataloader: DataLoader, pbar_epoch: tqdm, epoch: int):
        raise NotImplementedError()

    @torch.no_grad()
    def eval(self, mode: str = "test"):
        raise NotImplementedError()

    def train(self) -> None:
        if self.eval_less_is_better:
            best_score = np.inf
        else:
            best_score = -np.inf

        dataloader = self.train_dataloader()
        for epoch in range(1, self.epochs + 1):
            with tqdm(total=len(dataloader), bar_format="{l_bar}{bar}{r_bar}{bar:-10b}") as pbar_epoch:
                scores = self.train_per_epoch(dataloader, pbar_epoch, epoch)
                scores.update(self.eval("val"))
                pbar_epoch.set_postfix(scores)
                for tag, score in scores.items():
                    self.writer.add_scalar(tag, score, epoch)

            if self.eval_less_is_better:
                self.early_stopping(scores[self.eval_metric], self.model)
                if best_score > scores[self.eval_metric]:
                    best_score = scores[self.eval_metric]
            else:
                self.early_stopping(-scores[self.eval_metric], self.model)
                if best_score < scores[self.eval_metric]:
                    best_score = scores[self.eval_metric]
            if self.early_stopping.early_stop:
                logger.info(f"early stopping {epoch} / {self.epochs}")
                break
        self.model.load_state_dict(self.early_stopping.best_model)

    def print_evaluate(self) -> None:
        scores = self.eval()
        if self.save_model:
            torch.save(self.early_stopping.best_model, "./best_model")
        save_json(scores, "./")
        for key, score in scores.items():
            logger.info(f"{key}: {score}")
