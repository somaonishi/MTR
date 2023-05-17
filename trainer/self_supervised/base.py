from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.core.fttrans import ProjectionHead

from ..base import BaseTrainer

# Copied from https://github.com/clabrugere/pytorch-scarf/
class NTXent(nn.Module):
    def __init__(self, temperature=1.0):
        """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation
        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: Tensor, z_j: Tensor):
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch
        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples
        Returns:
            float: loss
        """
        batch_size = z_i.size(0)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device)).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss


class BaseSSLTrainer(BaseTrainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, tensorbord_dir="./self_supervised")
        new_head = ProjectionHead(self.model.head.linear.in_features)
        self.model.head = new_head.to(self.device)
        self.ssl_loss = NTXent()

    def train_per_epoch(self, dataloader: DataLoader, pbar_epoch: tqdm, epoch: int) -> dict:
        self.model.train()
        all_loss = []
        if self.scheduler is not None:
            self.scheduler.step()
        for batch in dataloader:
            pbar_epoch.update(1)
            self.optimizer.zero_grad()
            with autocast(enabled=self.scaler is not None):
                cont, cate, _ = self.apply_device(batch)
                _, loss = self.forward(cont, cate)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            all_loss.append(loss.item())
            scores = {"train/self-sl-loss": mean(all_loss)}
            pbar_epoch.set_description(f"epoch[{epoch} / {self.epochs}]")
            pbar_epoch.set_postfix(scores)
        return scores

    def forward_loss(self, z_i, z_j) -> float:
        return self.ssl_loss(z_i, z_j)

    @torch.no_grad()
    def eval(self, mode: str = "val") -> dict:
        self.model.eval()
        all_loss = []
        for batch in self.datamodule.dataloader(mode, self.eval_batch_size):
            with autocast(enabled=self.scaler is not None):
                cont, cate, _ = self.apply_device(batch)
                _, loss = self.forward(cont, cate)

            all_loss.append(loss.item())

        mean_loss = mean(all_loss)

        score = {f"{mode}/self-sl-loss": mean_loss}
        return score
