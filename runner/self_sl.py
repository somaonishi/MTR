import logging

import torch.nn as nn

import model
import trainer
from model import FTTransformer
from trainer import FTTransTraniner
from trainer.self_supervised.base import BaseSSLTrainer

from .base import BaseRunner

logger = logging.getLogger(__name__)


class SelfSLRunner(BaseRunner):
    def __init__(self, config) -> None:
        super().__init__(config)

    def self_sl_init(self):
        Model = getattr(model, self.config.model.name)
        logger.info(f"Model is {self.config.model.name}.")

        num_features = self.datamodule.num_continuous_features + self.datamodule.num_categorical_features
        self.model: FTTransformer = Model.make_default(
            num_features=num_features,
            n_num_features=self.datamodule.num_continuous_features,
            cat_cardinalities=self.datamodule.cat_cardinalities,
            last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
            d_out=self.datamodule.d_out,
        )
        self.model.to(self.device)

        optimizer = self.model.make_default_optimizer()
        criterion = None
        Trainer = getattr(trainer, self.config.model.trainer)
        logger.info(f"Trainer is {self.config.model.trainer}.")

        assert issubclass(Trainer, BaseSSLTrainer), "Trainer must be a subclass of BaseSLTrainer."
        del self.config.model.params.p
        self.self_sl_trainer: BaseSSLTrainer = Trainer(
            datamodule=self.datamodule,
            batch_size=self.config.batch_size,
            eval_batch_size=self.config.eval_batch_size,
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            epochs=200,
            patience=10,
            eval_metric="val/self-sl-loss",
            eval_less_is_better=True,
            mixed_fp16=self.config.mixed_fp16,
            **self.config.model.params,
        )

    def sl_init(self):
        state_dict = self.model.state_dict()
        num_features = self.datamodule.num_continuous_features + self.datamodule.num_categorical_features
        self.model = FTTransformer.make_default(
            num_features=num_features,
            n_num_features=self.datamodule.num_continuous_features,
            cat_cardinalities=self.datamodule.cat_cardinalities,
            last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
            d_out=self.datamodule.d_out,
        )
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        # for p in self.model.parameters():
        #     p.requires_grad = False
        # for p in self.model.head.parameters():
        #     p.requires_grad = True

        self.model.to(self.device)

        optimizer = self.model.make_default_optimizer()
        criterion = (
            nn.BCEWithLogitsLoss()
            if self.datamodule.task == "binary"
            else nn.CrossEntropyLoss()
            if self.datamodule.task == "multiclass"
            else nn.MSELoss()
        )
        self.sl_trainer = FTTransTraniner(
            datamodule=self.datamodule,
            batch_size=self.config.batch_size,
            eval_batch_size=self.config.eval_batch_size,
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            epochs=self.config.epochs,
            patience=self.config.patience,
            eval_metric=self.config.eval.metric,
            eval_less_is_better=self.config.eval.less_is_better,
            mixed_fp16=self.config.mixed_fp16,
            save_model=self.config.save_model,
            tensorbord_dir="./self_supervised",
        )

    def run(self):
        self.self_sl_init()
        self.self_sl_trainer.train()
        self.sl_init()
        self.sl_trainer.train()
        self.sl_trainer.print_evaluate()
