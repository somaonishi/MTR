import logging

import torch
from hydra.utils import to_absolute_path

from data import get_datamodule

logger = logging.getLogger(__name__)


class BaseRunner:
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Select {self.device}")

        config.data_dir = to_absolute_path(config.data_dir)
        self.datamodule = get_datamodule(config)
