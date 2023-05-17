import logging
import os
import time

import hydra

from runner import SelfSLRunner, SupervisedRunner
from trainer.utils import fix_seed

logger = logging.getLogger(__name__)


def check_existence_result() -> bool:
    if os.path.exists("results.json"):
        print("Skip this trial, because it has already been run.")
        return True
    else:
        return False


@hydra.main(config_path="conf", config_name="config.yaml", version_base=None)
def main(config):
    if check_existence_result():
        return

    fix_seed(config.seed)

    if config.train_mode == "supervised":
        runner = SupervisedRunner(config)
    elif config.train_mode == "self_sl":
        runner = SelfSLRunner(config)
    else:
        raise ValueError(f"train mode {config.train_mode} is not defined.")

    runner.run()
    time.sleep(1)


if __name__ == "__main__":
    main()
