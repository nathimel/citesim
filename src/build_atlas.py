"""Main driver script to build out regions of scientific literature using similarity-based retrieval."""

import hydra
import os
from omegaconf import DictConfig

from util import set_seed
from experiment import Experiment


import numpy as np
import pandas as pd
import plotnine as pn


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    set_seed(config.seed)

    exp = Experiment(config)

    # To disable huggingface parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    exp.expand_atlas()


if __name__ == "__main__":
    main()
