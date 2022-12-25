import hydra
import pandas as pd
from omegaconf.omegaconf import OmegaConf
from typing import Text

@hydra.main(config_path="./configs", config_name="configs")
def data_load(cfg: Text) -> None:
    OmegaConf.to_yaml(cfg, resolve=True)
    """Load raw data and saves it to path
    Args:
        cfg {Text}: path to config
    """
    data = pd.read_csv(cfg.train.ingest)
    data.to_csv(cfg.train.dataset_csv, index=False)


if __name__ == '__main__':
    data_load()