import hydra
import pandas as pd
from omegaconf.omegaconf import OmegaConf
from typing import Text
from sklearn.model_selection import train_test_split

@hydra.main(config_path="./configs", config_name="configs")
def data_split(cfg: Text) -> None:
    OmegaConf.to_yaml(cfg, resolve=True)
    """Split dataset into train/test.
    Args:
        cfg {Text}: path to config
    """

    dataset = pd.read_csv(cfg.train.dataset_csv)

    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=cfg.train.test_size,
        random_state=cfg.train.random_state
    )

    train_csv_path = cfg.train.trainset_path
    test_csv_path = cfg.train.testset_path
    train_dataset.to_csv(train_csv_path, index=False)
    test_dataset.to_csv(test_csv_path, index=False)


if __name__ == '__main__':
    data_split()
