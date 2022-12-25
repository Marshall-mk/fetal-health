import hydra
import pandas as pd
from omegaconf.omegaconf import OmegaConf
from typing import Text
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import RobustScaler

@hydra.main(config_path="./configs", config_name="configs")
def featurize(cfg: Text) -> None:
    OmegaConf.to_yaml(cfg, resolve=True)
    """Create new features.
    Args:
        cfg {Text}: path to config
    """

    # ingest
    train = pd.read_csv(cfg.train.trainset_path)
    test = pd.read_csv(cfg.train.testset_path)

    # features train
    x_train = train.drop(['fetal_health'], axis=1)
    n_sample, n_featrues = x_train.shape
    y_train = train['fetal_health']
    ros = RandomOverSampler(random_state=42)
    x_resampled, y_resampled = ros.fit_resample(x_train, y_train)

    # features test
    x_test = test.drop(['fetal_health'], axis=1)
    y_test = test['fetal_health']    

    # Scale train/ test predictors based on training data
    ro_scaler = RobustScaler().fit(x_resampled)
    x_train_scaled = ro_scaler.transform(x_resampled)
    x_test_scaled = ro_scaler.transform(x_test)

    idx = {1:0, 2:1, 3:2}

    # combine features and targets
    new_train = x_train_scaled
    new_train['fetal_health'] = y_resampled.map(idx)
    new_test = x_test_scaled
    new_test['fetal_health'] = y_test.map(idx)

    # save
    train_path = cfg.train.features_train_path
    test_path = cfg.train.features_test_path
    new_train.to_csv(train_path, index=False)
    new_test.to_csv(test_path, index=False)


if __name__ == '__main__':
    featurize()
