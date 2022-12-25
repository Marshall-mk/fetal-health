import joblib
import hydra
import pandas as pd
from omegaconf.omegaconf import OmegaConf
from typing import Text


from train.model import model


@hydra.main(config_path="./configs", config_name="configs")
def train_model(cfg: Text) -> None:
    OmegaConf.to_yaml(cfg, resolve=True)
    """Train model.
    Args:
        config_path {Text}: path to config
    """

    estimator_name = cfg.model.estimator_name
    #logger.info(f'Estimator: {estimator_name}')

    #logger.info('Load train dataset')
    train_df = pd.read_csv(cfg.train.features_train_path)

    #logger.info('Train model')
    model = model(
        df=train_df,
        target_column=cfg.train.target_column,
        estimator_name=estimator_name,
        param_grid=cfg.model.estimators[estimator_name]['param_grid'],
        cv=cfg.model.cv
    )
    #logger.info(f'Best score: {model.best_score_}')

    #logger.info('Save model')
    models_path = cfg.model.model_path
    joblib.dump(model, models_path)


if __name__ == '__main__':
    train_model()
