import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score
from typing import Text, Dict
import hydra
from omegaconf.omegaconf import OmegaConf

from visualize import plot_confusion_matrix

def convert_to_labels(indexes, labels):
    return [labels[i] for i in indexes]

def write_confusion_matrix_data(y_true, predicted, labels, filename):
    assert len(predicted) == len(y_true)
    predicted_labels = convert_to_labels(predicted, labels)
    true_labels = convert_to_labels(y_true, labels)
    cf = pd.DataFrame(list(zip(true_labels, predicted_labels)), columns=["y_true", "predicted"])
    cf.to_csv(filename, index=False)

@hydra.main(config_path="../configs", config_name="configs")
def evaluate_model(cfg: Text) -> None:
    OmegaConf.to_yaml(cfg, resolve=True)
    """Evaluate model.
    Args:
        config_path {Text}: path to config
    """

    #logger = get_logger('EVALUATE', log_level=config['base']['log_level'])

    #logger.info('Load model')
    model_path = cfg.model.model_path
    model = joblib.load(model_path)

    #logger.info('Load test dataset')
    test_df = pd.read_csv(cfg.train.features_test_path)

    #logger.info('Evaluate (build report)')
    target_column=cfg.train.target_column
    y_test = test_df.loc[:, target_column].values
    X_test = test_df.drop(target_column, axis=1).values

    prediction = model.predict(X_test)
    f1 = f1_score(y_true=y_test, y_pred=prediction, average='macro')

    labels = ['Normal', 'Suspect','Pathological']
    cm = confusion_matrix(y_test, prediction)

    report = {
        'f1': f1,
        'cm': cm,
        'actual': y_test,
        'predicted': prediction
    }

    #logger.info('Save metrics')
    # save f1 metrics file
    reports_folder = Path(cfg.train.reports_dir)
    metrics_path = reports_folder / cfg.train.metrics_file

    json.dump(
        obj={'f1_score': report['f1']},
        fp=open(metrics_path, 'w')
    )

    #logger.info(f'F1 metrics file saved to : {metrics_path}')

    #logger.info('Save confusion matrix')
    # save confusion_matrix.png
    plt = plot_confusion_matrix(cm=report['cm'],
                                target_names=labels,
                                normalize=False)
    confusion_matrix_png_path = reports_folder / cfg.train.confusion_matrix_image
    plt.savefig(confusion_matrix_png_path)
    #logger.info(f'Confusion matrix saved to : {confusion_matrix_png_path}')

    confusion_matrix_data_path = reports_folder / cfg.train.confusion_matrix_data
    write_confusion_matrix_data(y_test, prediction, labels=labels, filename=confusion_matrix_data_path)
    #logger.info(f'Confusion matrix data saved to : {confusion_matrix_data_path}')


if __name__ == '__main__':
    evaluate_model()
