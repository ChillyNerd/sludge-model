from model import NeuralNetwork
from data.data_prepare import prepare_dataset, Dataset, Target
from sklearn.metrics import classification_report
import os
import logging
from torch import tensor, float
from utils.config import Config


def main():
    config = Config('config.yaml')
    data_dir = os.path.join(os.getcwd(), 'data/raw')
    for dataset_type in [Dataset.RAW, Dataset.RAW_VELOCITY, Dataset.VELOCITY]:
        datasets = prepare_dataset(os.path.join(data_dir, 'correct_data.csv'), dataset_type, Target.SLUDGE)
        log = logging.getLogger("Main")
        for label, dataset in datasets.items():
            log.debug(f'--------------- {label} --------------------')
            x_train, x_test, y_train, y_test = dataset
            model = NeuralNetwork(input_shape=x_train.shape[1], output_shape=len(y_train.unique()))
            model.fit(features=x_train, target=y_train, x_test=x_test, y_test=y_test)
            y_pred = model.predict(x_test)
            cr = classification_report(y_test, y_pred, output_dict=True)
            log.debug(f'Accuracy {cr["accuracy"]}')
            log.debug(f'Macro f score {cr["macro avg"]["f1-score"]}')
            log.debug(f'Weighted f score {cr["weighted avg"]["f1-score"]}')


if __name__ == '__main__':
    main()
