from train.parameter_search.grid_search import GridSearch
from model import NeuralNetwork
from data.data_prepare import prepare_dataset, Dataset, Target
import os
import logging
from utils.config import Config


def main():
    config = Config('config.yaml')
    data_dir = os.path.join(os.getcwd(), 'data/raw')
    datasets = prepare_dataset(os.path.join(data_dir, 'correct_data.csv'), Dataset.RAW, Target.SLUDGE)
    log = logging.getLogger("Main")
    model = NeuralNetwork()
    for label, dataset in datasets.items():
        log.debug(f'--------------- {label} --------------------')
        x_train, x_test, y_train, y_test = dataset
        model.fit(features=x_train, target=y_train, x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    main()
