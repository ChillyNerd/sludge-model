from train.parameter_search.grid_search import GridSearch
from model import *
from data.data_prepare import prepare_dataset, Dataset, Target
import os
import logging
from utils.config import Config

config = Config('../../config.yaml')
data_dir = os.path.join(os.getcwd(), '../../data/raw')
datasets = prepare_dataset(os.path.join(data_dir, 'correct_data.csv'), Dataset.RAW, Target.SLUDGE)
log = logging.getLogger("Main")
model_parameters = {
    KNeighbour: {
        "n_neighbors": list(range(2, 30, 1))
    },
    LogReg: {
        "max_iter": list(range(100, 1000, 100)),
        "n_jobs": [16],
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs']
    },
    SupportVector: {
        "kernel": ['linear', 'rbf', 'poly', 'sigmoid']
    },
    LightGBM: {
        'verbose': [0],
        'num_leaves': list(range(5, 100, 5)),
        'n_estimators': list(range(10, 500, 50))
    },
    CatBoost: {
        'iterations': list(range(10, 500, 50)),
        'learning_rate': [0.01, 0.1, 1]
    },
    RandomForest: {
        'n_estimators': list(range(10, 500, 50))
    },
}
for label, dataset in datasets.items():
    log.debug(f'--------------- {label} --------------------')
    x_train, x_test, y_train, y_test = dataset
    for model, parameters in model_parameters.items():
        gs = GridSearch(model, parameters)
        gs.fit_predict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
