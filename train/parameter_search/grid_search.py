from typing import Dict, List, Type
from train.parameter_search.parameter_vector import ParameterVector
from sklearn.metrics import classification_report
from model import BaseModel
import logging


class GridSearch:
    def __init__(self, model_class: Type[BaseModel], parameters: Dict[str, List]):
        self.model_class: Type[BaseModel] = model_class
        self.parameters: ParameterVector = ParameterVector(parameters)
        self.models = {}
        self.log = logging.getLogger("GridSearch")

    def fit_predict(self, x_train, y_train, x_test, y_test):
        best_parameters = None
        best_score = 0
        for parameter_vector in self.parameters:
            model = self.model_class(**parameter_vector)
            self.log.debug(f'Inspecting {model.name} with parameters {parameter_vector}')
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            cr = classification_report(y_test, y_pred, output_dict=True)
            self.log.debug(f'Accuracy {cr["accuracy"]}')
            self.log.debug(f'Macro f score {cr["macro avg"]["f1-score"]}')
            self.log.debug(f'Weighted f score {cr["weighted avg"]["f1-score"]}')
            if best_score < cr["macro avg"]["f1-score"]:
                best_score = cr["macro avg"]["f1-score"]
                best_parameters = parameter_vector
        self.log.info(f'Best parameters for {self.model_class} are {best_parameters} with f score {best_score}')
