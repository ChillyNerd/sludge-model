from model.base_model import BaseModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from enum import Enum
from exceptions.wrong_boosting import WrongBoostingType


class BoostingType(Enum):
    lightgbm = 'LightGBM'
    xgboost = 'XGBoost'
    catboost = 'CatBoost'


class Boosting(BaseModel):
    def __init__(self, boosting_type: BoostingType, *args, **kwargs):
        super().__init__()
        self.name = boosting_type.value
        match boosting_type:
            case BoostingType.lightgbm:
                self.__model = LGBMClassifier(*args, **kwargs)
            case BoostingType.xgboost:
                self.__model = XGBClassifier(*args, **kwargs)
            case BoostingType.catboost:
                self.__model = CatBoostClassifier(*args, **kwargs)
            case _:
                raise WrongBoostingType()

    def fit(self, features, target):
        self.__model.fit(features, target)

    def predict(self, features):
        return self.__model.predict(features)
