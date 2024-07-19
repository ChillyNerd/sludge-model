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
                self.model = LGBMClassifier(*args, **kwargs)
            case BoostingType.xgboost:
                self.model = XGBClassifier(*args, **kwargs)
            case BoostingType.catboost:
                self.model = CatBoostClassifier(*args, **kwargs)
            case _:
                raise WrongBoostingType()
