from model.base_model import BaseModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class LightGBM(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = 'LightGBM'
        self.model = LGBMClassifier(*args, **kwargs)


class XGBoost(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = 'XGBoost'
        self.model = XGBClassifier(*args, **kwargs)


class CatBoost(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = 'CatBoost'
        self.model = CatBoostClassifier(*args, **kwargs)
