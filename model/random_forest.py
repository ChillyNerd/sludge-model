from model.base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier


class RandomForest(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = "RandomForest"
        self.model = RandomForestClassifier(*args, **kwargs)
