from abc import ABC


class BaseModel(ABC):
    def __init__(self, *args, **kwargs):
        self.name = "BaseModel"
        self.model = None

    def fit(self, features, target, *args, **kwargs):
        self.model.fit(features, target)

    def predict(self, features):
        return self.model.predict(features)

    def predict_proba(self, features):
        return self.model.predict_proba(features)
