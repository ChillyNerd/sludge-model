from model.base_model import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogReg(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = "LogisticRegression"
        self.model = LogisticRegression(*args, **kwargs)
        self.scaler = StandardScaler()

    def fit(self, features, target, *args, **kwargs):
        features = self.scaler.fit_transform(features)
        super().fit(features, target)

    def predict(self, features):
        features = self.scaler.transform(features)
        return super().predict(features)
