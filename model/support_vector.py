from model.base_model import BaseModel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class SupportVector(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = "SupportVector"
        self.model = SVC(*args, **kwargs)
        self.scaler = StandardScaler()

    def fit(self, features, target):
        features = self.scaler.fit_transform(features)
        super().fit(features, target)

    def predict(self, features):
        features = self.scaler.transform(features)
        return super().predict(features)
