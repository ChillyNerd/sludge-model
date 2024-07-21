from model.base_model import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNeighbour(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = "KNearestNeighbours"
        self.model = KNeighborsClassifier(*args, **kwargs)
        self.scaler = StandardScaler()

    def fit(self, features, target, *args, **kwargs):
        features = self.scaler.fit_transform(features)
        super().fit(features, target)

    def predict(self, features):
        features = self.scaler.transform(features)
        return super().predict(features)

