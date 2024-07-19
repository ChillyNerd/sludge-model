from model.base_model import BaseModel
from sklearn.neighbors import KNeighborsClassifier


class KNeighbour(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = "KNearestNeighbours"
        self.model = KNeighborsClassifier(*args, **kwargs)
