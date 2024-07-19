from model.base_model import BaseModel
from sklearn.svm import SVC


class SupportVector(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = "SupportVector"
        self.model = SVC(*args, **kwargs)
