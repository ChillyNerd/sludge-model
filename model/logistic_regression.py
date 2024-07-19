from model.base_model import BaseModel
from sklearn.linear_model import LogisticRegression


class LogReg(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = "LogisticRegression"
        self.model = LogisticRegression(*args, **kwargs)
