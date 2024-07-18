from abc import ABC, abstractmethod
from exceptions.not_implemented import NotImplementedYet


class BaseModel(ABC):
    def __init__(self):
        self.name = "BaseModel"
        self.__model = None

    @abstractmethod
    def fit(self, features, target):
        raise NotImplementedYet()

    @abstractmethod
    def predict(self, features):
        raise NotImplementedYet()
