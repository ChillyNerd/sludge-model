from model.base_model import BaseModel


class NeuralNetwork(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = "NeuralNetwork"
        self.model = None
