from model.base_model import BaseModel
from torch.nn import Linear, ReLU, Softmax, Sequential, BatchNorm1d, Sigmoid
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from torch import device, no_grad, argmax, tensor, float
from data.custom_dataset import CustomDataset
from torch.nn.functional import cross_entropy
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
from tqdm import tqdm


class NeuralNetwork(BaseModel):
    def __init__(self, input_shape: int, output_shape: int, *args, **kwargs):
        super().__init__()
        self.name = "NeuralNetwork"
        self.model = Sequential(
            BatchNorm1d(input_shape),
            Linear(
                input_shape, 18
            ),
            BatchNorm1d(18),
            ReLU(),
            Linear(
                18, 16
            ),
            BatchNorm1d(16),
            ReLU(),
            Linear(
                16, output_shape
            ),
            Softmax(dim=-1)
        )
        self.device = device('cpu')
        self.model.to(self.device)
        self.log = logging.getLogger("NeuralNetwork")

    def fit(self, features, target, *args, **kwargs):
        optimizer = SGD(self.model.parameters(), lr=0.05)
        train_data_loader = DataLoader(CustomDataset(features, target), batch_size=8, shuffle=False, num_workers=1)
        val_data_loader = DataLoader(CustomDataset(kwargs['x_test'], kwargs['y_test']), batch_size=8, shuffle=False, num_workers=1)
        for epoch in range(10):
            for x_train, y_train in tqdm(train_data_loader):
                y_pred = self.model(x_train.to(self.device))
                loss = cross_entropy(y_pred, y_train.to(self.device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if epoch % 2 == 0:
                val_loss = []
                val_accuracy = []
                with no_grad():
                    for x_val, y_val in tqdm(val_data_loader):
                        y_pred = self.model(x_val.to(self.device))
                        loss = cross_entropy(y_pred, y_val.to(self.device))
                        val_loss.append(loss.cpu().numpy())
                        val_accuracy.extend(
                            (argmax(y_pred, dim=-1) == y_val.to(self.device)).cpu().numpy().tolist()
                        )
                self.log.debug(f"Epoch: {epoch}, loss: {np.mean(val_loss)}, accuracy: {np.mean(val_accuracy)}")

    def predict(self, features):
        with no_grad():
            x = tensor(features.values, dtype=float).to(self.device)
            return argmax(self.model(x), dim=1).numpy()
