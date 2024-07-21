from model.base_model import BaseModel
from torch.nn import Linear, ReLU, Softmax, Sequential
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from torch import device, no_grad, argmax
from data.custom_dataset import CustomDataset
from torch.nn.functional import cross_entropy
import numpy as np
import logging
from tqdm import tqdm


class NeuralNetwork(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = "NeuralNetwork"
        self.model = Sequential(
            Linear(
                27, 20
            ),
            ReLU(),
            Linear(
                20, 15
            ),
            Softmax()
        )
        self.optimizer = SGD(self.model.parameters(), lr=0.05)
        self.device = device('cpu')
        self.model.to(self.device)
        self.log = logging.getLogger("NeuralNetwork")

    def fit(self, features, target, *args, **kwargs):
        train_data_loader = DataLoader(CustomDataset(features, target), batch_size=4, shuffle=False, num_workers=1)
        val_data_loader = DataLoader(CustomDataset(kwargs['x_test'], kwargs['y_test']), batch_size=4, shuffle=False, num_workers=1)
        for epoch in range(5):
            for x_train, y_train in tqdm(train_data_loader):  # берем батч из трейн лоадера
                y_pred = self.model(x_train.to(self.device))  # делаем предсказания
                loss = cross_entropy(y_pred, y_train.to(self.device))  # считаем лосс
                loss.backward()  # считаем градиенты обратным проходом
                self.optimizer.step()  # обновляем параметры сети
                self.optimizer.zero_grad()  # обнуляем посчитанные градиенты параметров
            if epoch % 2 == 0:
                val_loss = []  # сюда будем складывать **средний по бачу** лосс
                val_accuracy = []
                with no_grad():  # на валидации запрещаем фреймворку считать градиенты по параметрам
                    for x_val, y_val in tqdm(val_data_loader):  # берем батч из валидационного лоадера
                        y_pred = self.model(x_val.to(self.device))  # делаем предсказания
                        loss = cross_entropy(y_pred, y_val.to(self.device))  # считаем лосс
                        val_loss.append(loss.cpu().numpy())  # добавляем в массив
                        val_accuracy.extend(
                            (argmax(y_pred, dim=-1) == y_val.to(self.device)).cpu().numpy().tolist()
                        )
                self.log.debug(f"Epoch: {epoch}, loss: {np.mean(val_loss)}, accuracy: {np.mean(val_accuracy)}")

    def predict(self, features):
        return self.model(features)
