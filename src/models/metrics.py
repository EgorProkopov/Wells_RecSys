from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from itertools import combinations


class WellSimMetric:
    """
    Рассчитывает похожесть двух скважин.

    Мы называем скважины похожими, если при построении линейных регрессий
    они ведут себя на интервалах одинаково. Под "одинаково" я имею в виду,
    что забойное давление при одиныковых условиях у них не отличается более,
    чем на 10%.
    """

    def __init__(self,
                 X1_train: np.ndarray,
                 X2_train: np.ndarray,
                 y1_train: np.ndarray,
                 y2_train: np.ndarray):
        """Initialization and train linear regression.

        Args:
            X1_train (np.ndarray): фичи для скважины 1
            X2_train (np.ndarray): для 2
            y1_train (np.ndarray): давление на забое для скважины 1
            y2_train (np.ndarray): для 2
        """
        self.X1_train = X1_train
        self.X2_train = X2_train
        self.y1_train = y1_train
        self.y2_train = y2_train

        # Fit linear regression models
        self.model1 = LinearRegression().fit(X1_train, y1_train)
        self.model2 = LinearRegression().fit(X2_train, y2_train)

    def get_error(self, y1_pred: np.ndarray, y2_pred: np.ndarray) -> np.ndarray:
        """Подсчитывает ошибку между двумя предсказаниями

        Args:
            y1_pred (np.ndarray): предсказанное давление на забое для скважины 1
            y2_pred (np.ndarray): для 2
        Returns:
            (np.ndarray): 0 или 1 - похожи скважины или нет
        """
        return np.abs(y1_pred - y2_pred) / ((y1_pred + y2_pred) / 2)

    def get_sim(self):
        X = np.concatenate([self.X1_train, self.X2_train])

        # Predict target variable for both wells
        y1_pred = self.model1.predict(X)
        y2_pred = self.model2.predict(X)

        # Calculate similarity
        similarity = (self.get_error(y1_pred, y2_pred) <= 0.1).astype(int)

        return similarity
