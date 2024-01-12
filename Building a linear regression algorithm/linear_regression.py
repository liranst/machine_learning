import abc
import numpy as np
from typing import Optional


class LinearRegression(abc.ABC):
    """
    Linear Regression Model
    """

    def __init__(self, regularization_lambda: float, learning_rate: float):
        """
        :param regularization_lambda: The lambda parameter to use for the regularization
        """
        self._regularization_lambda = regularization_lambda
        self.thetas: Optional[np.ndarray] = None
        self._learning_rate = learning_rate

    @abc.abstractmethod
    def fit(self, train_data: np.ndarray, train_target: np.ndarray) -> None:
        """
        Fits a linear regression model for some input train_data.

        :param train_data: The data to fit the model to.
        :param train_target: The data ground truth.
        """
        pass

    def predict(self, test_data: np.ndarray,) -> np.ndarray:
        """
        Use the thetas which were calculated in the fit method to predict the target for the input data.

        :param test_data: The data to predict the target for.
        :return: The prediction results as a np-array
        """
        # Predict the target using self.thetas and the input test_data

        # The line below is WRONG - it is just a placeholder - replace it with actual code
        # Add bias column (1's column) to the data
        test_data = LinearRegression._add_bias_column(arr=test_data)
        predicted_target = test_data @ self.thetas    # replace this line with actual code!!

        return predicted_target

    @staticmethod
    def calculate_mse(predicted: np.ndarray, target: np.ndarray) -> float:
        """
        Returns the MSE (Mean Square Error) between the prediction on the test_data against the ground truth
        test_target.

        :param predicted: the predicted target for the data
        :param target: the real ground truth target for the data
        :return: the mean square error (MSE) of predicted_target and test_target
        """

        # return MSE
        return np.sum(np.square(predicted - target)) / target.shape[0]

    @staticmethod
    def _add_bias_column(arr: np.ndarray) -> np.ndarray:
        """
        Adds a bias column (an ones-column) to the array.

        :param arr: The input array
        :return: output array - same as the input array, but with an ones-column.
        """
        ones_col = np.ones(arr.shape[0])
        new_arr = np.column_stack((ones_col, arr))

        return new_arr