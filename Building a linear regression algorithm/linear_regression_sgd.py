from typing import Optional, List
import numpy as np

from linear_regression import LinearRegression


class LinearRegressionSGD(LinearRegression):
    """
    Linear Regression Model with using (Stochastic Gradient Descent)
    """

    def __init__(self, regularization_lambda: float = 1,
                 sgd_num_of_iterations: int = 5000,
                 learning_rate: float = 0.001):
        """
        :param regularization_lambda: The lambda parameter to use for the regularization
        :param sgd_num_of_iterations: Number of iterations for the SGD algorithm
        """
        self._sgd_num_of_iterations = sgd_num_of_iterations
        self._regularization_lambda = regularization_lambda
        super().__init__(regularization_lambda=regularization_lambda, learning_rate=learning_rate)

    def fit(self, train_data: np.ndarray, train_target: np.ndarray) -> None:
        """
        Fits a linear regression with L2 regularization model for some input train_data.

        :param train_data: The data to fit the model to.
        :param train_target: The data ground truth.

        """
        # Add bias column (1's column) to the data
        train_data = LinearRegression._add_bias_column(arr=train_data)

        # get dimensions
        n = train_data.shape[0]  # number of samples
        m = train_data.shape[1]  # number of features (including bias)
        # randomize initial thetas
        self.thetas = np.random.random(size=(m, 1))
        # Perform iterations of SGD
        for _ in range(self._sgd_num_of_iterations):
            # calculate new thetas using SGD -
            # θ := θ - a/n [(Xθ - y).T @ X + λθ]
            mul = self._learning_rate / n  # a/n
            d_normal = self._regularization_lambda * self.thetas.T  # λθ
            prediction = train_data @ self.thetas  # (Xθ - y)
            D = ((prediction - train_target).T @ train_data + d_normal)  # (Xθ - y).T @ X
            self.thetas -= D.T * mul  # θ := θ - a/n [(Xθ - y)X.T + λ @ θ.T]

            # A = 1/n * (train_target - train_data @ self.thetas)
            # loss = (A.T @ A)


if __name__ == "__main__":
    pass
