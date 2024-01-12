import numpy as np
from linear_regression import LinearRegression


class LinearRegressionPINV(LinearRegression):
    """
    Linear Regression Model using PINV (Pseudo Inverse)
    """

    def __init__(self, regularization_lambda: float = 1, sgd_num_of_iterations: int = 5000,
                 learning_rate: float = 0.001):

        super().__init__(regularization_lambda, learning_rate)
        self._sgd_num_of_iterations = sgd_num_of_iterations
        self._regularization_lambda = regularization_lambda
        """
        :param regularization_lambda: The lambda parameter to use for the regularization
        """


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

        # θ = [ X.T @ X + λ[1]] ^ -1 @ X.T @ y
        I = np.identity(m)
        I[0, 0] = 0
        new = train_data.T @ train_data + self._regularization_lambda * I
        self.thetas = np.linalg.inv(new) @ train_data.T @ train_target
        # A = 1 / n * (train_target - train_data @ self.thetas)
        # loss = (A.T @ A)
