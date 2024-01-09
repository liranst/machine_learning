import numpy as np

from linear_regression import LinearRegression


class LinearRegressionPINV(LinearRegression):
    """
    Linear Regression Model using PINV (Pseudo Inverse)
    """

    def __init__(self, regularization_lambda: float = 1):
        """
        :param regularization_lambda: The lambda parameter to use for the regularization
        """
        super().__init__(regularization_lambda=regularization_lambda)

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

        # calculate thetas using PINV
        """
        Your code goes here
        """

        X = train_data
        y = train_target
        new_thetas = np.linalg.lstsq(X, y)
        # self.thetas = y.T * X * np.linalg.inv(X.T * X)
        self.thetas = np.array([new_thetas])  # replace this line with actual code!!
        # Don't forget to use regularization_lambda!!

        # Assign the new thetas
        self.thetas = new_thetas