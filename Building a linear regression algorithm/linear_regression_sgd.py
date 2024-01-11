from typing import Optional, List
import numpy as np

from linear_regression import LinearRegression


class LinearRegressionSGD(LinearRegression):
    """
    Linear Regression Model with using (Stochastic Gradient Descent)
    """

    def __init__(self, regularization_lambda: float = 1, sgd_num_of_iterations: int = 1000, learning_rate: float = 0.001):
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
            def foo(one=True):
                if one:
                    # calculate new thetas using SGD -
                    # θ := θ - a/n [(Xθ - y).T @ X + λθ]
                    mul = self._learning_rate / n  # a/n
                    d_normal = self._regularization_lambda * self.thetas.T  # λθ
                    prediction = train_data @ self.thetas  # (Xθ - y)
                    D = ((prediction - train_target).T @ train_data + d_normal)  # (Xθ - y).T @ X
                    self.thetas -= D.T * mul  # θ := θ - a/n [(Xθ - y)X.T + λ @ θ.T]
                    return self.thetas
                else:
                    # θ = [ X.T @ X + λ[1]] ^ -1 @ X.T @ y
                    I = np.identity(m)
                    I[0, 0] = 0
                    new = train_data.T @ train_data + self._regularization_lambda * I
                    self.thetas = np.linalg.inv(new) @ train_data.T @ train_target
                    print("1")
                    return self.thetas

            a = True
            self.thetas = foo(one=True) if a else foo(one=False)
            A = 1/n * (train_target - train_data @ self.thetas)
            loss = (A.T @ A)
            print(loss[0,0])





if __name__ == "__main__":
    pass
