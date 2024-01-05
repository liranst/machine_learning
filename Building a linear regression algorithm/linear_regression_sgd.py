from typing import Optional, List
import numpy as np

from linear_regression import LinearRegression


class LinearRegressionSGD(LinearRegression):
    """
    Linear Regression Model with using (Stochastic Gradient Descent)
    """

    def __init__(self, regularization_lambda: float = 1, sgd_num_of_iterations: int = 100):
        """

        :param regularization_lambda: The lambda parameter to use for the regularization
        :param sgd_num_of_iterations: Number of iterations for the SGD algorithm
        """
        super().__init__(regularization_lambda=regularization_lambda)

        self._sgd_num_of_iterations = sgd_num_of_iterations
        self.regularization_lambda = regularization_lambda

    def fit(self, train_data: np.ndarray, train_target: np.ndarray) -> None:
        """
        Fits a linear regression with L2 regularization model for some input train_data.

        :param train_data: The data to fit the model to.
        :param train_target: The data ground truth.

        """
        # Add bias column (1's column) to the data
        X = LinearRegression._add_bias_column(arr=train_data)
        y = train_target
        alpha = 0.01

        # get dimensions
        m = train_data.shape[0]  # number of samples
        n = train_data.shape[1]  # number of features (including bias)

        # randomize initial thetas
        W = np.random.random(size=n)
        cost_history_list = []

        # Perform iterations of SGD
        for iteration in range(self._sgd_num_of_iterations):
            # calculate new thetas using SGD
            y_estimated = X @ W
            error = y_estimated - y

            # regularization term
            ridge_reg_term = (self.regularization_lambda / 2 * m) * np.sum(np.square(W))

            # calculate the cost (MSE) + regularization term
            cost = (1 / 2 * m) * np.sum(error ** 2) + ridge_reg_term

            # Update our gradient by the dot product between
            # the transpose of 'X' and our error + lambda value * W
            # divided by the total number of samples
            gradient = (1 / m) * (X.T.dot(error) + (self.regularization_lambda * W))

            # Now we have to update our weights
            W = W - alpha * gradient

            # Let's print out the cost to see how these values
            # changes after every iteration
            print(f"cost:{cost} \t iteration: {iteration}")

            # keep track the cost as it changes in each iteration
            cost_history_list.append(cost)

        return W