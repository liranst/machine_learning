import abc
from typing import Optional

import numpy as np


class Algorithm(abc.ABC):
    """
    Add Documentation HERE!
    """

    def __init__(self, regularization_lambda: float, learning_rate: float):
        """
        :param algo_name: The name of the algorithm.
        """
        self._regularization_lambda = regularization_lambda

        self.thetas: Optional[np.ndarray] = None

        self._learning_rate = learning_rate

    @abc.abstractmethod
    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Runs the algorithm on the input data.

        :param data: the input data.
        :return: the output of the algorithm run on the input data (as a numpy array)
        """
        pass
