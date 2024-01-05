import abc
import numpy as np


class Algorithm(abc.ABC):
    """
    Add Documentation HERE!
    """

    def __init__(self, algo_name: str):
        """
        :param algo_name: The name of the algorithm.
        """
        self.algo_name = algo_name

    @abc.abstractmethod
    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Runs the algorithm on the input data.

        :param data: the input data.
        :return: the output of the algorithm run on the input data (as a numpy array)
        """
        pass
