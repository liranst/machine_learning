import numpy as np
from algorithm import Algorithm

class StandardizationScale(Algorithm):
    """

    """
    def __init__(self, algo_name: str, regularization_lambda: float, learning_rate: float):
        """

        :param algo_name:
        """
        super().__init__(regularization_lambda, learning_rate)

    def run(self, data: np.ndarray) -> np.ndarray:
        """

        :param data:
        :return:
        """
        data = (data - np.average(data))/np.std(data)
        return data



if __name__ == "__main__":
    test_std_scaler_array = np.arange(24).reshape(6, 4)
    x = StandardizationScale("StandardizationScale").run(test_std_scaler_array)
    print(x)
