import numpy as np
from MinMaxScale_and_StandardizationScale.algorithm import Algorithm

class StandardizationScale(Algorithm):
    """

    """
    def __init__(self, algo_name: str):
        """

        :param algo_name:
        """
        super().__init__(algo_name)

    def run(self, data: np.ndarray) -> np.ndarray:
        """

        :param data:
        :return:
        """
        data = (data - np.average(data))/np.std(data)
        return data



if __name__ == "__main__":
    test_std_scaler_array = np.arange(24).reshape(6, 4)
    x = StandardizationScale("StandardizationScale")
    print(Algorithm.run(test_std_scaler_array))
