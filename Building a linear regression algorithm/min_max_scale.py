import numpy as np
from algorithm import Algorithm

class MinMaxScale(Algorithm):
    """
    Add Documentation HERE!
    """

    def __init__(self, algo_name: str):
        """
        Add Documentation HERE!
        """

        super().__init__(algo_name)

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Add Documentation HERE!
        """
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data


if __name__ == "__main__":
    x = MinMaxScale("MinMaxScale")
    test_min_max_scaler_array = np.arange(20).reshape(4, 5)
    print(x.run(test_min_max_scaler_array))
