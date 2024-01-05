import numpy as np


class DataLoader:
    """
    Add Documentation HERE!
    """

    def __init__(self, train_file_path: str, test_file_path: str):
        """
        :param train_file_path: A path to a train data file
        :param test_file_path: A path to a test data file
        """
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path


    def get_train_data(self) -> np.ndarray:
        """
        :return: Returns a numpy array with the train data.
        """
        get_train_data = np.load(self.train_file_path)
        return get_train_data[:, :-1]


    def get_train_target(self) -> np.ndarray:
        """
        :return: Returns a numpy array with the train target.
        """
        get_train_target = np.load(self.train_file_path)
        return get_train_target[:,-1:]

    def get_test_data(self) -> np.ndarray:
        """
        :return: Returns a numpy array with the test data.
        """

        get_test_data = np.load(self.test_file_path)
        return get_test_data[:,:-1]

    def get_test_target(self) -> np.ndarray:
        """
        :return: Returns a numpy array with the test target.
        """
        get_test_target = np.load(self.test_file_path)
        return get_test_target[:,-1:]



if __name__ == "__main__":
    """    See the following example to run DataLoader  """
    data_loader = DataLoader(train_file_path="train.npy", test_file_path="test.npy")

    train_data = data_loader.get_train_data()
    train_target = data_loader.get_train_target()
    test_data = data_loader.get_test_data()
    test_target = data_loader.get_test_target()

    print("train:")
    print(f"get_train_target shape = {train_target.shape}")
    print(f"get_train_data = {train_data.shape}")

    print("\ntest:")
    print(f"get_test_datashape = {test_data.shape}")
    print(f"get_test_target shape = {test_target.shape}")
