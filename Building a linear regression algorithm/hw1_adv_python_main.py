from data_loader import DataLoader
from min_max_scale import MinMaxScale
from standardization_scale import StandardizationScale

###################################
#
###################################
data_loader = DataLoader(train_file_path="train.npy", test_file_path="test.npy")

train_data = data_loader.get_train_data()
train_target = data_loader.get_train_target()
test_data = data_loader.get_test_data()
test_target = data_loader.get_test_target()

###################################
#
###################################

def scaled(data):
    """

    :param data:
    :return:
    """
    MinMax_data = MinMaxScale("MinMaxScale").run(data)
    Stand_data = StandardizationScale("StandardizationScale").run(data)
    list_data_scaled = [MinMax_data, Stand_data]
    return list_data_scaled


train_data_scaled = scaled(train_data)
test_data_scaled = scaled(test_data)
