import numpy as np
import time
from MinMaxScale_and_StandardizationScale.data_loader import DataLoader
from MinMaxScale_and_StandardizationScale.standardization_scale import StandardizationScale

# set the random seed so results can be reproduced
np.random.seed(16838)

####################################
# Part 1 - Load and Prepare the Data
####################################

# Load and Prepare the Data (Replace the 4 lines below with actual code)
from linear_regression_pinv import LinearRegressionPINV
from linear_regression_sgd import LinearRegressionSGD
from linear_regression import LinearRegression

# The 4 lines below are WRONG - they are just a placeholder - replace them with actual code
data_loader = DataLoader(train_file_path="train.npy", test_file_path="test.npy")

train_data = data_loader.get_train_data()
train_target = data_loader.get_train_target()
test_data = data_loader.get_test_data()
test_target = data_loader.get_test_target()

# Standardization activation
StrScale = StandardizationScale("Standard")
train_data = StrScale.run(train_data)
train_target = StrScale.run(train_target)
test_data = StrScale.run(test_data)
test_target = StrScale.run(test_target)

# Added feature
def added_feature(data, colum=-1):
    """
    :param data: We will take the train_data where we want to add a feature
    :param colum: Choosing the feature we want to square it,
     choosing the information is the last feature
    :return: With a new column of one of the features in the square
    """
    new_feature = np.square([data[:, colum]])
    data = np.concatenate((data, new_feature.T), axis=1)
    return data

train_data = added_feature(train_data)
test_data = added_feature(test_data)

###################################
# Part 2 - Linear Regression - MSE
###################################
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([7, 3, 4, 5, 1])
mse = LinearRegression.calculate_mse(predicted=arr1, target=arr2)
print(f"The mse in Part 2 is {mse}\n"
      "###################################\n"
      "# Part 3 - Linear Regression - SGD\n"
      "###################################")

sgd_algo = LinearRegressionSGD()
sgd_algo.fit(train_data=train_data, train_target=train_target)
print(f"The thetas in Part 3 are {sgd_algo.thetas}\n"
      "###################################\n"
      "# Part 4 - Linear Regression - PINV\n"
      "###################################")

pinv_algo = LinearRegressionPINV()
pinv_algo.fit(train_data=train_data, train_target=train_target)
print(f"The thetas in Part 4 are: \n {pinv_algo.thetas}\n"
      "######################################\n"
      "# Part 5 - Linear Regression - Predict\n"
      "#####################################")

predicted_target_sgd = sgd_algo.predict(test_data=test_data)
mse_sgd = sgd_algo.calculate_mse(predicted=predicted_target_sgd, target=test_target)
print(f"The MSE in part 5 for SGD is {mse_sgd}")

predicted_target_pinv = pinv_algo.predict(test_data=test_data)
mse_pinv = pinv_algo.calculate_mse(predicted=predicted_target_pinv, target=test_target)
print(f"The MSE in part 5 for PINV is {mse_pinv}\n"
      "#############################################################\n"
      "# Part 6 - Linear Regression - Compare Different Parameters\n"
      "#############################################################")

def foo(lambdas,number_of_iterations,train_data,train_target,alg="SGD"):

    if alg == "PINV":
        number_of_iterations = [1]
    print(f"Part 6 - {alg} Parameters")
    for regularization_lambda in lambdas:
        for iters in number_of_iterations:
            # Create algorithm instance
            algo = LinearRegressionSGD(
                regularization_lambda=regularization_lambda,
                sgd_num_of_iterations=iters) if alg == "SGD" else\
                LinearRegressionPINV(regularization_lambda=regularization_lambda)
            # Fit!
            algo.fit(train_data=train_data, train_target=train_target)

            # Predict
            predicted = algo.predict(test_data=test_data)

            # Calculate MSE
            mse = algo.calculate_mse(predicted=predicted, target=test_target)

            # Print Results!
            print(f"{alg}: lambda={regularization_lambda}, number_of_iterations={iters}, mse={mse:.4f}")

lambdas = [0, 0.1, 1, 100]
number_of_iterations = [10, 50, 100, 200]
foo(lambdas, number_of_iterations, train_data, train_target, alg="SGD")
foo(lambdas, number_of_iterations, train_data, train_target, alg="PINV")

