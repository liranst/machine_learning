import numpy as np

from MinMaxScale_and_StandardizationScale.data_loader import DataLoader
from MinMaxScale_and_StandardizationScale.standardization_scale import StandardizationScale

# set the random seed so results can be reproduced
np.random.seed(16838)

###################################
# Part 1 - Load and Prepare the Data
###################################

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

train_data = StandardizationScale("Standard_train_data").run(train_data)
train_target = StandardizationScale("Standard_train_target").run(train_target)
test_data = StandardizationScale("Standard_test_data").run(test_data)
test_target = StandardizationScale("Standard_test_data").run(test_target)

###################################
# OPTIONAL OPTIONAL OPTIONAL OPTIONAL
# Part 1.5 - Features Engineering - OPTIONAL
# OPTIONAL OPTIONAL OPTIONAL OPTIONAL
###################################
# Add your features here to train_data
def added_feature(data, colum=-1):
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

print(f"The mse in Part 2 is {mse}")
###################################
# Part 3 - Linear Regression - SGD
###################################
sgd_algo = LinearRegressionSGD()
sgd_algo.fit(train_data=train_data, train_target=train_target)

print(f"The thetas in Part 3 are {(np.sum(sgd_algo.thetas))}")

###################################
# Part 4 - Linear Regression - PINV
###################################
pinv_algo = LinearRegressionPINV()
pinv_algo.fit(train_data=train_data, train_target=train_target)

print(f"The thetas in Part 4 are: \n {pinv_algo.thetas}")

###################################
# Part 5 - Linear Regression - Predict
###################################
predicted_target_sgd = sgd_algo.predict(test_data=test_data)
mse_sgd = sgd_algo.calculate_mse(predicted=predicted_target_sgd, target=test_target)
print(f"The MSE in part 5 for SGD is {mse_sgd}")

predicted_target_pinv = pinv_algo.predict(test_data=test_data)
mse_pinv = pinv_algo.calculate_mse(predicted=predicted_target_pinv, target=test_target)
print(f"The MSE in part 5 for PINV is {mse_pinv}")

###################################
# Part 6 - Linear Regression - Compare Different Parameters
###################################
lambdas = [0, 0.1, 1, 100]
number_of_iterations = [10, 50, 100, 200]

print("=" * 50)
print("Part 6 - SGD Parameters")
for regularization_lambda in lambdas:
    for iters in number_of_iterations:
        # Create algorithm instance
        sgd_algo = LinearRegressionSGD(
            regularization_lambda=regularization_lambda,
            sgd_num_of_iterations=iters
        )

        # Fit!
        sgd_algo.fit(train_data=train_data, train_target=train_target)

        # Predict
        predicted = sgd_algo.predict(test_data=test_data)

        # Calculate MSE
        mse = sgd_algo.calculate_mse(predicted=predicted, target=test_target)

        # Print Results!
        print(f"SGD: lambda={regularization_lambda}, number_of_iterations={number_of_iterations}, mse={mse:.4f}")

print("=" * 50)
print("Part 6 - PINV Parameters")

for regularization_lambda in lambdas:
    # Create algorithm instance
    pinv_algo = LinearRegressionPINV(
        regularization_lambda=regularization_lambda
    )

    # Fit!
    pinv_algo.fit(train_data=train_data, train_target=train_target)

    # Predict
    predicted = pinv_algo.predict(test_data=test_data)

    # Calculate MSE
    mse = pinv_algo.calculate_mse(predicted=predicted, target=test_target)

    # Print Results!
    print(f"PINV: lambda={regularization_lambda}, mse={mse:.4f}")


