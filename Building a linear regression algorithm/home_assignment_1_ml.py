"""
Home Assignment #1 - Machine Learning
"""

###################################
# General Guidelines
###################################
"""
In this Homework assignment we will use Python (and OOP in Python) to build a linear regression algorithm.
Our algorithm will try to predict a house price based on numerical features.

Guidelines:
    - After completing each part, you should be able to run the program and test your results (so far).
    - Your program MUST run (if it will not, you will lose points).
    - You can use the code from the lectures and recitations as a base line.
    - Make sure to add documentation - as we saw in class.
    - Keep a clean code.
"""

###################################
# The Data
###################################
"""
---
The Data
---
In this assignment we will work on (a subset of) the "House Prices" dataset.
You can find the full dataset here:
     https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The full dataset:
    The full dataset contains a train and a test set, each has around 1460 samples.
    Each sample represent a house, and has ~80 features to describe the various properties of the house.
    The train set has also the target - the sale price of a house.

The HW dataset:
    The dataset in this HW, is a subset of the train dataset, with only 7 features.
    It is divided into a train set and a test set. 
    The train set has 1314 samples, and the test set has 146 samples.
    Each sample contains 7 features and the target (sale price).
    Keep reading for more details.

The data for this assignment is given to you as "npy" files.

You can load those files using:
    data = np.load("path/to/file.npy")

You are given 2 npy files:
* train.npy - 
    This files contains 1314 rows and 8 columns.
* test.npy - 
    This files contains 146 rows and 8 columns.

Each row represent a sample.
The first 7 columns represent a feature of the house.
The 8-th (last) column is the house price.
"""

###################################
# Part 1 - Load and Prepare the Data
###################################
"""
---
Part 1 - Load and Prepare the Data
---
In part 1, we will load and prepare the data.

IMPORTANT NOTE!!!
    If you did homework assignment #1 in "Advanced-Python" course -
    you can (and should) use the same code from there!
    (If you do, make sure to copy the classes you used into this assignment)

Tasks:
In hw1_ml_main.py: 
    1. Load the assignment's data files: train.npy and test.npy.
    
    2. From each of the loaded arrays - create 2 arrays - one for the features (first 7 columns) 
       and one for the target (last column).
       You should end up with 4 arrays:
            train_data, train_target, test_data, test_target
    
    3. Scale the features arrays using standardization scaling.
"""

###################################
# OPTIONAL OPTIONAL OPTIONAL OPTIONAL
# Part 1.5 - Features Engineering - OPTIONAL
# OPTIONAL OPTIONAL OPTIONAL OPTIONAL
###################################
"""
---
Part 1.5 - Features Engineering - OPTIONAL
---
This part is OPTIONAL!!

In part 1.5, you can add your own features to the dataset.

For example, as you saw in the lectures - you can add a feature which is the 
multiplication of 2 other features, or the square of a feature.

Tasks:
    1. THINK about the features you want to add!
    
    2. Add the features to train_data dataset (it is a numpy array).
    
    3. Add the features to test_data dataset (it is a numpy array).
        
    IMPORTANT NOTE:
        You MUST add the features both to train_data and test_data. 
        You CAN'T add them to only one of the arrays!!! 
"""

###################################
# Part 2 - Linear Regression - MSE
###################################
"""
---
Part 2 - Linear Regression - MSE
---
In part 2, we will implement Mean Square Error calculation.

Tasks:
    1. In linear_regression.py class, Implement code missing in the static method "calculate_mse".
    
    2. In hw1_ml_main.py - 
       Check that your MSE method works correctly with 2 arrays:
        [1,2,3,4,5], [7,3,4,5,1]
"""

###################################
# Part 3 - Linear Regression SGD
###################################
"""
---
Part 3 - Linear Regression - SGD
---
In part 3, we will implement the Linear Regression with SGD (Stochastic Gradient Descent) algorithm.

Tasks:
    1. In linear_regression_sgd.py, implement the missing code in the "fit" method.
       Your implementation should use the L2 regularization (LASSO).
    
       Notes and Hints:
        - You do not need to use any additional for loops.
        - The code missing here is relatively short (less than 20 lines).
        - Before you start coding - write the SGD thetas update formula. 
          Make sure you understand it - and only then start coding.
        - Don't forget about the regularization L2 Lambda!!
        - Don't forget to add bias column - this code is already implemented for you - you should take a look at it!
    
    2. In hw1_ml_main.py:
        Check your "fit" method - Create a LinearRegressionSGD object, and fit the train_data to it.
        Print the thetas.
        (The code for this task is already written)  
"""

###################################
# Part 4 - Linear Regression PINV
###################################
"""
---
Part 4 - Linear Regression - PINV
---
In part 4, we will implement the Linear Regression with PINV (Pseudo Inverse) algorithm.

Tasks:
    1. In linear_regression_pinv.py, implement the missing code in the "fit" method.
       Your implementation should use the L2 regularization (LASSO).

       Notes and Hints:
        - You do not need to use any additional for loops.
        - The code missing here is relatively short (less than 20 lines).
        - Before you start coding - write the PINV thetas formula. 
          Make sure you understand it - and only then start coding.
        - Don't forget about the regularization L2 Lambda!!

    2. In hw1_ml_main.py:
        Check your "fit" method - Create a LinearRegressionPINV object, and fit the train_data to it.
        Print the thetas.
        (The code for this task is already written)  
"""

###################################
# Part 6 - Linear Regression - Predict
###################################
"""
---
Part 6 - Linear Regression - Predict
---
In part 6, we will implement the Linear Regression prediction using the thetas calculated in the fit method.

Tasks:
    1. In linear_regression.py class, Implement code missing in the static method "predict".
       
       Notes and Hints:
        - The code missing here is relatively short (less than 20 lines).
        - Before you start coding - write the prediction formula. 
          Make sure you understand it - and only then start coding.
    
    2. In hw1_ml_main.py:
        Check your "predict" method - call the predict method with test_data for the objects from parts 3 and 4.
        Print the MSE's for the predicted results.
        (The code for this task is already written). 
"""

###################################
# Part 6 - Compare Different Parameters
###################################
"""
---
Part 6 - Linear Regression - Compare Different Parameters
---

In part 6, we will check the effect of lambdas and number of iterations on the SGD algorithm MSE, 
and we will compare that with the PINV MSE.

Tasks:
    1. In hw1_ml_main.py:
        Fit the Linear Regression model with different parameters and compare the results.
        (The code for this task is already written).
        This task is already written for you, just run the code!
"""