"""
Machine Learning
"""
###################################
# Some Imports
###################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hw2_classifier import Hw2Classifier

###################################
# General Guidelines
###################################
"""
In this Homework assignment we will use sklearn to build a classifier.

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
In this assignment we will work on (a modified version of a subset) the "Wine Quality" dataset.  

There are several versions of this dataset, for example you can find the full dataset from kaggle here:   
https://www.kaggle.com/datasets/rajyellow46/wine-quality

The full dataset:  
- The full dataset contains a train and a test set for red and white wines.  
- The dataset contains 11 features, and a target value which is the quality of the wine.  
- The quality of the wine is represented as an integer from 0 to 10.  

The assignment's dataset:
- The dataset in this HW, is a subset of the train dataset, only for white wines.
- The target feature was changed to be binary - is the wine of high quality or not - all wines with quality
  above 6 are considered high quality wines.

The data for this assignment is given to you as a csv file - `hw2_winequality.csv`.

You can load it into a pandas dataframe using:  
    `data = pd.read_csv("./hw2_winequality.csv")`
"""

###################################
# The Task
###################################
r"""
---
The Task
---
In this assignment you will build a classifier that will determine if the wine is of good quality or not.

You will build an "sklearn-compatible" class which represents an ensemble of classifying algorithms - 
Decision Trees, Logistic Regression and SVM.

The class will fit each of these algorithms on the dataset.
To predict if a wine is of high quality or not, the class will give each one of the models a weight, and 
based on the weights and the prediction of each model, will decide if the wine is of high quality or not.

You will have to split the dataset you are given with to train and test.

---
Assessment
---
To check your code, we kept some of the data aside as test.
We will run your classifier class on the this test data, and some of your grade will depend on how well your classifier
did on this unseen test data.
"""

###################################
# Part 0 - Student(s) Details
###################################
r"""
Print the Students details (replace <> with the real details):
"""
my_id = "313450264"
my_name = "liran shem tov"
print(f"Student 1 id: {my_id}")

print(f"Student 1 Name: {my_name}")

###################################
# Part 1 - Load the dataset
###################################
# This part is already implemented for you!
data = pd.read_csv("hw2_winequality.csv")

###################################
# Part 2 - Read the dataset into X and Y numpy arrays
###################################
features_columns = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]
target_column = ["is_high_quality"]

# Read the X and Y, and convert to numpy array
X = data[features_columns].to_numpy(copy=True)
Y = data[target_column].to_numpy(copy=True)

###################################
# Part 3 - Split the data into train and test
###################################
"""
Split the data into train and test.

Use `train_test_split` method of sklearn.
 - You can choose which "test_size" to use (typically should be a float from 0.05 to 0.25)
 - Make sure to pass the "random_state" argument (you can choose the first 3-4 digits of your id)
"""
# TODO - Fill the missing code instead of the "None"

# Hint: you should use:
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=<test_size>, random_state=123)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=int(my_id[-3:]))


# for the classifiers - need to reshape the target data to be 1d array
Y_train = Y_train.reshape(-1)
Y_test = Y_test.reshape(-1)

###################################
# Part 4 - Scale your data
###################################
# This part is already implemented for you!
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

###################################
# Part 5 - HW2Classifier - init
###################################
"""
Open the Hw2Classifier class in hw2_classifier.py

Finish the __init__ constructor implementation.
(See the TODO comments)
"""
# TODO - Part5

###################################
# Part 6 - HW2Classifier - fit
###################################
"""
Open the HW2Classifier class in hw2_classifier.py

implement the fit method.
(See the TODO comments)
"""
# TODO - Part6

###################################
# Part 7 - HW2Classifier - predict
###################################
"""
Open the HW2Classifier class in hw2_classifier.py

Implement the predict method.
(See the TODO comments)

---
Hint - How to use the weights
---
It is up to you how to implement the use of the models weights, but here is an idea
(you should use this idea...)

- The weights should sum to 1: 
      (w_dt, w_lor, w_svm)
- Use the weights to determine the "contribution" of each model prediction
      Assume "True" is 1, and "False" is 0.
      Then:
      prediction = (w_dt * dt_model_prediction) + (w_lor * lor_model_prediction) + (w_svm * svm_model_prediction)
      prediction = prediction > 0.5
  In other words - 
  If the weighted average of the prediction is above 0.5 - classify as True (1) otherwise classify as False (0).

"""
# TODO - Part7


###################################
# Part 8 - HW2Classifier - confusion matrix
###################################
"""
Open the HW2Classifier class in hw2_classifier.py

Implement the confusion_matrix method.
The confusion_matrix method should a numpy array representing the confusion matrix, in the following format:
    confusion_matrix[0][0] = True Positive
    confusion_matrix[0][1] = False Positive
    confusion_matrix[1][0] = False Negative
    confusion_matrix[1][1] = True Negative
"""
# TODO - Part8

###################################
# Part 9 - HW2Classifier - Precision
###################################
"""
Open the HW2Classifier class in hw2_classifier.py

Implement the precision method.
This method should calculate the precision.
Note - you can (and should) use the confusion_matrix method from part 8.
"""
# TODO - Part9

###################################
# Part 10 - HW2Classifier - Recall
###################################
"""
Open the HW2Classifier class in hw2_classifier.py

Implement the recall method.
This method should calculate the recall.
Note - you can (and should) use the confusion_matrix method from part 8.
"""
# TODO - Part10

###################################
# Part 11 - HW2Classifier - accuracy
###################################
"""
Open the HW2Classifier class in hw2_classifier.py

Implement the accuracy method.
This method should calculate the accuracy.
Note - you can (and should) use the confusion_matrix method from part 8.
"""
# TODO - Part11

###################################
# Part 12 - HW2Classifier - F1-Score
###################################
"""
Open the HW2Classifier class in hw2_classifier.py

Implement the f1_score method.
This method should calculate the f1 score.
Note - you can (and should) use the precision and recall from parts 9 and 10.
"""
# TODO - Part12

###################################
# Part 13 - HW2Classifier - score
###################################
"""
This part is already implemented for you!

Open the HW2Classifier class in hw2_classifier.py
Implement the score method.
This method should return the f1_score of the input data.
"""
# This part is already implemented for you!

###################################
# Part 14 - HW2Classifier - set models weights
###################################
"""
Open the HW2Classifier class in hw2_classifier.py

In the __init__ method,
Change the default models_weights tuple to get the best possible f1_score.
Best f1_score here - best f1_score for unseen data.

You can use the next part to run and test your classifier.
"""
# TODO Part14

###################################
# Part 15 - run your new classifier!
###################################
# This part is already implemented for you!

# create the classifier
hw2_classifier = Hw2Classifier()

# fit the train data
hw2_classifier = hw2_classifier.fit(X_train, Y_train)

# predict on train data
y_train_predicted = hw2_classifier.predict(X_train)
# predict on test data
y_test_predicted = hw2_classifier.predict(X_test)

# print the train stats
print("-" * 20)
print("Train Confusion Matrix:")
print(hw2_classifier.confusion_matrix(y_predicted=y_train_predicted, y_true=Y_train))
print(f"Train recall: {hw2_classifier.recall(y_predicted=y_train_predicted, y_true=Y_train):.5f}")
print(f"Train precision: {hw2_classifier.precision(y_predicted=y_train_predicted, y_true=Y_train):.5f}")
print(f"Train accuracy: {hw2_classifier.accuracy(y_predicted=y_train_predicted, y_true=Y_train):.5f}")
print(f"Train f1_score: {hw2_classifier.f1_score(y_predicted=y_train_predicted, y_true=Y_train):.5f}")
print(f"HW2Classifier train score: {hw2_classifier.score(X_train, Y_train):.5f}")

# print the test stats
print("-" * 20)
print("Test Confusion Matrix:")
print(hw2_classifier.confusion_matrix(y_predicted=y_test_predicted, y_true=Y_test))
print(f"Test recall: {hw2_classifier.recall(y_predicted=y_test_predicted, y_true=Y_test):.5f}")
print(f"Test precision: {hw2_classifier.precision(y_predicted=y_test_predicted, y_true=Y_test):.5f}")
print(f"Test accuracy: {hw2_classifier.accuracy(y_predicted=y_test_predicted, y_true=Y_test):.5f}")
print(f"Test f1_score: {hw2_classifier.f1_score(y_predicted=y_test_predicted, y_true=Y_test):.5f}")
print(f"HW2Classifier test score: {hw2_classifier.score(X_test, Y_test):.5f}")

