import numpy as np


def confusion_matrix(self, y_predicted, y_true):
    # TODO implement the confusion_matrix method
    """
    This method should return the confusion matrix as a numpy array,
    in the following format:
        confusion_matrix[0][0] = True Positive
        confusion_matrix[0][1] = False Positive
        confusion_matrix[1][0] = False Negative
        confusion_matrix[1][1] = True Negative
    """
    matrix = np.zeros((2, 2))

    matrix[0][0] = 0  # Replace this code with your calculation for True-Positive
    matrix[0][1] = 0  # Replace this code with your calculation for False-Positive
    matrix[1][0] = 0  # Replace this code with your calculation for False-Negative
    matrix[1][1] = 0  # Replace this code with your calculation for True-Negative

    return matrix

N = 8

X = np.random.choice(a=[False, True], size=(N, 1))
Y = np.random.choice(a=[False, True], size=(N, 1))
print(X.T)
print()
X = np.asarray(X)
print(X.T)

