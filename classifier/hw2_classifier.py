from typing import Tuple
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
class Hw2Classifier:

    def __init__(self, models_weights: Tuple[float, float, float] =  (0.5, 0.2, 0.75)):
        self._models_weights = np.asarray(models_weights)

        self._dt_model = DecisionTreeClassifier(
            criterion="entropy",
            splitter="best",
            max_depth=5,
            min_samples_split=2,
        )

        self._lor_model = LogisticRegression(C=100, class_weight="balanced")
        self._svm_model = SVC(C=100, class_weight="balanced", tol=0.01)

    def fit(self, X, y):
        self._dt_model.fit(X, y)
        self._lor_model.fit(X, y)
        self._svm_model.fit(X, y)
        return self

    def predict(self, X):
        predict_dt3 = self._dt_model.predict(X) * self._models_weights[0]
        predict_LR = self._lor_model.predict(X) * self._models_weights[1]
        predict_svm = self._svm_model.predict(X) * self._models_weights[2]
        prediction = predict_dt3 + predict_LR + predict_svm
        y_prediction = prediction > 0.5
        return y_prediction

    def score(self, X, y):
        y_predicted = self.predict(X)
        f1_score = self.f1_score(y_predicted=y_predicted, y_true=y)
        return f1_score

    def fit_predict(self, X, y):

        return self.fit(X, y).predict(y)

    def confusion_matrix(self, y_predicted, y_true):
        """
        This method should return the confusion matrix as a numpy array,
        in the following format:
            confusion_matrix[0][0] = True Positive
            confusion_matrix[0][1] = False Positive
            confusion_matrix[1][0] = False Negative
            confusion_matrix[1][1] = True Negative
        """
        matrix = np.zeros((2, 2))

        matrix[0][0] = np.logical_and(y_predicted == True, y_true == True).sum()
        matrix[0][1] = np.logical_and(y_predicted == True, y_true == False).sum()
        matrix[1][0] = np.logical_and(y_predicted == False, y_true == True).sum()
        matrix[1][1] = np.logical_and(y_predicted == False, y_true == False).sum()
        return matrix

    def precision(self, y_predicted, y_true):
        matrix = self.confusion_matrix(y_predicted, y_true)
        the_precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        return the_precision

    def recall(self, y_predicted, y_true):
        matrix = self.confusion_matrix(y_predicted, y_true)
        the_recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])
        return the_recall

    def accuracy(self, y_predicted, y_true):
        matrix = self.confusion_matrix(y_predicted, y_true)
        the_accuracy = (matrix[0][0] + matrix[1][1]) / (matrix.sum())
        return the_accuracy

    def f1_score(self, y_predicted, y_true):
        the_f1_score = 2 * (self.precision(y_predicted, y_true) * self.recall(y_predicted, y_true)) \
                       / (self.precision(y_predicted, y_true) + self.recall(y_predicted, y_true))
        # the_f1_score = sklearn.metrics.f1_score(y_predicted, y_true)
        return the_f1_score

    def shwo_visualization(self):
        pass
