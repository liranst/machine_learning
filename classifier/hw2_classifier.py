from typing import Tuple
from sklearn import tree
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import LogisticRegression


class Hw2Classifier:

    def __init__(self, models_weights: Tuple[float, float, float] = (0.34, 0.33, 0.33)):
        self._models_weights = np.asarray(models_weights)

        self._dt_model = tree.DecisionTreeClassifier(max_depth=3)
        self._lor_model = LogisticRegression(solver='lbfgs')
        self._svm_model = SVC(kernel='linear')

    def fit(self, X, y):
        # TODO implement the fit method (you should call the fit method of each model)
        self._dt_model.fit(X, y)
        self._lor_model.fit(X, y)
        self._svm_model.fit(X, y)
        return self

    def predict(self, X):
        # TODO implement the predict method
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
