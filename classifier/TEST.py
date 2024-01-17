from typing import Tuple
from sklearn import tree
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

N = 80

X = np.random.choice(a=[False,True], size=(N, 1))
Y = np.random.choice(a=[False,True], size=(N, 1))

sns.scatterplot(x=X[:,0], y=Y[:,0])
plt.show()
print(X[:,0])
print(Y[0])

