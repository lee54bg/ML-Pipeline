# Importing the libraries
print("Initiating ML test module\n")

import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import pickle

pickled_xtrain, pickled_ytrain = pickle.load(open("dataset/test_data/test_data.pkl", 'rb'))
pickled_xtest, pickled_ytest = pickle.load(open("dataset/train_data/train_data.pkl", 'rb'))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
pickled_xtrain = sc.fit_transform(pickled_xtrain)
pickled_xtest = sc.transform(pickled_xtest)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(pickled_xtrain, pickled_ytrain)

mlmodel_filename = 'trained_models/ml-model.pkl'
pickle.dump(classifier, open(mlmodel_filename, 'wb'))

print("Saved ml-model to " + mlmodel_filename)

# Predicting the Test set results
y_pred = classifier.predict(pickled_xtest)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pickled_ytest, y_pred)
print(cm)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = pickled_xtrain, pickled_ytrain
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = pickled_xtest, pickled_ytest
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

f = plt.figure()

plt.plot(range(10), range(10), "o")
plt.show()

f.savefig("foo.pdf", bbox_inches='tight')

# https://scikit-learn.org/stable/modules/model_evaluation.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.htmlpkl