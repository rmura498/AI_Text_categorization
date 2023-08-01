import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from performances import performances


def classify(X_train, y_train, X_test, y_test):
    labels_conversion_dict = {'earn': 0, 'acq': 1, 'money-fx': 2, 'crude': 3, 'grain': 4}

    y_train = [labels_conversion_dict[text] for text in y_train]
    y_test = [labels_conversion_dict[text] for text in y_test]

    clf1 = SVC()
    clf2 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    clf = OneVsRestClassifier(clf2, verbose=51).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('ytest', y_test[0:10])
    print('ypred', y_pred[0:10])
    performances(y_test, y_pred)


