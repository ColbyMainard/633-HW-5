# adaboost imports
from sklearn.ensemble import AdaBoostClassifier
import numpy as np


def adaboost_classifier(X, y, params):
    clf = AdaBoostClassifier(params)
    clf.fit(X,y)
    return clf 

def adaboost_score(clf, X, y):
    return clf.score(X,y)