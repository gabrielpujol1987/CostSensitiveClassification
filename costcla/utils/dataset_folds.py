"""
This module provides a function to split the training set before being
used for testing.
"""

import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split


class Bunch(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

class Info(object):
    def __init__(self):
        self.train_index = []
        self.test_index = []
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.cost_mat_train = []
        self.cost_mat_test = []

def create_all_folds(ds, numFolds, trainRatio, seed=0):
    np.random.seed(seed)
    n_samples = len(ds.data)

    main = Info()
    main.train_index, main.test_index = train_test_split(range(n_samples), train_size=trainRatio)
    main.x_train, main.x_test = ds.data[main.train_index], ds.data[main.test_index]
    main.y_train, main.y_test = ds.target[main.train_index], ds.target[main.test_index]
    main.cost_mat_train, main.cost_mat_test = ds.cost_mat[main.train_index], ds.cost_mat[main.test_index]

    n_train = len(main.x_train)
    kf = cross_validation.KFold(n=n_train, n_folds=numFolds, shuffle=False, random_state=None)
    count = 0
    folds = {}
    for train_index, test_index in kf:
        folds[count] = Info()
        folds[count].train_index = train_index
        folds[count].test_index = test_index
        folds[count].x_train, folds[count].x_test = main.x_train[train_index], main.x_train[test_index]
        folds[count].y_train, folds[count].y_test = main.y_train[train_index], main.y_train[test_index]
        folds[count].cost_mat_train, folds[count].cost_mat_test = main.cost_mat_train[train_index], main.cost_mat_train[test_index]

        count = count + 1

    return Bunch(main=main, folds=folds, feature_names=ds.feature_names, target_names=ds.target_names,
                 orig_data=ds.data, orig_target=ds.target, orig_cost_mat=ds.cost_mat)
