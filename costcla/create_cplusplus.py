"""
This module creates the necessary dataset files to run the C++ program.
"""

import csv
import numpy as np


def create_c_files(ds, filename, output_folder):
    orig_data, header = _create_matrix(ds.orig_data, ds.orig_target, ds.orig_cost_mat, ds.feature_names, ds.target_names)
    _write_csv(filename + '_ORIGINAL', output_folder, orig_data, header)
    train_data, header = _create_matrix(ds.main.x_train, ds.main.y_train, ds.main.cost_mat_train, ds.feature_names, ds.target_names)
    train_filename = filename + '_TRAIN'
    _write_csv(train_filename, output_folder, train_data, header)
    test_data, header = _create_matrix(ds.main.x_test, ds.main.y_test, ds.main.cost_mat_test, ds.feature_names, ds.target_names)
    _write_csv(filename + '_TEST', output_folder, test_data, header)

    for key, fold in ds.folds.items():
        _write_fold_index(train_filename + '_train', output_folder, key, fold.train_index)
        _write_fold_index(train_filename + '_val', output_folder, key, fold.test_index)


def _write_fold_index(filename, output_folder, key, indexes):
    with open(output_folder + filename + '_' + str(key) + '.txt', 'w') as file:
        file.write(str(len(indexes)) + '\n')
        for inx in indexes:
            file.write(str(inx) + '\n')


def _write_csv(filename, output_folder, data, header):
    with open(output_folder + filename + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        writer.writerow(header)
        writer.writerows(data)


def _create_matrix(data, target, cost_mat, feature_names, target_names):
    n_samples = len(target)
    good = target[:] == 0
    bad = target[:] == 1
    good_cost = np.zeros([n_samples, 1])
    good_cost[good, 0] = cost_mat[good, 3] # TN
    good_cost[bad, 0] = cost_mat[bad, 1] # FN
    bad_cost = np.zeros([n_samples, 1])
    bad_cost[good, 0] = cost_mat[good, 0] # FP
    bad_cost[bad, 0] = cost_mat[bad, 2] # TP

    final_data = np.append(data, good_cost, axis=1)
    final_data = np.append(final_data, bad_cost, axis=1)

    return final_data, np.append(feature_names, target_names)
