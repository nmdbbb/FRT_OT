"""
utilities.py — lightweight version
Only contains run_knn; all FRT logic moved to frt.run_frt_pipeline
"""
from __future__ import annotations

import os

import joblib
import numpy as np
from aeon.datasets import load_from_tsfile
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import ot
import time

from frt import run_frt_pipeline
from gow import gow_sinkhorn_autoscale


def load_human_action_dataset(data_dir, dataset_name):
    '''
    Loads train and test data from the folder in which
    the Human Actions dataset are stored.
    '''
    X_train = joblib.load(os.path.join(data_dir, dataset_name, "X_train.pkl"))
    y_train = joblib.load(os.path.join(data_dir, dataset_name, "y_train.pkl"))
    X_test = joblib.load(os.path.join(data_dir, dataset_name, "X_test.pkl"))
    y_test = joblib.load(os.path.join(data_dir, dataset_name, "y_test.pkl"))

    print("Successfully loaded dataset:", dataset_name)
    print("Size of train data:", len(y_train))
    print("Size of test data:", len(y_test))

    return X_train, y_train, X_test, y_test

def fix_array(X):
    X_new = np.empty((X.shape[0], X.shape[2], X.shape[1]))

    for i in range (X_new.shape[0]):
        X_new[i] = np.transpose(X[i])

    return X_new

def load_ucr_dataset(data_dir, dataset_name):
    '''
    Loads train and test data from the folder in which
    the UCR dataset are stored.
    '''
    X, y_train = load_from_tsfile(os.path.join(data_dir, dataset_name, f'{dataset_name}_TRAIN'))
    X_train = fix_array(X)
    X, y_test = load_from_tsfile(os.path.join(data_dir, dataset_name, f'{dataset_name}_TEST'))
    X_test = fix_array(X)

    print("Successfully loaded dataset:", dataset_name)
    print("Size of train data:", len(y_train))
    print("Size of test data:", len(y_test))

    return X_train, y_train, X_test, y_test

def run_knn(X_train, y_train, X_test, y_test,
            alg, k_list=[1, 3, 5, 10, 15, 20]):
    
    train_len = len(y_train)
    test_len = len(y_test)
    D_tr = np.ones((train_len, train_len))
    D_te = np.empty((test_len, train_len))
    knn_secs = {}
#=========================================================================

    # Set parameters of GOW
    normalize_cost_matrix=True
    cost_metric="minkowski"
    
    # Set parameters of FRT
    n_trees = 16
    time_weight = "auto"
    time_factor = 2**6
    random_state = 123
    level_edge_shift = 1
    depth_shift = "auto"
    
#=========================================================================
    if alg == "FRT":
        X_train = [x for x in X_train]
        X_test = [x for x in X_test]
        D_tr, D_te, meta = run_frt_pipeline(
            X_train, X_test,
            n_trees=n_trees,
            time_weight=time_weight,
            time_factor=time_factor,
            random_state=random_state,
            level_edge_shift=level_edge_shift,
            depth_shift=depth_shift
        )
    elif alg == "GOW":
        D_te = np.empty((test_len, train_len))

        for i in range(test_len):
            print("Batch:", str(i+1) + "/" + str(test_len))
            for j in range(train_len):
                C = ot.dist(X_test[i], X_train[j], metric=cost_metric)
                if normalize_cost_matrix:
                    C = C / C.max()

                D_te[i][j] = gow_sinkhorn_autoscale([], [], C)

    else:
        raise ValueError("alg not recognized")

    n_tr = len(y_train)
    results = {}
    
    for k in k_list:
        
        knn_secs[k] = 0.0
        knn_sec_start = time.time()
        k_actual = min(int(k), n_tr)
        clf = neighbors.KNeighborsClassifier(n_neighbors=k_actual, metric="precomputed")
        clf.fit(D_tr, y_train)
        knn_secs[k] += time.time() - knn_sec_start
        acc = accuracy_score(y_test, clf.predict(D_te))
        print(f"Accuracy (k={k_actual}): {acc:.4f}")
        results[k_actual] = float(acc)
        results["build_tree_sec"] = float(meta.build_tree_sec) if alg == "FRT" else 0.0
        results["distance_calc_sec"] = float(meta.distance_calc_sec) if alg == "FRT" else 0.0
        
    return results,knn_secs
