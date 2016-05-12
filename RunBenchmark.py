# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:16:49 2016

@author: GabrielLocal
"""




from enum import Enum
class Dataset_Enum(Enum):
    bankmarketing   = 1
    creditscoring1  = 2
    creditscoring2  = 3
    german          = 4



def RunBenchmark(dataset_enum=Dataset_Enum.bankmarketing, numFolds=1):
    """
        This function runs the benchmark from AAAAAAAAAAAAAAAAAAAAAAAA


    """

    print_verbose = False


    ######################################################
    ##############  I  M  P  O  R  T  S  #################
    ######################################################

    from costcla.datasets import load_creditscoring1
    from costcla.datasets import load_creditscoring2
    from costcla.datasets import load_bankmarketing

    from sklearn.cross_validation import train_test_split

    from costcla.datasets.base import load_creditgerman
    from costcla.utils.dataset_folds import create_all_folds


    ######################################################
    ######## L O A D I N G   D A T A S E T S #############
    ######################################################

    # Load dataset
    data = {}
    if   dataset_enum is Dataset_Enum.bankmarketing:
        data = load_bankmarketing()
    elif dataset_enum == Dataset_Enum.creditscoring1:
        data = load_creditscoring1()
    elif dataset_enum == Dataset_Enum.creditscoring2:
        data = load_creditscoring2()
    elif dataset_enum == Dataset_Enum.german:
        data = load_creditgerman()

    if print_verbose:
        print("data loaded, ", dataset_enum.name)





    ######################################################
    ###### R U N N I N G   T H E   B E N C H M A R K #####
    ######################################################

    results = {}


    if numFolds == 1:       # run the Benchmark once and go!

        # Split dataset in training and testing
        X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = \
        train_test_split(data.data, data.target, data.cost_mat)

        results[0] = RunBenchmark_private(X_train, X_test, y_train, y_test, \
        cost_mat_train, cost_mat_test)

    else:                   # run each fold, then average the values!
        ds = create_all_folds(data, numFolds=numFolds, trainRatio=0.8)

        for key, fold in ds.folds.items():
            results[key] = RunBenchmark_private(fold.x_train, fold.x_test, \
            fold.y_train, fold.y_test, fold.cost_mat_train, fold.cost_mat_test)

    return results








def RunBenchmark_private(X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test):

    """
    In this function, the code of the tutorial published by @ALBAHNSEN in
    http://nbviewer.jupyter.org/github/albahnsen/CostSensitiveClassification/blob/master/doc/tutorials/tutorial_edcs_credit_scoring.ipynb
    is reproduced in the form of a Benchmark.

    There are 8 classifiers in total, given in three groups:

        - The classical classifiers:
            + RF - Random Forest
            + DT - Decision Tree
            + LR - Logistic Regression

        - The Bayes Minimum Risk classifier approach, based on the three classical classifiers:
            + RF-BMR
            + DT-BMR
            + LR-BMR

        - The Cost Sensitive classifiers:
            + CSDT - Cost Sensitive Decision Tree
            + CSRF - Cost Sensitive Random Forest

    """

    print_verbose = True

    ######################################################
    ###########  I  M  P  O  R  T  S  ####################
    ######################################################

    # First round of imports, the libraries and the dataset
    import pandas as pd
    import numpy as np


    # Second round, the models and the dataset splitter
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    # Third round, the metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    # Fourth round, the plotting tools
    from IPython.core.pylabtools import figsize
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Fifth round, the new classifiers (finally!)
    from costcla.metrics import savings_score
    from costcla.models import BayesMinimumRiskClassifier
    from costcla.models import CostSensitiveDecisionTreeClassifier
    from costcla.models import CostSensitiveRandomPatchesClassifier



    ######################################################
    ##### M E A S U R E S   A N D   R E S U L T S ########
    ######################################################

    # Measures to evaluate the performance
    measures = {"f1": f1_score, "pre": precision_score,
                "rec": recall_score, "acc": accuracy_score}

    results = pd.DataFrame(columns=measures.keys())
    results["sav"] = np.zeros(results.shape[0])






    ######################################################
    ######################################################
    ### Fit the classifiers using the training dataset ###
    ######################################################
    ######################################################



    ######################################################
    ######## C L A S S I C A L   A P P R O A C H #########
    ######################################################

    # First set of classifiers: RF, DT and LR
    classifiers = {"RF": {"f": RandomForestClassifier()},
                   "DT": {"f": DecisionTreeClassifier()},
                   "LR": {"f": LogisticRegression()}}

    for model in classifiers.keys():
        # Fit
        classifiers[model]["f"].fit(X_train, y_train)
        # Predict
        classifiers[model]["c"] = classifiers[model]["f"].predict(X_test)
        classifiers[model]["p"] = classifiers[model]["f"].predict_proba(X_test)
        classifiers[model]["p_train"] = classifiers[model]["f"].predict_proba(X_train)

        # Evaluate
        results.loc[model] = 0
        results.loc[model, measures.keys()] = \
        [measures[measure](y_test, classifiers[model]["c"]) for measure in measures.keys()]
        results["sav"].loc[model] = savings_score(y_test, classifiers[model]["c"], cost_mat_test)

    if print_verbose:
        print("the first three classifiers are fitted")





    ######################################################
    ######## B A Y E S   M I N I M U M   R I S K #########
    ######################################################

    # Second set of classifiers: RF-BMR, DT-BMR and LR-BMR
    ci_models = classifiers.copy().keys()
    for model in ci_models:
        classifiers[model+"-BMR"] = {"f": BayesMinimumRiskClassifier()}
        # Fit
        classifiers[model+"-BMR"]["f"].fit(y_test, classifiers[model]["p"])
        # Calibration must be made in a validation set
        # Predict
        classifiers[model+"-BMR"]["c"] = classifiers[model+"-BMR"]["f"].predict(classifiers[model]["p"], cost_mat_test)
        # Evaluate
        results.loc[model+"-BMR"] = 0
        results.loc[model+"-BMR", measures.keys()] = \
        [measures[measure](y_test, classifiers[model+"-BMR"]["c"]) for measure in measures.keys()]
        results["sav"].loc[model+"-BMR"] = savings_score(y_test, classifiers[model+"-BMR"]["c"], cost_mat_test)

    if print_verbose:
        print("the second three classifiers are done")






    ######################################################
    ########### C S D T   C L A S S I F I E R ############
    ######################################################

    # Third set of classifiers: CSTD and CSRP
    classifiers["CSDT"] = {"f": CostSensitiveDecisionTreeClassifier()}
    # Fit
    classifiers["CSDT"]["f"].fit(X_train, y_train, cost_mat_train)
    # Predict
    classifiers["CSDT"]["c"] = classifiers["CSDT"]["f"].predict(X_test)
    # Evaluate
    results.loc["CSDT"] = 0
    results.loc["CSDT", measures.keys()] = \
    [measures[measure](y_test, classifiers["CSDT"]["c"]) for measure in measures.keys()]
    results["sav"].loc["CSDT"] = savings_score(y_test, classifiers["CSDT"]["c"], cost_mat_test)

    if print_verbose:
        print("the CSDT clasifier is ready")


    ######################################################
    ########### C S R P   C L A S S I F I E R ############
    ######################################################

    classifiers["CSRP"] = {"f": CostSensitiveRandomPatchesClassifier(combination='weighted_voting')}
    # Fit
    classifiers["CSRP"]["f"].fit(X_train, y_train, cost_mat_train)
    # Predict
    classifiers["CSRP"]["c"] = classifiers["CSRP"]["f"].predict(X_test)
    # Evaluate
    results.loc["CSRP"] = 0
    results.loc["CSRP", measures.keys()] = \
    [measures[measure](y_test, classifiers["CSRP"]["c"]) for measure in measures.keys()]
    results["sav"].loc["CSRP"] = savings_score(y_test, classifiers["CSRP"]["c"], cost_mat_test)

    if print_verbose:
        print("the CSRP clasifier is ready")





    ######################################################
    ####### P L O T T I N G   T H E   R E S U L T S ######
    ######################################################
    ## Note to myself: I don't know about plotting, so i just Copy+Paste this here.


    # Plot the results
    #%matplotlib inline
    figsize(10, 5)
    ax = plt.subplot(111)

    ind = np.arange(results.shape[0])
    width = 0.2
    l = ax.plot(ind, results, "-o")
    plt.legend(iter(l), results.columns.tolist(), loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim([-0.25, ind[-1]+.25])
    ax.set_xticks(ind)
    ax.set_xticklabels(results.index)
    plt.show()






    # Plot the results
    colors = sns.color_palette()
    ind = np.arange(results.shape[0])

    figsize(10, 5)
    ax = plt.subplot(111)
    l = ax.plot(ind, results["f1"], "-o", label='F1Score', color=colors[2])
    b = ax.bar(ind-0.3, results['sav'], 0.6, label='Savings', color=colors[0])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim([-0.5, ind[-1]+.5])
    ax.set_xticks(ind)
    ax.set_xticklabels(results.index)
    plt.show()




    return results