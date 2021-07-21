#!/usr/bin/env python

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds, \
    TruePositiveRateParity, FalsePositiveRateParity, ErrorRateParity, BoundedGroupLoss
from fairlearn.metrics import *
from raiwidgets import FairnessDashboard


def get_data(file):
    data = pd.read_csv(file)
    data[['score', 'race']] = data[['score', 'race']].astype(int)
    print(data)
    return data


def prep_data(data, test_size, weight_index):
    # might need to include standardscaler here

    x = data[['score', 'race']].values
    y = data['repay_indices'].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    # collect our sensitive attribute
    race_train = X_train[:, 1]
    race_test = X_test[:, 1]

    # weight_index: 1 means all equal weights
    if weight_index:
        print('Sample weights are all equal.')
        sample_weight_train = np.ones(shape=(len(y_train),))
        sample_weight_test = np.ones(shape=(len(y_test),))
    # weight_index: 0 means use sample weights
    elif weight_index:
        print('Sample weights are NOT all equal.')
        # TODO
        print('TODO')
    return X_train, X_test, y_train, y_test, race_train, race_test, sample_weight_train, sample_weight_test


def evaluation_outcome_rates(y_true, y_pred, sample_weight):
    fner = false_negative_rate(y_true, y_pred, pos_label=1, sample_weight=sample_weight)
    print('FNER=FN/(FN+TP)= ', fner)
    fper = false_positive_rate(y_true, y_pred, pos_label=1, sample_weight=sample_weight)
    print('FPER=FP/(FP+TN)= ', fper)
    tnr = true_negative_rate(y_true, y_pred, pos_label=1, sample_weight=sample_weight)
    print('TNR=TN/(TN+FP)= ', tnr)
    tpr = true_positive_rate(y_true, y_pred, pos_label=1, sample_weight=sample_weight)
    print('TPR=TP/(FP+FN)= ', tpr)
    return


def evaluation_by_race(X_test, y_test, race_test, y_predict, sample_weight):
    y_test_black, y_pred_black, sw_black, y_test_white, y_pred_white, sw_white = [], [], [], [], [], []

    # splitting up the y_test and y_pred values by race to then use for race specific classification reports
    for index, race in enumerate(race_test):
        if (race == 0):  # black
            y_test_black.append(y_test[index])
            y_pred_black.append(y_predict[index])
            sw_black.append(sample_weight[index])
        elif (race == 1):  # white
            y_test_white.append(y_test[index])
            y_pred_white.append(y_predict[index])
            sw_white.append(sample_weight[index])

        else:
            print('You should not end up here...')

    print('EVALUATION FOR BLACK GROUP')
    cm_black = confusion_matrix(y_test_black, y_pred_black)
    # display_cm(cm_black, 'Confusion Matrix for Black Group')
    print(cm_black)
    print(classification_report(y_test_black, y_pred_black))
    evaluation_outcome_rates(y_test_black, y_pred_black, sw_black)

    print('\nEVALUATION FOR WHITE GROUP')
    cm_white = confusion_matrix(y_test_white, y_pred_white)
    # display_cm(cm_white, 'Confusion Matrix for White Group')
    print(cm_white)
    print(classification_report(y_test_white, y_pred_white))
    evaluation_outcome_rates(y_test_white, y_pred_white, sw_white)
    return


# Reference: https://fairlearn.org/v0.5.0/api_reference/fairlearn.metrics.html
def add_contraint(model, constraint_str, reduction_alg, X_train, y_train, race_train, race_test, X_test, y_test, y_predict, sample_weight_test, dashboard_bool):
    # set seed for consistent results with ExponentiatedGradient
    np.random.seed(0)
    if constraint_str == 'DP':
        constraint = DemographicParity()
    elif constraint_str == 'EO':
        constraint = EqualizedOdds()
    elif constraint_str == 'TPRP':
        constraint = TruePositiveRateParity()
    elif constraint_str == 'FPRP':
        constraint = FalsePositiveRateParity()
    elif constraint_str == 'ERP':
        constraint = ErrorRateParity()
    elif constraint_str == 'BGL':
        # Parameters:
        #   loss : {SquareLoss, AbsoluteLoss}
        #   A loss object with an `eval` method, e.g. `SquareLoss` or `AbsoluteLoss`
        constraint = BoundedGroupLoss('SquareLoss')

    if reduction_alg == 'EG':
        mitigator = ExponentiatedGradient(model, constraint)
        print('Exponentiated Gradient Reduction Alg is used here with ', constraint_str,
              ' as the fairness constraint.\n')
    elif reduction_alg == 'GS':
        mitigator = GridSearch(model, constraint)
        print('Grid Search Reduction Alg is used here with ', constraint_str, ' as the fairness constraint.\n')
    else:
        print('ISSUE: need to put in a valid reduction_alg parameter')

    mitigator.fit(X_train, y_train, sensitive_features=race_train)
    y_pred_mitigated = mitigator.predict(X_test)

    print('Evaluation of ', constraint_str, '-constrained classifier overall:')
    cm = confusion_matrix(y_test, y_pred_mitigated)
    print(cm)
    print(classification_report(y_test, y_pred_mitigated))
    evaluation_outcome_rates(y_test, y_pred_mitigated, sample_weight_test)
    print('\n')
    print('Evaluation of ', constraint_str, '-constrained classifier by race:')
    evaluation_by_race(X_test, y_test, race_test, y_pred_mitigated, sample_weight_test)
    print('\n')
    print('Fairness metric evaluation of ', constraint_str, '-constrained classifier')
    print_fairness_metrics(y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=race_test)

    if dashboard_bool:
        FairnessDashboard(sensitive_features=race_test,y_true=y_test,
                          y_pred={"initial model": y_predict, "mitigated model": y_pred_mitigated})
    calculate_delayed_impact(X_test, y_test, y_pred_mitigated, race_test)
    return


def print_fairness_metrics(y_true, y_pred, sensitive_features):
    sr_mitigated = MetricFrame(metric=selection_rate, y_true=y_true, y_pred=y_pred,
                               sensitive_features=sensitive_features)
    print('Selection Rate Overall: ', sr_mitigated.overall)
    print('Selection Rate By Group: ', sr_mitigated.by_group, '\n')

    dp_diff = demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    print('DP Difference: ', dp_diff)
    print('-->difference of 0 means that all groups have the same selection rate')
    dp_ratio = demographic_parity_ratio(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    print('DP Ratio:', dp_ratio)
    print('-->ratio of 1 means that all groups have the same selection rate \n')

    eod_diff = equalized_odds_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    print('EOD Difference: ', eod_diff)
    print('-->difference of 0 means that all groups have the same TN, TN, FP, and FN rates')
    eod_ratio = equalized_odds_ratio(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    print('EOD Ratio:', eod_ratio)
    print('-->ratio of 1 means that all groups have the same TN, TN, FP, and FN rates rates')
    return

def calculate_delayed_impact(X_test, y_true, y_pred, race_test):
    # TPs --> score increase by 75
    # FPs --> score drop of 150
    # TNs and FNs do not change (in this case)
    # Delayed Impact (DI) is the average score change of each group
    # In race_test array, Black is 0 and White it 1

    di_black, di_white = 0, 0
    score_diff_black, score_diff_white = [], []
    scores = X_test[:,0]

    for index, true_label in enumerate(y_true):
        # check for TPs
        if true_label == y_pred[index] and true_label==1:
            if race_test[index] == 0:  # black borrower
                score_diff_black.append(75)
            elif race_test[index] == 1:  # white borrower
                score_diff_white.append(75)
        # check for FPs
        elif true_label == 0 and y_pred[index] == 1:
            if race_test[index] == 0:  # black borrower
                score_diff_black.append(-150)
            elif race_test[index] == 1:  # white borrower
                score_diff_white.append(-150)
        elif (true_label == y_pred[index] and true_label == 0) or (true_label == 1 and y_pred[index] == 0):
            if race_test[index] == 0:  # black indiv
                score_diff_black.append(0)
            elif race_test[index] == 1:  # white indiv
                score_diff_white.append(0)


    # calculate mean score difference or delayed impact of each group
    di_black = sum(score_diff_black)/len(score_diff_black)
    di_white = sum(score_diff_white)/len(score_diff_white)

    print('The delayed impact of the black group is: ', di_black)
    print('The delayed impact of the white group is: ', di_white)
    return

