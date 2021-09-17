#!/usr/bin/env python

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score, f1_score
import pandas as pd
import numpy as np
from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds, \
    TruePositiveRateParity, FalsePositiveRateParity, ErrorRateParity, BoundedGroupLoss
from fairlearn.metrics import *
from raiwidgets import FairnessDashboard
import matplotlib.pyplot as plt



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

    print('Here are the x values: ', x, '\n')
    print('Here are the y values: ', y)

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

def get_selection_rates(y_true, y_pred, sensitive_features, type_index):
    sr_mitigated = MetricFrame(metric=selection_rate, y_true=y_true, y_pred=y_pred,
                               sensitive_features=sensitive_features)
    if type_index == 0:
        print('Selection Rate Overall: ', sr_mitigated.overall)
    elif type_index == 1:
        print('Selection Rate By Group: ', sr_mitigated.by_group, '\n')
    else:
        print('ISSUE: input 0 or 1 as 4th parameter')
    return


def evaluation_outcome_rates(y_true, y_pred, sample_weight):
    tnr = true_negative_rate(y_true, y_pred, pos_label=1, sample_weight=sample_weight)
    print('TNR=TN/(TN+FP)= ', tnr)
    tpr = true_positive_rate(y_true, y_pred, pos_label=1, sample_weight=sample_weight)
    print('TPR=TP/(FP+FN)= ', tpr)
    fner = false_negative_rate(y_true, y_pred, pos_label=1, sample_weight=sample_weight)
    print('FNER=FN/(FN+TP)= ', fner)
    fper = false_positive_rate(y_true, y_pred, pos_label=1, sample_weight=sample_weight)
    print('FPER=FP/(FP+TN)= ', fper)
    return


# Resource for below: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

def get_f1_scores(y_test, y_predict):
    # F1 score micro: calculate metrics globally by counting the total true positives, false negatives and false positives
    print('F1 score micro: ')
    print(f1_score(y_test, y_predict, average='micro'))
    # F1 score weighted: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    print('F1 score weighted: ')
    print(f1_score(y_test, y_predict, average='weighted'))
    # F1 score binary: Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
    print('F1 score binary: ')
    print(f1_score(y_test, y_predict, average='binary'))
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

    get_selection_rates(y_test, y_predict, race_test, 1)

    print('EVALUATION FOR BLACK GROUP')
    cm_black = confusion_matrix(y_test_black, y_pred_black)
    # display_cm(cm_black, 'Confusion Matrix for Black Group')
    print(cm_black)
    print(classification_report(y_test_black, y_pred_black))
    evaluation_outcome_rates(y_test_black, y_pred_black, sw_black)
    get_f1_scores(y_test_black, y_pred_black)

    print('\nEVALUATION FOR WHITE GROUP')
    cm_white = confusion_matrix(y_test_white, y_pred_white)
    # display_cm(cm_white, 'Confusion Matrix for White Group')
    print(cm_white)
    print(classification_report(y_test_white, y_pred_white))
    evaluation_outcome_rates(y_test_white, y_pred_white, sw_white)
    get_f1_scores(y_test_white, y_pred_white)
    return


def grid_search_show(model, constraint, y_predict, X_test, y_test, race_test, constraint_name, model_name, models_dict, decimal):
    sweep_preds = [predictor.predict(X_test) for predictor in model.predictors_]
    sweep_scores = [predictor.predict_proba(X_test)[:, 1] for predictor in model.predictors_]

    sweep = [constraint(y_test, preds, sensitive_features=race_test)
             for preds in sweep_preds]
    balanced_accuracy_sweep = [balanced_accuracy_score(y_test, preds) for preds in sweep_preds]
    # auc_sweep = [roc_auc_score(y_test, scores) for scores in sweep_scores]

    # Select only non-dominated models (with respect to balanced accuracy and equalized odds difference)
    all_results = pd.DataFrame(
        {"predictor": model.predictors_, "accuracy": balanced_accuracy_sweep, "disparity": sweep}
    )
    non_dominated = []
    for row in all_results.itertuples():
        accuracy_for_lower_or_eq_disparity = all_results["accuracy"][all_results["disparity"] <= row.disparity]
        if row.accuracy >= accuracy_for_lower_or_eq_disparity.max():
            non_dominated.append(True)
        else:
            non_dominated.append(False)

    sweep_non_dominated = np.asarray(sweep)[non_dominated]
    balanced_accuracy_non_dominated = np.asarray(balanced_accuracy_sweep)[non_dominated]
    # auc_non_dominated = np.asarray(auc_sweep)[non_dominated]

    # Plot DP difference vs balanced accuracy
    plt.scatter(balanced_accuracy_non_dominated, sweep_non_dominated, label=model_name)
    plt.scatter(balanced_accuracy_score(y_test, y_predict),
                constraint(y_test, y_predict, sensitive_features=race_test),
                label="Unmitigated Model")
    plt.xlabel("Balanced Accuracy")
    plt.ylabel(constraint_name)
    plt.legend(bbox_to_anchor=(1.55, 1))
    plt.show()

    models_dict = update_model_perf_dict(sweep, models_dict, sweep_preds, sweep_scores, non_dominated, decimal, y_test, race_test, model_name)

    return

def update_model_perf_dict(sweep, models_dict, sweep_preds, sweep_scores, non_dominated, decimal, y_test, race_test, model_name):
    # Compare GridSearch models with low values of fairness-diff with the previously constructed models
    #print(model_name)
    grid_search_dict = {model_name.format(i): (sweep_preds[i], sweep_scores[i]) #{"GS_DP".format(i): (sweep_preds[i], sweep_scores[i])
                        for i in range(len(sweep_preds))
                        if non_dominated[i] and sweep[i] < decimal}
    models_dict.update(grid_search_dict)
    print(get_metrics_df(models_dict, y_test, race_test))
    return models_dict

def get_metrics_df(models_dict, y_true, group):
    metrics_dict = {
        "Overall selection rate": (
            lambda x: selection_rate(y_true, x), True),
        "Demographic parity difference": (
            lambda x: demographic_parity_difference(y_true, x, sensitive_features=group), True),
        "Demographic parity ratio": (
            lambda x: demographic_parity_ratio(y_true, x, sensitive_features=group), True),
        "------": (lambda x: "", True),
        "Overall balanced error rate": (
            lambda x: 1-balanced_accuracy_score(y_true, x), True),
        "Balanced error rate difference": (
            lambda x: MetricFrame(metrics=balanced_accuracy_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), True),
        " ------": (lambda x: "", True),
        "True positive rate difference": (
            lambda x: true_positive_rate_difference(y_true, x, sensitive_features=group), True),
        "True negative rate difference": (
            lambda x: true_negative_rate_difference(y_true, x, sensitive_features=group), True),
        "False positive rate difference": (
            lambda x: false_positive_rate_difference(y_true, x, sensitive_features=group), True),
        "False negative rate difference": (
            lambda x: false_negative_rate_difference(y_true, x, sensitive_features=group), True),
        "Equalized odds difference": (
            lambda x: equalized_odds_difference(y_true, x, sensitive_features=group), True),
        "  ------": (lambda x: "", True),
        "Overall AUC": (
            lambda x: roc_auc_score(y_true, x), False),
        "AUC difference": (
            lambda x: MetricFrame(metrics=roc_auc_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), False),
    }
    df_dict = {}
    for metric_name, (metric_func, use_preds) in metrics_dict.items():
        df_dict[metric_name] = [metric_func(preds) if use_preds else metric_func(scores)
                                for model_name, (preds, scores) in models_dict.items()]
    return pd.DataFrame.from_dict(df_dict, orient="index", columns=models_dict.keys())

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
    get_f1_scores(y_test, y_pred_mitigated)
    get_selection_rates(y_test, y_pred_mitigated, race_test, 0)
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
    return mitigator


def print_fairness_metrics(y_true, y_pred, sensitive_features):
    #sr_mitigated = MetricFrame(metric=selection_rate, y_true=y_true, y_pred=y_pred,
    #                           sensitive_features=sensitive_features)
    #print('Selection Rate Overall: ', sr_mitigated.overall)
    #print('Selection Rate By Group: ', sr_mitigated.by_group, '\n')

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
    print('-->ratio of 1 means that all groups have the same TN, TN, FP, and FN rates rates \n')
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

