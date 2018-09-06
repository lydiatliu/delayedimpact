import numpy as np
import scipy.linalg

################## functions for construction of transition probability matrix ###################


def drift_up_coeff_default(i, N_scores, mean=None):
    # pushes you up if you're below halfway (proportional to squared distance to middle)
    c = 0
    if mean is None:
        mean = (N_scores-1)/2
    distance_below_middle = max(0, mean - i)
    return (1-c)*abs(distance_below_middle/N_scores) + c/3.0


def drift_down_coeff_default(i, N_scores, mean=None):
    # pushes you down if you're above halfway (proportional to squared distance to middle)
    c = 0
    if mean is None:
        mean = (N_scores-1)/2
    distance_above_middle = max(0, i - mean)
    return (1-c)*abs(distance_above_middle/N_scores) + c/3.0


def loan_repaid_prob_default(i, N_scores):
    c = 1.0
    return c*(i+1)/N_scores


def mat_to_fn(i,j, mat):
        return mat[i,j]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def logistic_repayment(i, N_scores):
    c = 0.9
    return c * sigmoid(i - (N_scores - 1) / 2)


def fake_normal_dist(mu, sigma=1, N_scores=10):
    p = np.random.randn(10000) + mu
    p = np.floor(p)
    pi = np.zeros(N_scores)
    for i in range(N_scores):
        pi[i] = len(np.where(p == i)[0])
    return pi / np.sum(pi)
