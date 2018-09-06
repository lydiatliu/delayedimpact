import numpy as np
import matplotlib.pyplot as plt
import math
import solve_credit as sc
import prob_util_functions as puf
from scipy.stats import binom 



def get_thresholds(loan_repaid_probs,pis,group_size_ratio,utils,score_change_fns,scores):
    # unpacking bank utilities
    utility_default, utility_repaid = utils
    break_even_prob = utility_default/(utility_default-utility_repaid)

    # getting loan policies
    data = sc.fairness_data(get_rhos(loan_repaid_probs,scores), pis,
                                                group_size_ratio, break_even_prob)
    tau_MP = data.max_profit_loans()
    tau_DP = data.demographic_loans()
    tau_EO = data.equal_opp_loans()

    thresh_MP = get_thresholds_from_taus(tau_MP,scores)
    thresh_DP = get_thresholds_from_taus(tau_DP,scores)
    thresh_EO = get_thresholds_from_taus(tau_EO,scores)

    # unpacking repay probability as a function of score
    loan_repay_fns = [lambda x: loan_repaid_prob(x) for loan_repaid_prob in loan_repaid_probs]
    # getting threshold at which average score change becomes negative
    thresh_downwards = [get_mean_movement_threshold(pis[0], scores, loan_repay_fn, score_change_fns) for loan_repay_fn in loan_repay_fns]

    return thresh_DP, thresh_EO, thresh_MP, thresh_downwards

def get_outcome_curve(loan_repay_fn,pi,scores,impacts):
    delta_mu = np.zeros(scores.size)
    delta_mu[0] = exp_move(scores[-1], loan_repay_fn,bounds=[300,850], move_vec=impacts)*pi[-1]
    for i in range(1,scores.size):
        delta_mu[i] = exp_move(scores[-(i+1)], loan_repay_fn,bounds=[300,850], move_vec=impacts)*pi[-(i+1)] + delta_mu[i-1]

    return delta_mu

def get_utility_curve(loan_repay_fns,pis,scores,utils):
    Util = np.zeros([2,scores.size])
    for j in range(2):
        Util[j,0] = bank_util(scores[-1], loan_repay_fns[j], utils= utils)*pis[j,-1]
        for i in range(1,scores.size):
            Util[j,i] =  bank_util(scores[-(i+1)], loan_repay_fns[j], utils= utils)*pis[j,-(i+1)] + Util[j,i-1]
    return Util

def get_utility_curves_DP(Util,cdfs,group_size_ratio,scores):
    Util_total = np.zeros([2,scores.size])
    cdfs[0] = list(reversed(1-cdfs[0]))
    cdfs[1] = list(reversed(1-cdfs[1]))

    for i_A in range(scores.size):
        prop_A = cdfs[0,i_A]
        i_B = find_nearest(cdfs[1],prop_A)
        Util_total[0, i_A] = group_size_ratio[0]*Util[0,i_A] + group_size_ratio[1]*Util[1,i_B]

    for i_B in range(scores.size):
        prop_B = cdfs[1,i_B]
        i_A = find_nearest(cdfs[0],prop_B)
        Util_total[1, i_B] = group_size_ratio[0]*Util[0,i_A] + group_size_ratio[1]*Util[1,i_B]

    return Util_total

def get_utility_curves_EO(Util,loan_repaid_probs,pis,group_size_ratio,scores):
    rescaled_pis = np.zeros(pis.shape)
    N_groups, N_scores = pis.shape
    for group in range(N_groups):
        for x in range(N_scores):
            rescaled_pis[group,x] = pis[group,x] * loan_repaid_probs[group](scores[x])
        rescaled_pis[group] = rescaled_pis[group] / np.sum(rescaled_pis[group])

    cdfs = np.zeros(pis.shape)
    for group in range(N_groups):
        cdfs[group,0] = rescaled_pis[group,0]
        for x in range(1,N_scores):
            cdfs[group,x] = rescaled_pis[group,x] + cdfs[group,x-1]
    return get_utility_curves_DP(Util,cdfs,group_size_ratio, scores)

########## helper functions #####################################################

def get_rhos(loan_repaid_probs,scores):
    """set up rhos[i,j] = probability of group i member repaying loan at state j"""
    N_scores = len(scores)
    N_groups = len(loan_repaid_probs)
    rhos = np.zeros((N_groups,N_scores))
    for j,s in enumerate(scores):
        for i in range(N_groups):
            rhos[i,j] = loan_repaid_probs[i](s)
    return rhos

# randomized threshold below which group's mean score will decrease, above which it increases
def get_mean_movement_threshold(pi, scores, loan_repay_fn, score_change_fns, compare_pt=0.0):
    running_sum = 0
    for i,s in enumerate(reversed(scores)):
        x = len(scores) - 1 - i 
        if ((running_sum + pi[x] * exp_move(s, loan_repay_fn, move_vec=score_change_fns)) < compare_pt):
            randomized = (compare_pt-running_sum) /(exp_move(s, loan_repay_fn, move_vec=score_change_fns)*pi[x])
            assert randomized >= 0
            assert randomized <= 1
            if i == 0:
                return s
            return s + randomized * abs(scores[x] - scores[x+1])
        else:
            running_sum += pi[x] * exp_move(s, loan_repay_fn)
    return s

def exp_move(x, loan_repay_fn, bounds=[300,850], move_vec= [-150,75]):
    move_down = move_vec[0] 
    move_up = move_vec[1] 
    move = (1-loan_repay_fn(x))*move_down + loan_repay_fn(x)*move_up
    move_to_within_bounds = max(min(x+move, bounds[1]),bounds[0])
    move_within_bounds = move_to_within_bounds - x
    return move_within_bounds

def bank_util(x, loan_repay_fn, utils):
    util_repay = utils[1] 
    util_def = utils[0] 
    return (1-loan_repay_fn(x))*util_def + loan_repay_fn(x)*util_repay

# helper functions for turning loaning policies into thresholds for visualization
def get_thresholds_from_taus(taus,scores):
    thresholds = []
    for tau in taus:
        x = np.amin(np.where(tau>0))
        thresholds.append(scores[x] + (1-tau[x])*np.abs(scores[x+1]-scores[x])) #to visualize the randomization
    return thresholds

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return int(np.amin(idx))
