import numpy as np
import matplotlib.pyplot as plt

# binary search implementation
def binary_search(f, target, left=1e-5, right=1 - 1e-5, tol=1e-8):
    midpoint = (left + right) * .5
    if abs(midpoint - left) < tol:
        return midpoint
    elif f(midpoint) < target:
        return binary_search(f, target, midpoint, right, tol)
    elif (f(midpoint) >= target):
        return binary_search(f, target, left, midpoint, tol)
    else:
        print("we should never get down here")    

def a_max(f,lst):
    a_max = lst[0]
    for l in lst:
        if f(l) > f(a_max):
            a_max = l
    return a_max

# ternary search on the equalized criterion (loan rate, ppv, etc..)
def ternary_maximize(f, left=1e-5, right=1 - 1e-5, tol=1e-5):
    m_1 = (2. / 3) * left + (1. / 3) * right
    m_2 = (1. / 3) * left + (2. / 3) * right
    if abs(m_1 - m_2) < tol:
        return a_max(f,[m_1,m_2])
    if f(m_1) < f(m_2):
        return ternary_maximize(f, m_1, right, tol)
    if f(m_1) >= f(m_2):
        return ternary_maximize(f, left, m_2, tol)
    else:
        print("fly you fools - we should never get down here")

# takes in a loan rate between 0 and 1 and a pdf
# returns a vector of the optimal loaning per score
# that achieves the rate 
def rate_to_loans(rate,pdf):
    output = np.zeros(len(pdf))
    total = 0
    for i in range(len(pdf)):
        if pdf[-i-1] + total > rate:
            thing = rate - total
            output[-i-1] = thing/pdf[-i-1]
            return output
        else:
            output[-i-1] = 1
            total = total+pdf[-i-1]

    return output

def loans_to_rate(loans,pdf):
    return sum(pdf*loans)

# computes the tp of a given loan rate
def rate_to_tp(rate,pdf,perf):
    loans = rate_to_loans(rate,pdf)
    return loans_to_tp(loans,pdf,perf)

def loans_to_tp(loans,pdf,perf):
    would_pay = np.sum(pdf*perf)
    will_pay = np.sum(pdf*perf*loans)
    return will_pay/would_pay

def tp_to_loan_rate(tp,pdf,perf):
    f = lambda t: rate_to_tp(t,pdf,perf)
    rate =  binary_search(f , target = tp)
    return rate

def tp_to_loans(tp,pdf,perf):
    rate = tp_to_loan_rate(tp,pdf,perf)
    loans = rate_to_loans(tp_to_loan_rate(tp,pdf,perf), pdf)
    return loans

def profit_from_roc():
    return 

class fairness_data:
    # @param perf[i][j] = performance of class i, score j
    # @param pdf[i][j] = fraction of class i w/ score j
    # @param props[i] = proportion of popultion in class i
    # @param break_even = loan rate yielding break_even (
    # equivalent to loan utility)
    def __init__(self, perf, pdf, props, break_even):
        self.perf = perf
        self.pdf = pdf
        self.props = props
        self.num_groups = len(props)
        self.break_even = break_even

    def update_break_even(break_even):
        self.break_even = break_even

    def update_pdf(pdf):
        self.pdf = pdf

    def loans_from_rate(self,rate):
        loans = np.zeros(self.pdf.shape)
        for i in range(self.num_groups):
            loans[i] = rate_to_loans(rate,self.pdf[i])
        return loans

    def loans_from_tp(self,tp):
        loans = np.zeros(self.pdf.shape)
        for i in range(self.num_groups):
            loans[i] = tp_to_loans(tp,self.pdf[i],self.perf[i])      
        return loans

    def compute_profit(self,break_even,loans):
        in_groups_prof = np.sum((self.perf - break_even) * loans*self.pdf, axis = 1)
        return np.dot(self.props, in_groups_prof)
    
    def get_break_even(self,break_even):
        if break_even == None:
            return self.break_even
        return break_even

    # for testing purposes
    def graph_dem_parity(self):
        f_prof = lambda rate: self.compute_profit(self.break_even,self.loans_from_rate(rate))
        rates = []
        profits = []
        for t in range(10):
            rate = .899 +  t/1000.0
            rates.append(rate)
            print(self.loans_from_rate(rate))
            profits.append(f_prof(rate))
        print(ternary_maximize(f_prof))
        plt.plot(rates,profits)
        return rates,profits

    def demographic_loans(self, break_even = None):
        break_even = self.get_break_even(break_even)
        f_prof = lambda rate: self.compute_profit(break_even,self.loans_from_rate(rate))
        target_rate = ternary_maximize(f_prof) 
        return self.loans_from_rate(target_rate)

    def max_profit_loans(self,break_even = None):
        break_even = self.get_break_even(break_even)
        return np.trunc(self.perf + 1 - break_even)

    def equal_opp_loans(self,break_even = None):
        break_even = self.get_break_even(break_even)
        f_prof = lambda tp: self.compute_profit(break_even,self.loans_from_tp(tp))
        target_tp = ternary_maximize(f_prof)
        return self.loans_from_tp(target_tp)
   
