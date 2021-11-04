# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 07:34:40 2021

@author: Pernille
"""

import numpy as np
import pandas as pd
import pystan
import pickle

# Fitting function
def bhip_fit(dataframe, model_description, seed):
    Y = dataframe.Y
    X = dataframe.drop(['Y'], axis=1)

    N, D = X.shape
    E = len(dataframe.index.unique())
    e = dataframe.index.tolist()

    data = {'N': N, 'D': D, 'E': E, 'e': e, 'X': X, 'y': Y}
    # Stan model object
    sm = pystan.StanModel(model_code=model_description)
    fit = sm.sampling(data=data, iter=2000, chains=4, verbose=True,
                      algorithm="NUTS", seed=seed,
                      control=dict(adapt_delta=0.95, max_treedepth=15))
    return sm, fit


# Invariance tests
def pooling_factor(sample_beta, sample_mu, E: int):
    """
    Parameters
    ----------
    sample_beta:
    sample_mu: 
    E : int
  """
    # difference matrix Nx
    delta = [sample_beta[:, e] - sample_mu for e in range(E)]

    # For each environment: compute expectation and variance of difference
    exp_diff = [np.mean(delta[e]) for e in range(E)]
    var_diff = [np.var(delta[e]) for e in range(E)]

    # Compute variance of expected difference and expected variance of difference
    var_exp_diff = np.var(exp_diff)
    exp_var_diff = np.mean(var_diff)

    # pooling factor
    lambda_pool = 1 - var_exp_diff / exp_var_diff
    return lambda_pool


def estimate_hdi(sample, alpha):
    N = len(sample)
    hdi_upper = N - 1
    hdi_lower = 1
    sorted_sample = sorted(sample)
    # number of samples to exclude from 1-alpha interval
    N_exclude = round(alpha * N)

    while (N_exclude > 0):

        diff_low = abs(sorted_sample[hdi_lower] - sorted_sample[hdi_lower + 1])
        diff_up = abs(sorted_sample[hdi_upper] - sorted_sample[hdi_lower - 1])

        if diff_low == diff_up:
            hdi_lower += 1
            hdi_upper -= 1
            N_exclude -= 2

        if diff_low > diff_up:
            hdi_lower += 1
            N_exclude -= 1

        if diff_low < diff_up:
            hdi_upper -= 1
            N_exclude -= 1

    return sorted_sample[hdi_lower], sorted_sample[hdi_upper]


def hdi_rope_test(sample, margin, alpha):
    ## Credible Interval
    q_low = alpha * 0.5
    CI_low = np.quantile(sample, q_low)
    CI_high = np.quantile(sample, 1 - q_low)
    # CI_low, CI_high = estimate_hdi(sample, alpha)

    # ROPE interval
    rope_low = -1 * margin
    rope_high = margin

    # Decision
    if (CI_low > rope_high) or (CI_high < rope_low):
        return 'Rejected'
    elif (CI_low >= rope_low) and (CI_high <= rope_high):
        return 'Accepted'
    else:
        return "Undecided"


def bhip_test(fit, scm, alpha_beta=0.05, alpha_mu=0.05, global_rope=0):
    D = fit.data["D"]
    E = fit.data["E"]
    Y = fit.data["y"]
    mu = fit.extract()['mu']  # matrix N x D
    beta = fit.extract()['beta']  # N x D x E

    # true parental/child set
    tp = set([i for i in range(len(scm.W)) if scm.W[i][0] != 0])
    tc = set([i for i in range(len(scm.W)) if scm.W[0][i] != 0])
    IP = list()
    rope = [0.1 * np.std(Y[Y.index == (e + 1)]) for e in range(E)]
    print(rope)
    for d in range(D):
        print(f'---------beta_{d + 1}---------')
        invariant = 0
        # Test for zero estimates

        zero_test_beta = [hdi_rope_test(beta[:, d, e], rope[0], alpha_beta) for e in range(E)]
        print(zero_test_beta)
        if global_rope == 0:
            zero_test_mu = hdi_rope_test(mu[:, d], np.mean(rope), alpha_mu)
            print(zero_test_mu)
        else:
            zero_test_mu = hdi_rope_test(mu[:, d], global_rope, alpha_mu)
        if zero_test_mu != "Rejected":
            invariant += 1
        if invariant == 0:
            IP.append(d+1)

        # Pooling factor
        pooling = pooling_factor(beta[:, d, :], mu[:, d], E)
        if pooling < 0.5:
            invariant += 1
        for e in range(E):
            if zero_test_beta[e] != "Rejected":
                invariant += 1
        print(pooling)

    return tp, tc, set(IP)


def data_to_pd(data):
    # Combine the data into a single matrix with an extra column indicating the environment
    df = pd.DataFrame()
    for e, X in enumerate(data):
        df_e = pd.DataFrame(X, index=np.repeat(e + 1, len(X)))
        df = pd.concat([df, df_e])
    df.columns = ['Y'] + ['X' + str(col + 1) for col in range(df.shape[1] - 1)]
    return df


# # ## Testing ##
# # # Fit
# N_scm = 5
# S = [0]
# S1 = [0, 2, 5]
# S2 = list(range(20,40,4))
# S3 = list(range(6,20,3))
#
# S4 = list(set(range(30, 50)) - set(S1 + S2 + S3))
#
# N_data = 5
# if __name__ == '__main__':
#     list_of_fits = list()
#     for i in S4:
#         filename_scm = 'data/experimentA/sim_vs2_' + str(i) + '.pkl'
#         with open(filename_scm, 'rb') as inp:
#             scenario = pickle.load(inp)
#         print('scenario', i)
#         # Simulated SCM
#         scm = scenario[0]
#         fits = list()
#         for j in range(N_data):
#             print(i, j)
#             df = data_to_pd(scenario[2][j])
#             # fit data with stan
#             SM, F = bhip_fit(df, model_NC, 101)
#             fits.append([SM, F])
#         list_of_fits.append(fits)
#
# with open('data/experimentA/bhip_r20_50_compl.pkl', 'wb') as outp:  # Overwrites any existing file.
#     pickle.dump(list_of_fits, outp, pickle.HIGHEST_PROTOCOL)
#
# S = list()
# for i in range(N_scm*2):
#     filename = 'data/experimentA/sim_vs2_' + str(i) + '.pkl'
#     with open(filename, 'rb') as inp:
#         scen = pickle.load(inp)
#         S.append(scen)
#
# filename_scm = 'data/experimentA/bhip_0_2_5.pkl'
# with open(filename_scm,'rb') as inp:
#     fit00 = pickle.load(inp)


# f = pickle.load('data/experimentA/b_0to5_0to5.pkl',/, *, fix_imports=True, encoding="ASCII", errors="strict", buffers=None)
#
# model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
# model = pystan.StanModel(model_code=model_code) # this will take a minute
# y = model.sampling(n_jobs=1).extract()['y']
# y.mean() # should be close to 0

filename = 'data/experimentA/icp_results.pkl'
with open(filename,'rb') as inp:
    r = pickle.load(inp)

list_of_fits = list()
filename = 'data/experimentA/bhip_0_2_5.pkl'
with open(filename,'rb') as inp:
    fits1 = pickle.load(inp)
list_of_fits = list_of_fits + fits1

# filename = 'data/experimentA/bhip_r6_20_3.pkl'
# with open(filename,'rb') as inp:
#     fits2 = pickle.load(inp)
# list_of_fits = list_of_fits + fits2

filename = 'data/experimentA/bhip_r20_40_4.pkl'
with open(filename,'rb') as inp:
    fits3 = pickle.load(inp)
list_of_fits = list_of_fits + fits3

filename = 'data/experimentA/bhip_r0_20_compl.pkl'
with open(filename ,'rb') as inp:
    fits4 = pickle.load(inp)
list_of_fits = list_of_fits + fits4

filename = 'data/experimentA/bhip_r20_50_compl.pkl'
with open(filename,'rb') as inp:
    fits5 = pickle.load(inp)
list_of_fits = list_of_fits + fits5

S1 = [0, 2, 5]
S2 = list(range(20,40,4))
S3 = list(range(6,20,3))
#S3 = []
S4 = list(set(range(0, 30)) - set(S1 + S2 + S3))
S5 = list(set(range(30, 50)) - set(S1 + S2 + S3))

scenarios = S1 + S2 + S4 + S5
p_succes = list()
p_false_select = list()
p_sub_succes = list()

def result_probs(true_set,est_sets):
    n_ests = len(est_sets)
    p_succes = sum([est is not None and true_set == est for est in est_sets])/n_ests
    p_sub_succes = sum([est is not None and true_set != est and len(est.intersection(true_set)) != 0 for est in est_sets]) / n_ests
    p_false_select = sum([est is not None and len(est-true_set) != 0 for est in est_sets]) / n_ests
    return p_succes, p_sub_succes, p_false_select


scenarios = S1 + S2  + S4 + S5
probs = list()
b_ests = list()
i_ests = list()
TP = list()
k = 0
for scenario in scenarios:
    filename_scm = 'data/experimentA/sim_vs2_' + str(scenario) + '.pkl'
    with open(filename_scm, 'rb') as inp:
        s = pickle.load(inp)
    scm = s[0]
    bhip_est = list()
    for i in range(5):
        tp, tc, est = bhip_test(list_of_fits[k][i][1],scm,alpha_mu =0.11)
        bhip_est.append(est)

    TP.append(tp)

    icp_est = r[scenario][2][0:5]

    i_ests.append(icp_est)
    b_ests.append(bhip_est)
    probs.append([result_probs(tp,icp_est), result_probs(tp,bhip_est)])
    k += 1
    print(probs)
    # print(b_ests)

import matplotlib.pyplot as plt
labels = ['S' + str(i) for i in scenarios]
bp1 = [probs[s][1][0] for s in range(len(labels))]
ip1 = [probs[s][0][0] for s in range(len(labels))]

bp2 = [probs[s][1][1] for s in range(len(labels))]
ip2 = [probs[s][0][1] for s in range(len(labels))]

bp3 = [probs[s][1][2] for s in range(len(labels))]
ip3 = [probs[s][0][2] for s in range(len(labels))]

x_icp, x_bhip = np.random.normal(1, 0.1, len(ip1)), np.random.normal(1.5, 0.1, len(bp1))

fig, axs = plt.subplots(3)
fig.suptitle(' Replication of ICP experiment (small) ')
axs[0].plot(x_icp, ip1,'ro',  label='ICP')
axs[0].plot(x_bhip, bp1,'bo', label = 'BHIP')
for i in range(len(ip1)):
    axs[0].plot([x_icp[i], x_bhip[i]], [ip1[i], bp1[i]],'--',color='grey')
axs[0].set_title("Probability of estimating true causal set")
axs[0].legend()
axs[1].plot(x_icp, ip2,'ro', label='ICP')
axs[1].plot(x_bhip, bp2,'bo', label = 'BHIP')
for i in range(len(ip1)):
    axs[1].plot([x_icp[i], x_bhip[i]], [ip2[i], bp2[i]],'--',color='grey')
axs[1].set_title("Probability of partially correct estimation")

axs[2].plot(x_icp, ip3,'ro', label='ICP')
axs[2].plot(x_bhip, bp3,'bo', label = 'BHIP')
for i in range(len(ip1)):
    axs[2].plot([x_icp[i], x_bhip[i]], [ip3[i], bp3[i]],'--',color='grey')
axs[2].set_title("Probability of erroneous selections")
axs[0].legend()
plt.show()

