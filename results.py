import pickle
from invariant_prediction.BHIP import invariance_tests
import matplotlib.pyplot as plt
import numpy as np

import os
cwd = os.getcwd()  # Get the current working directory (cwd)



icp_results = list()
filename = 'data/experimentA/ICP_results_simA_0_15.pkl'
with open(filename, 'rb') as inp:
    icp_results = icp_results + pickle.load(inp)

filename = 'data/experimentA/ICP_results_simA_15_25.pkl'
with open(filename, 'rb') as inp:
    icp_results = icp_results + pickle.load(inp)

filename = 'data/experimentA/ICP_results_simA_25_35.pkl'
with open(filename, 'rb') as inp:
    icp_results = icp_results + pickle.load(inp)

list_of_fits = list()
filename = 'data/experimentA/bhip_results0_2.pkl'
with open(filename, 'rb') as inp:
    list_of_fits = list_of_fits + pickle.load(inp)

filename = 'data/experimentA/bhip_results2_8.pkl'
with open(filename, 'rb') as inp:
    list_of_fits = list_of_fits + pickle.load(inp)

filename = 'data/experimentA/bhip_results8_15.pkl'
with open(filename, 'rb') as inp:
    list_of_fits = list_of_fits + pickle.load(inp)

filename = 'data/experimentA/bhip_results15_20.pkl'
with open(filename, 'rb') as inp:
    list_of_fits = list_of_fits + pickle.load(inp)

filename = 'data/experimentA/bhip_results20_25.pkl'
with open(filename, 'rb') as inp:
    list_of_fits = list_of_fits + pickle.load(inp)

filename = 'data/experimentA/bhip_results25_30.pkl'
with open(filename, 'rb') as inp:
    list_of_fits = list_of_fits + pickle.load(inp)

filename = 'data/experimentA/bhip_results30_35.pkl'
with open(filename, 'rb') as inp:
    list_of_fits = list_of_fits + pickle.load(inp)

N_cases = 34
N_samples = 10
list_of_scm = list()
for i in range(N_cases):
    filename_scm = 'data/experimentA/simulation_A' + str(i) + '.pkl'
    with open(filename_scm, 'rb') as inp:
        scenario = pickle.load(inp)
    # Simulated SCM
    list_of_scm.append(scenario[0])

p_succ = list()
p_false = list()
p_sub = list()


def result_probs(true_set, est_sets):
    n_ests = len(est_sets)
    p_succ = sum([est is not None
                  and true_set == est
                  for est in est_sets]) / n_ests
    p_sub = sum([est is not None
                 and true_set != est
                 and len(est.intersection(true_set)) != 0
                 for est in est_sets]) / n_ests
    p_false = sum([est is not None
                   and len(est - true_set) != 0
                   for est in est_sets]) / n_ests
    return p_succ, p_sub, p_false


def false_types(est_sets, child, pachild):
    false_type = [est is not None
                  and len(est) != 0
                  and len(est.intersection(child.union(pachild))) != 0
                  for est in est_sets]
    return false_type


# # scenarios = S1 + S2  + S4 + S5
probs = list()
b_ests = list()
i_ests = list()
TP = list()
TC = list()
CP = list()
FT = list()
k = 0
Ds =list()
for case in range(1, N_cases):
    scm = list_of_scm[case]
    Ds.append(len(scm.W))
    tp = set([i for i in range(len(scm.W)) if scm.W[i][0] != 0])
    tc = set([i for i in range(len(scm.W)) if scm.W[0][i] != 0])
    cp = set()
    for child in tc:
        cp = cp.union(set([i for i in range(1, len(scm.W)) if scm.W[i][child] != 0]))
    CP.append(cp)
    TP.append(tp)
    TC.append(tc)
    case_fits = list_of_fits[case]
    bhip_est = list()
    for i in range(N_samples):
        mu_pass, beta_pass, pool_pass = invariance_tests(fit=case_fits[i][1],
                                                         alpha_beta=0.05, local_rope='two_sd',
                                                         alpha_mu=0.11, global_rope="two_sd",
                                                         p_thres=0.5, printing=True)
        #est = set(mu_pass + beta_pass + pool_pass)
        est = set(mu_pass).intersection(set(beta_pass), set(pool_pass))
        bhip_est.append(est)

    icp_est = icp_results[case][2][0:N_samples]

    i_ests.append(icp_est)
    b_ests.append(bhip_est)
    probs.append([result_probs(tp, icp_est), result_probs(tp, bhip_est)])
    FT.append([false_types(icp_est,tc,cp), false_types(bhip_est,tc,cp)])
    k += 1
    print(probs)
    print(b_ests)

#
# #
# scenarios = S
labels = ['S' + str(i) for i in range(1, N_cases)]
bp1 = [probs[s][1][0] for s in range(len(labels))]
ip1 = [probs[s][0][0] for s in range(len(labels))]

bp2 = [probs[s][1][1] for s in range(len(labels))]
ip2 = [probs[s][0][1] for s in range(len(labels))]

bp3 = [probs[s][1][2] for s in range(len(labels))]
ip3 = [probs[s][0][2] for s in range(len(labels))]

x_icp = np.random.uniform(1, 1.35, len(ip1))
x_bhip = x_icp + 0.5

fig, axs = plt.subplots(3,figsize=(5,8))
fig.suptitle(' Replication of ICP experiment (small scale) ')
axs[0].plot(x_icp, ip1, 'bo', label='ICP', ms = 2.5)
axs[0].plot(x_bhip, bp1, 'go', label='BHIP', ms = 2.5)
for i in range(len(ip1)):
    axs[0].plot([x_icp[i], x_bhip[i]], [ip1[i], bp1[i]], '--', color='grey',lw = 0.5)
axs[0].set_title("Probability of estimating the true causal set")

axs[1].plot(x_icp, ip2, 'bo', label='ICP', ms = 2.5)
axs[1].plot(x_bhip, bp2, 'go', label='BHIP', ms = 2.5)
for i in range(len(ip1)):
    axs[1].plot([x_icp[i], x_bhip[i]], [ip2[i], bp2[i]], '--', color='grey',lw = 0.5)
axs[1].set_title("Probability of an partially correct estimation")
axs[1].legend()
axs[2].plot(x_icp, ip3, 'bo', label='ICP', ms = 2.5)
axs[2].plot(x_bhip, bp3, 'go', label='BHIP', ms = 2.5)
for i in range(len(ip1)):
    axs[2].plot([x_icp[i], x_bhip[i]], [ip3[i], bp3[i]], '--', color='grey',lw = 0.5)
axs[2].set_title("Probability of erroneous selections")
# remove the x and y ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_ylim([-0.05, 1.05])
plt.show()
