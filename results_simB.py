import pickle
from invariant_prediction.BHIP import invariance_tests
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

r_result = pd.read_csv('data/experimentB/answer4')


def reorder(r_results):
    lst1 = []
    k = 0
    for i in range(12):
        lst2 = []
        for j in range(15):

            if r_results[k] == "empty" or r_results[k] == "None":
                r_set = set([])
            else:
                str_lst = list(r_results[k].split(" "))
                int_lst = [int(elm) for elm in str_lst]
                r_set = set(int_lst)
            lst2.append(r_set)
            if k < 178:
                k += 1
        lst1.append(lst2)
    return lst1


icp_results = reorder(r_result.answer.values)

filename = 'data/experimentB/fits_33.pkl'
with open(filename, 'rb') as inp:
    list_of_fits = pickle.load(inp)

filename = 'data/experimentB/scm_3.pkl'
with open(filename, 'rb') as inp:
    list_of_scm = pickle.load(inp)


# p_succ = list()
# p_false = list()
# p_sub = list()

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

N_cases = 12
N_samples = 15
for case in range(N_cases):
    scm = list_of_scm[case]
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
    for i in range(15):
        mu_pass, beta_pass, pool_pass = invariance_tests(fit=case_fits[i][1],
                                                         alpha_beta=0.05, local_rope="two_sd",
                                                         alpha_mu=0.11, global_rope="two_sd",
                                                         p_thres=0.5, printing=False)
        est = set(mu_pass).intersection(set(beta_pass), set(pool_pass))
        # est = set(mu_pass + beta_pass + pool_pass)
        bhip_est.append(est)

    icp_est = icp_results[case]

    i_ests.append(icp_est)
    b_ests.append(bhip_est)
    probs.append([result_probs(tp, icp_est), result_probs(tp, bhip_est)])
    FT.append([false_types(icp_est, tc, cp), false_types(bhip_est, tc, cp)])
    k += 1
    print(probs)
    print(b_ests)



# USELESS?
# scenarios = S
# # labels = ['S' + str(i) for i in range(N_cases)]
# bp1 = [probs[s][1][0] for s in range(len(labels))]
# ip1 = [probs[s][0][0] for s in range(len(labels))]
#
# bp2 = [probs[s][1][1] for s in range(len(labels))]
# ip2 = [probs[s][0][1] for s in range(len(labels))]
#
# bp3 = [probs[s][1][2] for s in range(len(labels))]
# ip3 = [probs[s][0][2] for s in range(len(labels))]
#
# x_icp = np.random.uniform(1, 1.35, len(ip1))
# x_bhip = x_icp + 0.5
#
# fig, axs = plt.subplots(3, figsize=(5, 8))
# fig.suptitle(' Replication of ICP experiment (small) ')
# axs[0].plot(x_icp, ip1, 'r0', label='ICP', ms=2.5)
# axs[0].plot(x_bhip, bp1, 'bo', label='BHIP', ms=2.5)
# for i in range(len(ip1)):
#     axs[0].plot([x_icp[i], x_bhip[i]], [ip1[i], bp1[i]], '--', color='grey', lw=0.5)
# axs[0].set_title("Probability of estimating true causal set")
# axs[0].legend()
# axs[1].plot(x_icp, ip2, 'ro', label='ICP', ms=2.5)
# axs[1].plot(x_bhip, bp2, 'bo', label='BHIP', ms=2.5)
# for i in range(len(ip1)):
#     axs[1].plot([x_icp[i], x_bhip[i]], [ip2[i], bp2[i]], '--', color='grey', lw=0.5)
# axs[1].set_title("Probability of partially correct estimation")
#
# axs[2].plot(x_icp, ip3, 'ro', label='ICP', ms=2.5)
# axs[2].plot(x_bhip, bp3, 'bo', label='BHIP', ms=2.5)
# for i in range(len(ip1)):
#     axs[2].plot([x_icp[i], x_bhip[i]], [ip3[i], bp3[i]], '--', color='grey', lw=0.5)
# axs[2].set_title("Probability of erroneous selections")
# axs[0].legend()
# # remove the x and y ticks
# for ax in axs:
#     ax.set_xticks([])
#     ax.set_ylim([-0.05, 1.05])
# plt.show()

#
## Experiment B PLots
#scenarios = S
labels = ['S' + str(i) for i in range(N_cases)]
bp1 = [probs[s][1][0] for s in range(len(labels))]
ip1 = [probs[s][0][0] for s in range(len(labels))]

bp2 = [probs[s][1][1] for s in range(len(labels))]
ip2 = [probs[s][0][1] for s in range(len(labels))]

bp3 = [probs[s][1][2] for s in range(len(labels))]
ip3 = [probs[s][0][2] for s in range(len(labels))]

x_icp = np.linspace(1, 1.35, len(ip1))
x_bhip = x_icp + 0.5

fig, axs = plt.subplots(3, figsize=(5, 8))
#fig.suptitle(' Replication of ICP experiment (small) ')
axs[0].plot(x_icp[0:3], ip1[0:3], 'b.', label='ICP: D=4', ms=4)
axs[0].plot(x_icp[3:5], ip1[3:5], 'b+', label='ICP: D=9', ms=4)
axs[0].plot(x_icp[5:10], ip1[5:10], 'bo', label='ICP: D=49', ms=4)
axs[0].plot(x_icp[10:12], ip1[10:12], 'bx', label='ICP: TP=Ø', ms=4)
axs[0].plot(x_bhip[0:3], bp1[0:3], 'g.', label='BHIP D=4', ms=4)
axs[0].plot(x_bhip[3:5], bp1[3:5], 'g+', label='BHIP', ms=4)
axs[0].plot(x_bhip[5:10], bp1[5:10], 'go', label='BHIP', ms=4)
axs[0].plot(x_bhip[10:12], bp1[10:12], 'gx', label='BHIP', ms=4)
for i in range(len(ip1)):
    axs[0].plot([x_icp[i], x_bhip[i]], [ip1[i], bp1[i]], '--', color='grey', lw=0.5)
axs[0].set_title("Probability of estimating the true causal set")
axs[1].plot(x_icp[0:3], ip2[0:3], 'b.', label='ICP: D = 5', ms=4)
axs[1].plot(x_icp[3:5], ip2[3:5], 'b+', label='ICP: D = 5', ms=4)
axs[1].plot(x_icp[5:10], ip2[5:10], 'bo', label='ICP: D = 5', ms=4)
axs[1].plot(x_icp[10:12], ip2[10:12], 'bx', label='ICP: TP= Ø', ms=4)
axs[1].plot(x_bhip[0:3], bp2[0:3], 'g.', label='BHIP', ms=4)
axs[1].plot(x_bhip[3:5], bp2[3:5], 'g+', label='BHIP', ms=4)
axs[1].plot(x_bhip[5:10], bp2[5:10], 'go', label='BHIP', ms=4)
axs[1].plot(x_bhip[10:12], bp2[10:12], 'gx', label='BHIP', ms=4)
for i in range(len(ip2)):
    axs[1].plot([x_icp[i], x_bhip[i]], [ip2[i], bp2[i]], '--', color='grey', lw=0.5)
axs[1].set_title("Probability of a partially correct estimation")
axs[2].plot(x_icp[0:3], ip3[0:3], 'b.', label='ICP: D=4', ms=4)
axs[2].plot(x_icp[3:5], ip3[3:5], 'b+', label='ICP: D=9', ms=4)
axs[2].plot(x_icp[5:10], ip3[5:10], 'bo', label='ICP: D=49', ms=4)
axs[2].plot(x_icp[10:12], ip3[10:12], 'bx', label='ICP: TP=Ø', ms=4)
axs[2].plot(x_bhip[0:3], bp3[0:3], 'g.', label='BHIP: D=4', ms=4)
axs[2].plot(x_bhip[3:5], bp3[3:5], 'g+', label='BHIP: D=9', ms=4)
axs[2].plot(x_bhip[5:10], bp3[5:10], 'go', label='BHIP: D=49', ms=4)
axs[2].plot(x_bhip[10:12], bp3[10:12], 'gx', label='BHIP: TP=Ø', ms=4)
for i in range(len(ip1)):
    axs[2].plot([x_icp[i], x_bhip[i]], [ip3[i], bp3[i]], '--', color='grey', lw=0.5)
axs[2].set_title("Probability of erroneous selections")
axs[2].legend(loc='lower center', bbox_to_anchor=(0.5, -0.6),
              ncol=2, fancybox=True, shadow=True)
# remove the x and y ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_ylim([-0.05, 1.05])
plt.tight_layout()
plt.show()


I1 = [0,1,2]
I2 = [3,4,10]
I3 = [5,7,8]
I = I1+I2+I3
E4 = [[result_probs(TP[ind], i_ests[ind][0:5]), result_probs(TP[ind], b_ests[ind][0:5])]for ind in I]
E9 = [[result_probs(TP[ind], i_ests[ind][5:10]), result_probs(TP[ind], b_ests[ind][5:10])]for ind in I]
E49 = [[result_probs(TP[ind], i_ests[ind][10:15]), result_probs(TP[ind], b_ests[ind][10:15])]for ind in I]

#E49[I,index][icp or bhip][TP,Part,FP]

# width of the bars
barWidth = 0.05
# The x position of bars
r1 = np.arange(6)
r2 = [x + barWidth for x in r1]

r = [[0.2,2.2,4.2,1,3,5]]
fig1, ax1 = plt.subplots()
for ind in range(len(I)):
    ax1.bar(r[ind][0:3], [E4[ind][0][0]+0.01, E9[ind][0][0]+0.01, E49[ind][0][0]+0.01], width = 0.05, color = 'blue', edgecolor = 'black')
    ax1.bar(r[ind][3:6], [E4[ind][1][0]+0.01, E9[ind][1][0]+0.01, E49[ind][1][0]+0.01], width = 0.05, color = 'green', edgecolor = 'black')
    r.append([x + barWidth for x in r[ind]])
ax1.set_titlel("")
ax1.set_xlabel("Environment 3, 6, 12")

r = [[0.2,2.2,4.2,1,3,5]]
fig1, ax1 = plt.subplots()
r = 0
for ind in range(3):
    r = r + 0.05 + 0.03
    ax1.bar(r, E4[ind][0][0]+0.01, width = 0.05, color = 'blue', edgecolor = 'black')
    r = r + 0.05
    ax1.bar(r, E4[ind][1][0]+0.01, width = 0.05, color = 'green', edgecolor = 'black')
r = 0.5
for ind in range(3):
    r = r + 0.05 + 0.03
    ax1.bar(r, E9[ind][0][0]+0.01, width = 0.05, color = 'blue', edgecolor = 'black')
    r = r + 0.05
    ax1.bar(r, E9[ind][1][0]+0.01, width = 0.05, color = 'green', edgecolor = 'black')
r = 1
for ind in range(2):
    r = r + 0.05+ 0.03
    ax1.bar(r, E49[ind][0][0]+0.01, width = 0.05, color = 'blue', edgecolor = 'black')
    r = r + 0.05
    ax1.bar(r, E49[ind][1][0]+0.01, width = 0.05, color = 'green', edgecolor = 'black')
r = r + 0.05+ 0.03
ax1.bar(r, E49[3][0][0]+0.01, width = 0.05, color = 'blue', edgecolor = 'black', label='ICP')
r = r + 0.05
ax1.bar(r, E49[3][1][0]+0.01, width = 0.05, color = 'green', edgecolor = 'black', label='BHIP')

ax1.set_title("")
my_xticks = ['E=3','E=6','E=12']
plt.xticks([0.25,0.75,1.2], my_xticks)
ax1.set_ylabel("Probability of estimating true causal set")
ax1.legend(loc = 'upper center')
#ax1.set_xlabel("Environment 3, 6, 12")
## Scatter

# fig, axs = plt.subplots(1, figsize=(5, 8))
# plt.scatter(range(690),f.data['y'].values)
# plt.show()

# fig, axs = plt.subplots(1, figsize=(5, 8))
# plt.scatter(range(690),f.data['X']['X40'].values)
# plt.show()
#
#
# fig, axs = plt.subplots(1, figsize=(5, 8))
# sns.kdeplot(beta[:,1,0])
# plt.show()
