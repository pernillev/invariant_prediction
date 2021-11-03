# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 16:41:38 2021

@author: Perni
"""
import numpy as np
import pandas as pd
import random
import sempler
import causalicp
from sempler.generators import dag_avg_deg


## Generate random DAG
def pick_coef(lb, ub):
    coef = np.round(np.random.uniform(lb, ub, 1) * random.choice([-1, 1]), 2)
    return coef[0]


## Experiment A - ICP simulation replication
def generate_random_SCM(seed):
    # generate random quantities
    D = random.randint(5, 15)  # number of nodes
    deg = random.randint(1, 4)  # average degree of graph
    lb1 = random.uniform(0.1, 2)  # lower bound on linear coefficients
    ub1 = lb1 + random.uniform(0.1, 1)  # upper bound of coefficient values
    sigmas = np.random.uniform(0.1, 2, 2)  # two noise variance
    if sigmas[0] < sigmas[1]:
        sigma_min = sigmas[0]
        sigma_max = sigmas[1]
    else:
        sigma_min = sigmas[1]
        sigma_max = sigmas[0]

        # Generate a random DAG and construct a linear-Gaussian SCM
    W = dag_avg_deg(D, deg, random_state=seed)

    def set_coef(W):
        for node1 in range(D):
            for node2 in range(D):
                if (W[node1][node2] > 0):
                    W[node1][node2] = pick_coef(lb1, ub1)
        return (W)

    DAG_coefs = set_coef(W)
    # Geenerate SCM
    scm = sempler.LGANM(DAG_coefs, (0, 0), (sigma_min, sigma_max))

    return scm


def generate_dataframe(E, Ns, scm):
    # Sample observational and interventional data
    N_obs = random.choice(Ns)
    e = 1
    data_obs = scm.sample(N_obs)
    df = pd.DataFrame(data_obs, index=np.repeat(e, N_obs))

    for e in range(2, E + 1):
        N_int = random.choice(Ns)
        D = len(scm.W)

        # generate random quantities regarding interventions
        a_min = np.random.uniform(0.1, 4, D)
        a_Delta = [0 if (random.choice(range(3)) == 0)
                   else np.random.uniform(0.1, 2) for node in range(D)]
        a = a_min + a_Delta

        # sample indexset of nodes to intervene on (disregarding X0 = Y)
        inv_theta = [D if (random.choice(range(6)) == 0)
                     else np.random.uniform(1.1, 3)]
        A = random.sample(range(0, D), round(D / inv_theta[0]))  # subset can also contain Y

        # lower and upper bound of new coefficients
        coef_bounds = np.random.uniform(0.1, 2, 2)  # two noise variance
        if coef_bounds[0] < coef_bounds[1]:
            lbe = coef_bounds[0]
            ube = coef_bounds[1]
        else:
            lbe = coef_bounds[1]
            ube = coef_bounds[0]

        # Intervene on nodes in A
        scm_intervention = scm
        shift_dict = {}
        for node in A:
            # dictionary for shift intervention
            shift_dict[node] = (0, a[node])
            # potentially sample new coefficient
            if (random.choice(range(3)) == 0):
                new_coefs = [0 if scm.W[node][relation] == 0
                             else pick_coef(lbe, ube) for relation in range(D)]
                scm_intervention.W[node] = new_coefs

        # sample shift-intervention data for environment e
        data_int = scm_intervention.sample(N_int, shift_interventions=shift_dict)
        df_e = pd.DataFrame(data_int, index=np.repeat(e, N_int))
        df = pd.concat([df, df_e])
    return df


# # Import data 
# list_of_SCM = list()
# list_of_df = list()


# for i in range(2):

#     #Import SCM
#     filename_scm = 'data/experimentA/scm' + str(i) + '.pkl'
#     with open(filename_scm, 'rb') as inp:
#         SCM = pickle.load(inp)
#     list_of_SCM.append()

#     #Import data
#     filename_df = 'data/experimentA/df' + str(i)
#     dataframe = pd.read_csv(filename_df, index_col=0)
#     dataframe.columns = ['Y'] + ['X' + str(i) for i in range(1,dataframe.shape[1])]
#     list_of_df.append(dataframe)
list_of_SCM = list()
list_of_df = list()
# simulate and save random SCM and correponding data
for exp in range(500):
    seed = exp + 1000
    scm = generate_random_SCM(seed)
    list_of_SCM.append(scm)
    # simulate data
    E = 2
    Ns = [100, 200, 300, 400, 500]

    df = generate_dataframe(E, Ns, scm)
    # save data
    list_of_df.append(df)

results = list()

for index in range(50):
    SCM = list_of_SCM[index]
    # true parental set
    true_pa = set([i for i in range(len(SCM.W)) if SCM.W[0][i] > 0])
    # true children set
    true_ch = set([i for i in range(len(SCM.W)) if SCM.W[i][0] > 0])
    # Import data
    dataframe = list_of_df[index]
    environments = dataframe.index.unique().to_list()
    data = list()
    for e in environments:
        data.append(dataframe[dataframe.index == e].to_numpy())
    ICP = causalicp.fit(data, 0, alpha=0.05, sets=None, precompute=True, verbose=False, color=True)
    estimate = ICP.estimate
    print(index)
    results.append((true_pa, true_ch, estimate))
