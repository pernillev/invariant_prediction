import copy
import random
import pickle
import time
from datetime import datetime
import numpy as np
import pandas as pd

from invariant_prediction.simulate_functions import generate_scms
from invariant_prediction.simulate_functions import generate_interventions
from invariant_prediction.Models.stan_program_codes import model_NC
from invariant_prediction.BHIP_test import bhip_fit_exp_A
# Generate SCM and interventions
random.seed(1010)
N_environments = [3, 6, 12]
N_nodes = [5, 10, 50]
N_cases = 5
list_of_scm = list()
for p in N_nodes:
    scms = generate_scms(G=N_cases,
                         p_min=p, p_max=p,
                         k_min=1, k_max=5,
                         w_min=0.3, w_max=1.8,
                         m_min=0, m_max=0,
                         v_min=0.1, v_max=2,
                         random_state=101)
    list_of_scm.append(scms)
scms = [item for sublist in list_of_scm for item in sublist]
# Remove empty set parents
parents = [set([i for i in range(len(SCM.W)) if SCM.W[i][0] != 0]) for SCM in scms]
# ps = [scm.p for scm in scms]
lst_1 = [scms[i] for i in range(3*N_cases) if len(parents[i]) > 0]
lst_2 = [scms[i] for i in range(3*N_cases) if len(parents[i]) == 0]
listB1 = lst_1 + random.choices(lst_2, k=round(len(lst_1) * 0.25))
ps = [scm.p for scm in listB1]

def data_to_pd(data,p):
    # Combine the data into a single matrix with an extra column indicating the environment
     df = pd.DataFrame()
     for e, X in enumerate(data):
         df_e = pd.DataFrame(X, index=np.repeat(e + 1, len(X)))
         df = pd.concat([df, df_e])
     df.columns = ['Y'] + [f'X{i}' for i in range(1, p)]
     return df



list_of_df = list()
 Ns = [35, 65, 95]
 for case in range(len(listB1)):
     scm = copy.deepcopy(listB1[case])
     p = scm.p
     dfs = list()
     for E in N_environments:
         noise_int = generate_interventions(int(E-1), int(np.ceil(0.15*p)), int(np.ceil(0.50*p)),
                                            list(range(1, p)),
                                            m_min=0.5, m_max=2, v_min=1, v_max=4, seed=101)
         do_int = generate_interventions(int(E-1), int(np.ceil(0.15*p)), int(np.ceil(0.50*p)),
                                         list(range(1, p)),
                                         0, 0, 1, 1, seed=101)
         for _ in range(5):
             X = list()
             X.append(scm.sample(random.choice(Ns)))
             for e in range(E-1):
                 X.append(scm.sample(random.choice(Ns), do_interventions=do_int[e], noise_interventions=noise_int[e]))
             df = data_to_pd(X, p)
             dfs.append(df)
     list_of_df.append(dfs)

list_of_df3 = list()
filename = 'data/experimentB/df_3.pkl'

with open(filename, 'rb') as inp:
    list_of_df3  = pickle.load(inp)
list_of_scm3 = list()

filename = 'data/experimentB/scm_3.pkl'
with open(filename, 'rb') as inp:
    list_of_scm3 = pickle.load(inp)
# with open('data/experimentB/df_3.pkl', 'wb') as outp:  # Overwrites any existing file.
#     pickle.dump(list_of_df, outp, pickle.HIGHEST_PROTOCOL)
# #
# with open('data/experimentB/scm_3.pkl', 'wb') as outp:  # Overwrites any existing file.
#     pickle.dump(listB1, outp, pickle.HIGHEST_PROTOCOL)
# #
# #
list_of_fits = list()
if __name__ == '__main__':
    for i in range(12):
        # Simulated SCM
        fits = list()
        ldf = list_of_df3[i]
        for j in range(15):
            print(i, j)
            # fit data with stan
            SM, F = bhip_fit_exp_A(ldf[j], model_NC, 101)
            fits.append([SM, F])
        list_of_fits.append(fits)
# # # ## Dump
with open('data/experimentB/fits_33.pkl', 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(list_of_fits, outp, pickle.HIGHEST_PROTOCOL)


# # Generate data and save it
# def pick_coef(lb, ub):
#     coef = np.round(np.random.uniform(lb, ub, 1) * random.choice([-1, 1]), 2)
#     return coef[0]


# with open('data/experimentB/B_fit_0_10_2_pkl', 'wb') as outp:  # Overwrites any existing file.
#     pickle.dump(list_of_fits, outp, pickle.HIGHEST_PROTOCOL)
#
#

