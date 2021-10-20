# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 10:46:16 2021

@author: Pernille V. S.
"""
import numpy as np
import pandas as pd
import random
import sempler
from sempler.generators import dag_avg_deg
import pickle
## Generate random DAG
def pick_coef(lb,ub):
   coef = np.round(np.random.uniform(lb,ub,1)*random.choice([-1,1]),2)
   return coef[0]

## Experiment A - ICP simulation replication
def generate_random_SCM(seed):
    # generate random quantities
    D = random.randint(5,40) # number of nodes
    deg = random.randint(1,4) # average degree of graph
    lb1 = random.uniform(0.1,2) # lower bound on linear coefficients
    ub1 = lb1 + random.uniform(0.1,1) # upper bound of coefficient values
    sigmas = np.random.uniform(0.1,2,2) # two noise variance
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
                if (W[node1][node2]>0):
                    W[node1][node2] = pick_coef(lb1,ub1)
        return(W)
    DAG_coefs = set_coef(W)
    # Geenerate SCM
    scm = sempler.LGANM(DAG_coefs, (0,0), (sigma_min,sigma_max))
    
    return scm

def generate_dataframe(E,Ns,scm):
    # Sample observational and interventional data
    N_obs = random.choice(Ns)
    e = 1 
    data_obs = scm.sample(N_obs)
    df = pd.DataFrame(data_obs, index = np.repeat(e,N_obs))
    
    for e in range(2,E+1):
        N_int = random.choice(Ns)
        D = len(scm.W)
        
        # generate random quantities regarding interventions
        a_min = np.random.uniform(0.1,4,D)
        a_Delta = [0 if (random.choice(range(3))==0) 
                   else np.random.uniform(0.1,2) for node in range(D)]
        a = a_min + a_Delta
        
        # sample indexset of nodes to intervene on (disregarding X0 = Y)
        inv_theta = [D if (random.choice(range(6))==0) 
                     else np.random.uniform(1.1,3)]
        A = random.sample(range(0,D), round((D)/inv_theta[0])) 
        
        # lower and upper bound of new coefficients
        lbe = random.uniform(0.1,1) 
        ube = lbe + random.uniform(0.1,1)
        
        # Intervene on nodes in A
        scm_intervention = scm
        shift_dict = {}
        for node in A:
            # dictionary for shift intervention
            shift_dict[node] = (0,a[node])
            # potentially sample new coefficient     
            if (random.choice(range(3))==0): 
                new_coefs = [0 if scm.W[node][relation] == 0 
                             else pick_coef(lbe,ube) for relation in range(D)]
                scm_intervention.W[node] = new_coefs
            
        # sample shift-intervention data for environment e
        data_int = scm_intervention.sample(N_int, shift_interventions = shift_dict) 
        df_e = pd.DataFrame(data_int, index = np.repeat(e,N_int))
        df = pd.concat([df,df_e])
    return df
# %%
# simulate and save random SCM and correponding data
for exp in range(500):
    seed = exp + 1000
    scm = generate_random_SCM(seed)
    
    def save_object(obj, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    
    # save SCM
    filename = 'data/experimentA/scm' + str(exp) + '.pkl'
    save_object(scm, filename)

    #simulate data
    E  = 2
    Ns = [100,200,300,400,500]
    
    df = generate_dataframe(E,Ns,scm)
    # save data
    filename = 'data/experimentA/df' + str(exp) 
    df.to_csv(filename,index = True)    

# %%
## Experiment B

def transform(N, sample, inv_prob_N, inv_prob_S, scale):
    # transform a sample with noise intervention and/or scaling
    n = random.randint(1, inv_prob_N)
    s = random.randint(1, inv_prob_S)
    if n == 1:
        sample = np.random.normal(0, 1, N)
    if s == 1:
        sample = sample * np.random.uniform(0.5, scale)
    return sample


def simulate_sample(N, e_index, inv_prob_N, inv_prob_S, scale):
    """Simulate a sample, generated from a SCM with
    random noise interventions and/or scaling.

    Parameters
    ----------
    N : int, sample size
    e_index : int, environment index number
    inv_prob_N: int, one in inv_prob_N chance of a node getting noise intervened
    inv_prob_S: int, one in inv_prob_S chance of a node getting scaled
    scale: int, scale with a random number between 0.5 ans "scale"
    Returns
    -------
    panda dataframe, a dataframe with columns corresponding for each node
    """

    ## X1
    X1 = np.random.normal(0, 1, N)
    X1 = transform(N, X1, inv_prob_N, inv_prob_S, scale)

    ## X2
    X2 = -2 * X1 + np.random.normal(0, 1, N)
    X2 = transform(N, X2, inv_prob_N, inv_prob_S, scale)

    ## X3
    X3 = np.random.normal(0, 1, N)
    X3 = transform(N, X3, inv_prob_N, inv_prob_S, scale)

    ## X4
    X4 = np.random.normal(0, 1, N)
    X4 = transform(N, X4, inv_prob_N, inv_prob_S, scale)

    ## X5 parent of child of Y
    X5 = np.random.normal(0, 1, N)
    X5 = transform(N, X5, inv_prob_N, inv_prob_S, scale)

    ## Y
    Y = X2 + X3 + np.random.normal(0, 1, N)

    ## X6 child of Y and X5
    X6 = -0.3 * Y + X5 + np.random.normal(0, 1, N)
    X6 = transform(N, X6, inv_prob_N, inv_prob_S, scale)

    df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y,
                       'X3': X3, 'X4': X4, 'X5': X5,
                       'X6': X6}, index=np.repeat(e_index, N))

    return df


def simulate(environments, sample_sizes, inv_prob_N, inv_prob_S, scale, rseed):
    """Simulate several dataset with.

    Parameters
    ----------
    environments : list,  list of number of environments,
    sample_sizes : list,  list, list of sample sizes
    inv_prob_N: int, one in inv_prob_N chance of a node getting noise intervened
    inv_prob_S: int, one in inv_prob_S chance of a node getting scaled
    scale: int, scale with a random number between 0.5 ans "scale"
    rseed : int, random seed

    Returns
    -------
    list, a list of dataframes
    """

    # initialize list of dataframes
    list_of_df = list()
    # Set seed
    np.random.seed(rseed)
    for E in environments:
        for N in sample_sizes:
            # first sample, no inteventions
            df = simulate_sample(N, 1, 1000, 1000, 5)
            for e in range(2, E + 1):
                # simulte sample
                sample = simulate_sample(N, e, inv_prob_N, inv_prob_S, scale)
                df = pd.concat([df, sample])

        # Append df to list of dataframes
        list_of_df.append(df)
    return list_of_df
