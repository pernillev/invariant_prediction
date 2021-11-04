# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:47:30 2021

@author: Perni

We sample nint data points from an interventional
setting (jEj = 2).



"""

import random
import numpy as np
import sempler
import sempler.generators
import pandas as pd


####### Functions #########
def generate_scms(G, p_min, p_max, k_min, k_max, w_min, w_max, m_min, m_max, v_min, v_max, random_state=None):
    """
    Generate random experimental cases (ie. linear SEMs). Parameters:
      - G: total number of cases
      - p_min, p_max: number of variables in the SCMs sampled at uniform between p_min and p_max
      - k_min, k_max: average node degree sampled at uniform between p_min and p_max
      - w_min, w_max: Weights of the SCMs are sampled at uniform between w_min and w_max
      - v_min, v_max: Variances of the variables are sampled at uniform between v_min and v_max
      - m_min, m_max: Intercepts of the variables of the SCMs are sampled at uniform between m_min and m_max
      - random_state: to fix the random seed for reproducibility
    """
    if random_state is not None:
        random.seed(random_state)
    cases = list()
    while len(cases) < G:
        p = np.random.randint(p_min, p_max) if p_min != p_max else p_min
        k = np.random.randint(k_min, k_max) if k_min != k_max else k_min
        W = sempler.generators.dag_avg_deg(p, k, w_min, w_max)
        W *= np.random.choice([-1, 1], size=W.shape)
        scm = sempler.LGANM(W, (m_min, m_max), (v_min, v_max), )
        cases.append(scm)
    return cases


def generate_shift_interventions(scm, no_ints, int_size, m_min, m_max, v_min, v_max, include_obs=False, exclude_target=False):
    """Generate a set of shift interventions for a given scm, randomly sampling
    no_ints sets of targets of size int_size, and sampling the
    intervention means/variances uniformly from [m_min,m_max], [v_min, v_max].

    If include_obs is True, include an empty intervention to represent
    the reference or observational environment.

    If exclude_target is True, exclude target X = 0 from intervention
    """
    interventions = [None] if include_obs else []
    # For each intervention
    for _ in range(no_ints):
        # sample targets
        if exclude_target:
            # without intervening on  node 0 (= target Y)
            if int_size == scm.p:
                int_size -= 1
            targets = np.random.choice(range(1, scm.p), int_size, replace=False)
        else:
            targets = np.random.choice(range(scm.p), int_size, replace=False)
        # sample parameters
        means = np.random.uniform(m_min, m_max, len(targets)) if m_min != m_max else [m_min] * len(targets)
        variances = np.random.uniform(v_min, v_max, len(targets)) if v_min != v_max else [v_min] * len(targets)
        # assemble intervention
        intervention = dict([(t, (mean, var)) for (t, mean, var) in zip(targets, means, variances)])
        interventions.append(intervention)
    return interventions


# Simulation functions

def transform(N, sample, chance_N, chance_S, scale):
    # transform a sample with noise intervention and/or scaling
    n = random.randint(1, chance_N)
    s = random.randint(1, chance_S)
    if n == 1:
        sample = np.random.normal(0, 1, N)
    if s == 1:
        sample = sample * np.random.uniform(0.5, scale)
    return sample


def simulate_sample(N, e_index, chance_N, chance_S, scale):
    """Simulate a sample, generated from a SCM with 
    random noise interventions and/or scaling.

    Parameters
    ----------
    N : int, sample size
    e_index : int, environment index number
    chance_N: int, one in chance_N chance of a node getting noise intervened
    chance_S: int, one in chance_S chance of a node getting scaled
    scale: int, scale with a random number between 0.5 ans "scale" 
    Returns
    -------
    panda dataframe, a dataframe with columns corresponding for each node
    """

    ## X1
    X1 = np.random.normal(0, 1, N)
    X1 = transform(N, X1, chance_N, chance_S, scale)

    ## X2
    X2 = -2 * X1 + np.random.normal(0, 1, N)
    X2 = transform(N, X2, chance_N, chance_S, scale)

    ## X3
    X3 = np.random.normal(0, 1, N)
    X3 = transform(N, X3, chance_N, chance_S, scale)

    ## X4
    X4 = np.random.normal(0, 1, N)
    X4 = transform(N, X4, chance_N, chance_S, scale)

    ## X5 parent of child of Y 
    X5 = np.random.normal(0, 1, N)
    X5 = transform(N, X5, chance_N, chance_S, scale)

    ## Y 
    Y = X2 + X3 + np.random.normal(0, 1, N)

    ## X6 child of Y and X5
    X6 = -0.3 * Y + X5 + np.random.normal(0, 1, N)
    X6 = transform(N, X6, chance_N, chance_S, scale)

    df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y,
                       'X3': X3, 'X4': X4, 'X5': X5,
                       'X6': X6}, index=np.repeat(e_index, N))

    return df


def simulate(environments, sample_sizes, chance_N, chance_S, scale, rseed):
    """Simulate several dataset with.
    
    Parameters
    ----------
    environments : list,  list of number of environments,
    sample_sizes : list,  list, list of sample sizes
    chance_N: int, one in chance_N chance of a node getting noise intervened
    chance_S: int, one in chance_S chance of a node getting scaled
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
                sample = simulate_sample(N, e, chance_N, chance_S, scale)
                df = pd.concat([df, sample])

        # Append df to list of dataframes 
        list_of_df.append(df)
    return list_of_df
