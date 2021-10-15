# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:47:30 2021

@author: Perni
"""

import numpy as np
import pandas as pd
import random

#Simulation functions

def transform(N,sample,chance_N,chance_S,scale):
    # transform a sample with noise intervention and/or scaling
    n = random.randint(1,chance_N)
    s = random.randint(1,chance_S)
    if n == 1:
      sample = np.random.normal(0,1,N)
    if s == 1:
        sample = sample*np.random.uniform(0.5, scale)
    return sample

def simulation(N,e_index,chance_N,chance_S,scale):
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
    X1 = np.random.normal(0,1,N)
    X1 = transform(N,X1,chance_N,chance_S,scale)

      ## X2
    X2 = -2*X1 + np.random.normal(0,1,N)
    X2 = transform(N,X2,chance_N,chance_S,scale)
      
    ## X3
    X3 = np.random.normal(0,1,N)
    X3 = transform(N,X3,chance_N,chance_S,scale)

    ## X4
    X4 = np.random.normal(0,1,N)
    X4 = transform(N,X4,chance_N,chance_S,scale)

    ## X5 parent of child of Y 
    X5 = np.random.normal(0,1,N)
    X5 = transform(N,X5,chance_N,chance_S,scale) 
  
    ## Y 
    Y = X2 + X3 + np.random.normal(0,1,N) 
  
    ## X6 child of Y and X5
    X6 = -0.3*Y + X5 + np.random.normal(0,1,N)
    X6 = transform(N,X6,chance_N,chance_S,scale)

    df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y ,
                       'X3': X3, 'X4': X4, 'X5': X5, 
                       'X6': X6}, index = np.repeat(e_index,N))

    return df


def simulate_environments(environments,sample_sizes,chance_N,chance_S,scale,rseed):
        
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
    #Set seed   
    np.random.seed(rseed)
    for E in environments:
        for N in sample_sizes:
            # first sample, no inteventions
            df = simulation(1,N,1000,1000,5,101)      
            for e in range(2,E+1):
                # simulte sample
                sample = simulation(N,e+2,chance_N,chance_S,scale)
                df = pd.concat([df,sample])
              
        # Append df to list of dataframes 
        list_of_df.append(df)