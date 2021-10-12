# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:09:59 2021

@author: PernilleV
"""

import numpy as np
import pandas as pd
import random

dag_base_adj = pd.DataFrame(
    {'X1': np.zeros(7), 
     'X2': [-2,0,0,0,0,0,0], 
     'X3': np.zeros(7), 
     'X4': np.zeros(7), 
     'X5': np.zeros(7), 
     'Y': [0,1,1,0,0,0,0],
     'X6':[0,0,0,0,1,-0.3,0]}
    )

dag = dag_base_adj.to_numpy()

#Simulation functions

def transform(N,sample,noise_intervention,scale):
    # transforms sample with noise intervention and/or scaling
    if noise_intervention == 1:
      sample = np.random.normal(0,1,N)
    if scale > 0:
        sample = sample*np.random.uniform(0.1, scale)*np.random.choice((-1, 1))
    return sample


def simulation_dag(dag_adj,N,e_index,noise_intvs,scales):
    
    X = list()
    nodes = len(dag_adj)
    
    for i in range(nodes):
      #add random noise to sample for node i in DAG
      X.append(np.random.normal(0,1,N))
      # add influence from nodes j<i in DAG
      for j in range(i+1):
          X[i] = X[i] + dag_adj[j-1,i]*X[j-1]
      # transform node sample with noise intervention and/or scaling    
      X_trans = transform(N,X[i],noise_intvs[i],scales[i])
      X[i] = X_trans

    df = pd.DataFrame({'X1': X[0], 'X2': X[1], 
                     'X3': X[2], 'X4': X[3],
                     'X5': X[4], 'Y': X[5],
                     'X6': X[6]}, index = np.repeat(e_index,N))
    return df


dag_base_adj = pd.DataFrame(
    {'X1': np.zeros(7), 
     'X2': [-2,0,0,0,0,0,0], 
     'X3': np.zeros(7), 
     'X4': np.zeros(7), 
     'X5': np.zeros(7), 
     'Y': [0,1,1,0,0,0,0],
     'X6':[0,0,0,0,1,-0.3,0]}
    )

dag = dag_base_adj.to_numpy()

np.random.seed(101)
Es = [2,5,10]
Ns = [50,250,500]
list_of_df_A = list()


for env in Es:
  for size in Ns:
    E = env
    N = size

    df = simulation_dag(dag,N,1,[0,0,0,0,0,0,0],[0,0,0,0,0,0,0])
    for e in range(E-1):
      interventions = [random.randint(0,2) for i in range(7)]
      s = [random.randint(0,5) for i in range(7)]
      interventions[5] = 0
      s[5] = 0
      sample = simulation_dag(dag,N,e+2,interventions,s)
      df = pd.concat([df,sample])
    # Append df to list of dataframes
    list_of_df_A.append(df)
