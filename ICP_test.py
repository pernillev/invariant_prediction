# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 16:41:38 2021

@author: Perni
"""


import numpy as np
import pandas as pd
import pickle
import causalicp


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


def ICP_expA(seed):
    #Import SCM
    filename_scm = 'data/experimentA/scm' + str(seed) + '.pkl'
    with open(filename_scm, 'rb') as inp:
        SCM = pickle.load(inp)
    
    true_pa = set([i for i in range(len(SCM.W)) if SCM.W[0][i]>0])
    true_ch = set([i for i in range(len(SCM.W)) if SCM.W[i][0]>0])
    
    #Import data
    filename_df = 'data/experimentA/df' + str(seed)
    dataframe = pd.read_csv(filename_df, index_col=0)
    environments = dataframe.index.unique().to_list()
    data = list()
    for e in environments:        
        data.append(dataframe[dataframe.index == e].to_numpy())
    ICP = causalicp.fit(data, 0, alpha=0.05, sets=None, precompute=True, verbose=True, color=True)
    estimate = ICP.estimate
    
    
    return true_pa,true_ch,estimate
    # return true_pa,true_ch,data

pa,ch,est = ICP_expA(11)



