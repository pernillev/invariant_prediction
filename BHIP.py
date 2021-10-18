# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 07:34:40 2021

@author: Pernille
"""

import numpy as np
import pandas as pd
import pystan
import pickle

# Models
# non centered reparamitrization
model_2 = """
data {
  int<lower=0> N;             // number of observations
  int<lower=1> D;             // number of covariates
  int<lower=1> E;             // number of environments
  int<lower=1,upper=E> e[N];  // associated environment
  matrix[N,D] X;              // covariate matrix
  vector[N] y;                // target vector
}

parameters {
  real mu[D];                 // population mean
  real<lower=0> tau[D];       // population scale
  matrix[D,E] gamma;          // Non-centered coefficients
}

transformed parameters {
  
  matrix[D,E] beta; // Recentered coefficients

  // Recentering             
  for (d in 1:D){
    for (i in 1:E){
      beta[d,i] = mu[d] + tau[d]*gamma[d,i];
    }
  }
}

model {
  for (d in 1:D){
    mu[d] ~ normal(0, 5);                     // Prior model
    tau[d] ~ cauchy(0, 2.5);                  // Prior model
      for (i in 1:E)
        gamma[d,i] ~ normal(mu[d], tau[d]);   // Non-centered hierarchical model
  }             
  for (n in 1:N)
    y[n] ~ normal(X[n, :]*beta[:, e[n]], 1);  // Observational model
}
"""


model_3 = """
data {
  int<lower=0> N;             // number of observations
  int<lower=1> D;             // number of covariates
  int<lower=1> E;             // number of environments
  int<lower=1,upper=E> e[N];  // associated environment
  matrix[N,D] X;              // covariate matrix
  vector[N] y;                // target vector
}

parameters {
  real mu[D];                 // population mean
  real<lower=0> tau[D];       // population scale
  matrix[D,E] gamma;          // Non-centered coefficients
}

transformed parameters {
  
  matrix[D,E] beta; // Recentered coefficients

  // Recentering             
  for (d in 1:D){
    for (i in 1:E){
      beta[d,i] = mu[d] + tau[d]*gamma[d,i];
    }
  }
}

model {
  for (d in 1:D){
    mu[d] ~ normal(0, 5);                     // Prior model
    tau[d] ~ cauchy(0, 2.5);                  // Prior model
    gamma[d, :] ~ std_normal();   // Non-centered hierarchical model
  }             
  for (n in 1:N)
    y[n] ~ normal(X[n, :]*beta[:, e[n]], 1);  // Observational model
}
"""

model_1 = """
data {
    int<lower=0> N;             // number of observations
    int<lower=1> D;             // number of covariates
    int<lower=1> E;             // number of environments
    int<lower=1,upper=E> e[N];  // associated environment
    matrix[N,D] X;              // covariate matrix
    vector[N] y;                // target vector
}
parameters {
    real mu[D];                 // population mean
    real<lower=0> tau[D];       // population scale
    matrix[D,E] beta;           // column of coefficients for each environment
} 
model {
  for (d in 1:D){
    mu[d] ~ normal(0, 1);
    tau[d] ~ cauchy(0, 1);
    for (i in 1:E)
      beta[d,i] ~ normal(mu[d], tau[d]); 
  }
  
  for (n in 1:N)
    y[n] ~ normal(X[n, :]*beta[:, e[n]], 1);
}
"""


# Simulation functions
# fit
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
    return fit

# Fit 
if __name__ == '__main__':
    list_of_fit = list()
    for i in range(1):
        filename_scm = 'data/experimentA/scm' + str(i) + '.pkl'
        with open(filename_scm, 'rb') as inp:
            S = pickle.load(inp)
        if sum(S.W[0]) != 0:
            #Import data
            filename_df = 'data/experimentA/df' + str(i)
            dataframe = pd.read_csv(filename_df, index_col=0)
            dataframe.columns = ['Y'] + ['X' + str(i + 1) for i in range(dataframe.shape[1] - 1)]
            #fit data with stan
            F = bhip_fit(dataframe, model_1, 100 + i)
            list_of_fit.append(F)


# Tests
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
    hdi_upper = N
    hdi_lower = 0
    sorted_sample = sorted(sample)
    N_discard = round(alpha*N)
    
    while(N_discard>0):
        
        diff_low = abs(sorted_sample[hdi_lower] - sorted_sample[hdi_lower + 1])
        diff_up = abs(sorted_sample[hdi_upper] - sorted_sample[hdi_lower - 1])
        
        if diff_low == diff_up:
            hdi_lower += 1
            hdi_upper -= 1
            N_discard -= 2
        
        if diff_low > diff_up:
            hdi_lower += 1
            N_discard -= 1
       
        if diff_low < diff_up:
            hdi_upper -= 1
            N_discard -= 1
    return(sorted_sample[hdi_lower],sorted_sample[hdi_upper])

def hdi_rope_test(sample, margin, alpha):
    ## Credible Interval
    q_low = alpha * 0.5
    CI_low = np.quantile(sample, q_low)
    CI_high = np.quantile(sample, 1 - q_low)

    # ROPE interval
    rope_low = -1 * margin
    rope_high = margin

    # Decision
    if (CI_low > rope_high) or (CI_high < rope_low):
        return 'Rejected'
    elif (CI_low > rope_low) and (CI_high < rope_high):
        return 'Accepted'
    else:
        return "Undecided"


def bhip_test(fit,scm,alpha = 0.05,global_rope = 0):

    D = fit.data["D"]
    E = fit.data["E"]
    Y = fit.data["y"]
    mu = fit.extract()['mu']  # matrix N x D
    beta = fit.extract()['beta']  # N x D x E
    
    IP = list()

    for d in range(D):
        print(f'---------beta_{d + 1}---------')
        
        invariant = 0
        # Test for zero estimates
        rope = [0.1 * np.std(Y[Y.index == (e + 1)]) for e in range(E)]
        zero_test_beta = [hdi_rope_test(beta[:, d, e], rope[0], alpha) for e in range(E)]
        
        if global_rope == 0:
            zero_test_mu = hdi_rope_test(mu[:, d], np.mean(rope), alpha)
        else:
            zero_test_mu = hdi_rope_test(mu[:, d], global_rope, alpha)
        if zero_test_mu != "Rejected":
            invariant += 1
        
        # Pooling factor
        pooling = pooling_factor(beta[:, d, :], mu[:, d], E)
        if pooling < 0.5:
            invariant += 1
        for e in range(E):
            if zero_test_beta[e] != "Rejected":
                invariant += 1

        if invariant == 0:
            IP.append(d)
    return IP


# # Fit 
# if __name__ == '__main__':
#     list_of_fit = list()
#     for i in range(100):

#         filename_scm = 'data/experimentA/scm' + str(i) + '.pkl'
#         with open(filename_scm, 'rb') as inp:
#             S = pickle.load(inp)
#         if sum(S.W[0]) != 0:
#             filename_df = 'data/experimentA/df' + str(i)
#             dataframe = pd.read_csv(filename_df, index_col=0)
#             dataframe.columns = ['Y'] + ['X' + str(i + 1) for i in range(dataframe.shape[1] - 1)]
#             F = bhip_fit(dataframe, model_2, 100 + i)
#             list_of_fit.append(F)

# if __name__ == '__main__':
#     list_of_fit = list()
#     for i in range(5):
#         filename_scm = 'data/experimentA/scm' + str(i) + '.pkl'
#         with open(filename_scm, 'rb') as inp:
#             S = pickle.load(inp)
#         filename_df = 'data/experimentA/df' + str(i)
#         dataframe = pd.read_csv(filename_df, index_col=0)
#         dataframe.columns = ['Y'] + ['X' + str(i+1) for i in range(dataframe.shape[1]-1)]
#         F = bhip_fit(dataframe, model, 'data/experimentA/fit0',100+i)
#         list_of_fit.append(F)
