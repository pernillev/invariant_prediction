import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pystan
import seaborn as sns
from BHIP import invariance_tests
import arviz

# non centered coefficients
model_NC = """
data {
  int<lower=0> N;             // number of observations
  int<lower=1> D;             // number of covariates
  int<lower=1> E;             // number of environments
  int<lower=1,upper=E> e[N];  // associated environment
  matrix[N,D] X;              // covariate matrix
  real y[N];                // target vector
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
    mu[d] ~ normal(0, 1);           // Prior model
    tau[d] ~ cauchy(0, 1);         // Prior model
    gamma[d, :] ~ std_normal();   // Non-centered hierarchical model
  }             
  for (n in 1:N)
    y[n] ~ normal(X[n, :]*beta[:, e[n]], 1);  // Observational model
}
"""

model_regular = """
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

X = pd.read_csv('data/X_small_ex_100.csv')
Y = pd.read_csv('data/Y_small_ex_100.csv').x.values

# Stan model object
if __name__ == '__main__':
    N, D = X.shape
    E = 2
    e = np.concatenate((np.repeat(1, int(N / 2)), np.ones(int(N / 2)) * 2), axis=0).astype(int)
    data = {'N': N, 'D': D, 'E': E, 'e': e, 'X': X, 'y': Y}
    sm = pystan.StanModel(model_code=model_regular)
    fit = sm.sampling(data=data, iter=2000, chains=4,
                      algorithm="NUTS", seed=123,
                      control=dict(adapt_delta=0.95, max_treedepth=15))

    mu = fit.extract()['mu']  # vector 1 x D
    tau = fit.extract()['tau']  # vector 1 x D
    beta = fit.extract()['beta']  # matrix D x E

    mu_pass, beta_pass, pool_pass = invariance_tests(fit)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    sd_beta = arviz.summary(fit, var_names=['beta'])['sd'].values
    sd_mu = arviz.summary(fit, var_names=['mu'])['sd'].values

    for i in range(0, D):
        my_kde = sns.kdeplot(mu[:, i], label=f'mu{i + 1}', ax=axs[0])
        line = my_kde.lines[i]
        hdi_data = arviz.hdi(mu[:, i], hdi_prob=1 - 0.15)
        x, y = line.get_data()
        mask = x > hdi_data[0]
        x, y = x[mask], y[mask]
        mask = x < hdi_data[1]
        x, y = x[mask], y[mask]
        axs[0].fill_between(x, y1=y, alpha=0.15)
    for d in range(D):
        rope_d = sd_mu[d]
        print(rope_d)
        axs[0].axvline(rope_d, color=sns.color_palette("tab10")[d], linestyle=':')
        axs[0].axvline(-rope_d, color=sns.color_palette("tab10")[d], linestyle=':')
    axs[0].legend()
    axs[0].set_xlim(-1, 2)
    axs[0].set_title(f'Posterior distribution of global mean')

    for i in range(0, D):
        sns.kdeplot(tau[:, i], label=f'tau{i + 1}', ax=axs[1])
    axs[1].legend()
    axs[1].set_xlim(-0.2, 1.25)
    axs[1].set_title(f'Posterior distribution of global variance')

    fig, axs = plt.subplots(nrows=D, ncols=1, figsize=(10, 15))
    for d in range(0, D):
        for i in range(0, E):
            my_kde2 = sns.kdeplot(beta[:, d, i], label=f'Environment e={i + 1}', ax=axs[d])
            line = my_kde2.lines[i]
            hdi_data = arviz.hdi(beta[:, d, i], hdi_prob=1 - 0.05)
            x, y = line.get_data()
            mask = x > hdi_data[0]
            x, y = x[mask], y[mask]
            mask = x < hdi_data[1]
            x, y = x[mask], y[mask]
            axs[d].fill_between(x, y1=y, alpha=0.35)
        for i in range(0, E):
            cols = ['blue', 'orange']
            print(d*E + i,2 * sd_beta[d * E + i])
            axs[d].axvline(2 * sd_beta[d * E + i], 0, 5, c=cols[i])
            axs[d].axvline(-1 *2 * sd_beta[d * E + i], 0, 5, label=f'ROPE e={i + 1}', c=cols[i])
        axs[d].set_xlim(-0.75, 1.25)
        axs[d].legend()
        axs[d].set_title(f'beta{d + 1}')
