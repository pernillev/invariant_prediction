import pickle
from invariant_prediction.BHIP import invariance_tests
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import arviz
warnings.filterwarnings('ignore')
from invariant_prediction.BHIP_test import bhip_fit_exp_A

model_log = """
data {
    int<lower=0> N;             // number of observations
    int<lower=1> D;             // number of covariates
    int<lower=1> E;             // number of environments
    int<lower=1,upper=E> e[N];  // associated environment
    matrix[N,D] X;              // covariate matrix
    int<lower=0,upper=1> y[N];                // target vector
}
parameters {
  real intercept[E];  
  real mu[D];                 // population mean
  real<lower=0> tau[D];       // population scale
  matrix[D,E] beta;           // column of coefficients for each environment

} 
model {
  intercept ~ normal(0,1);
  for (d in 1:D){
    mu[d] ~ normal(0, 1);
    tau[d] ~ cauchy(0, 0.25);
    for (i in 1:E)
      beta[d,i] ~ normal(mu[d], tau[d]); 
  }
  for (n in 1:N)
    y[n] ~ bernoulli(inv_logit(intercept[e[n]] + X[n, :]*beta[:, e[n]]));  // Observational model
}
"""


def data_to_pd(data):
    # Combine the data into a single matrix with an extra column indicating the environment
    df = pd.DataFrame()
    for e, X in enumerate(data):
        df_e = pd.DataFrame(X, index=np.repeat(e + 1, len(X)))
        df = pd.concat([df, df_e])
    df.columns = ['Y'] + ['X' + str(col + 1) for col in range(df.shape[1] - 1)]
    return df


def replace_education(edu):
    if edu < 16:
        return int(0)
    else:
        return int(1)


## Experiment C
filename = 'data/CD_fit2.pkl'
with open(filename, 'rb') as inp:
    fit = pickle.load(inp)
f = fit[1]


# f = list_of_fits[5][11][1]
mu = f.extract()['mu']  # vector 1 x D
tau = f.extract()['tau']  # vector 1 x D
beta = f.extract()['beta']  # matrix D x E
intercept = f.extract()['intercept']  # matrix D x E

E = 2
D = 14

mu_pass, beta_pass, pool_pass = invariance_tests(fit=f, global_rope="two_sd", alpha_mu=0.35,alpha_beta=0.05, printing=True)
est = set(mu_pass).intersection(set(beta_pass), set(pool_pass))

col_names = ['score', 'unemp', 'wage', 'distance', 'tuition', 'gender_male',
       'ethnicity_afam', 'ethnicity_other', 'fcollege_no', 'mcollege_no',
       'home_no', 'urban_no', 'income_low', 'region_other']

#D = 7
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

sd_beta = arviz.summary(f, var_names=['beta'])['sd'].values
sd_mu = arviz.summary(f, var_names=['mu'])['sd'].values

for i in range(7,D):
    my_kde = sns.kdeplot(mu[:, i], label=col_names[i], ax=axs[0])
    line = my_kde.lines[i-7]
    hdi_data = arviz.hdi(mu[:, i], hdi_prob=1 - 0.35)
    x, y = line.get_data()
    mask = x > hdi_data[0]
    x, y = x[mask], y[mask]
    mask = x < hdi_data[1]
    x, y = x[mask], y[mask]
    axs[0].fill_between(x, y1=y, alpha=0.15)
for d in range(7,D):
    rope_d = round(np.mean([2*sd_beta[d*E + e] for e in range(E)]),3)
    print(rope_d)
    axs[0].axvline(rope_d, color=sns.color_palette("tab10")[d-7], linestyle=':')
    axs[0].axvline(-rope_d, color=sns.color_palette("tab10")[d-7], linestyle=':')
axs[0].legend()
axs[0].set_xlim(-1.5, 1.5)
axs[0].set_title(f'Posterior distribution of global mean')

for i in range(7,D):
    sns.kdeplot(tau[:, i], label=col_names[i], ax=axs[1])
axs[1].legend()
axs[1].set_xlim(-0.2, 1.25)
axs[1].set_title(f'Posterior distribution of global variance')

fig, axs = plt.subplots(nrows=7, ncols=1, figsize=(10, 20))
for d in range(7,D):
    for i in range(0, E):
        my_kde2 = sns.kdeplot(beta[:, d, i], label=f'Environment e={i + 1}', ax=axs[d-7])
        line = my_kde2.lines[i]
        hdi_data = arviz.hdi(beta[:, d, i], hdi_prob=1 - 0.05)
        x, y = line.get_data()
        mask = x > hdi_data[0]
        x, y = x[mask], y[mask]
        mask = x < hdi_data[1]
        x, y = x[mask], y[mask]
        axs[d-7].fill_between(x, y1=y, alpha=0.35)
    for i in range(0, E):
        cols = ['blue', 'orange']
        print(d * E + i, 2 * sd_beta[d * E + i])
        axs[d-7].axvline(2 * sd_beta[d * E + i], 0, 5, c=cols[i])
        axs[d-7].axvline(-1 * 2 * sd_beta[d * E + i], 0, 5, label=f'ROPE e={i + 1}', c=cols[i])
    axs[d-7].set_xlim(-1, 1)
    axs[d-7].legend()
    axs[d-7].set_title(f'beta_{col_names[d]}')












