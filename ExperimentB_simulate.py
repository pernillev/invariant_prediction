from ExperimentA_simulate import gen_scms
from ExperimentA_simulate import generate_interventions
import numpy as np
import random


##### Generate SCM and interventions ######
N_scenarios = 5
list_of_scm = list()
for _ in range(N_scenarios):
    D = random.randint(5, 15)  # number of nodes
    deg = random.randint(2, 4)  # average degree of graph
    lb1 = random.uniform(0.1, 2)  # lower bound on linear coefficients
    ub1 = lb1 + random.uniform(0.1, 1)  # upper bound of coefficient values
    sigmas = np.random.uniform(0.1, 2, 2)  # two noise variance
    if sigmas[0] < sigmas[1]:
        sigma_min = sigmas[0]
        sigma_max = sigmas[1]
    else:
        sigma_min = sigmas[1]
        sigma_max = sigmas[0]
    scm = gen_scms(G=1, p=D, k=deg,
                   w_min=lb1, w_max=ub1,
                   m_min=0, m_max=0,
                   v_min=sigma_min, v_max=sigma_max,
                   random_state=101)
    list_of_scm.append(scm[0])

N_intervention = 5
N_environments = 5
cases = list()
for scm in list_of_scm:
    interventions = list()
    D = scm.p
    for _ in range(N_intervention):
        inv_theta = [D if (random.choice(range(6)) == 0) else np.random.uniform(1.1, 3)]
        int_size = round(D / inv_theta[0])
        a_min = random.uniform(0.1, 4)
        a_Delta = [0 if (random.choice(range(3)) == 0) else np.random.uniform(0.1, 2)]
        a_max = a_min + a_Delta[0]
        intervention = generate_interventions(scm=scm,
                                              no_ints=1,
                                              int_size=int_size,
                                              m_max=0,
                                              m_min=0,
                                              v_min=a_min,
                                              v_max=a_max,
                                              include_obs=True,
                                              exclude_target =True)
        interventions.append(intervention)
    cases.append((scm, interventions))

model_missing = """
data {
    int<lower=1> N_total                // observations in total
    int<lower=1> B;                     // number of blocks
    int<lower=1> N[B];                  // vector of number of observations in each block
    int<lower=1> D;                     // number of covariates
    int<lower=1> E;                     // number of environments
    int<lower=1,upper=E> e[N_total];    // associated environment
    int<lower=1,upper=E> b[N_total];    // associated block    

    matrix[N_total,D] X;              // covariate matrix
    vector[N_total] y;                // target vector
}
parameters {
    matrix[D,E] gamma;          // Non-centered coefficients   
    real mu[D];                 // global means
    real<lower=0> tau[D];       // global scales
    real<lower=0> sigma[B];     // local (block) scale
}
transformed parameters {
  matrix[D,E] beta;             // Recentered coefficients
  // Recentering             
  for (d in 1:D){
    for (i in 1:E){
      beta[d,i] = mu[d] + tau[d]*gamma[d,i];
    }
  }
}

model {
  for (d in 1:D){
    mu[d] ~ normal(0, 1);         // Prior        
    tau[d] ~ cauchy(0, 1);                   
    gamma[d, :] ~ std_normal();            
  } 
  int start;
  int end;
  start = 1; 
  for (m in 1:B){
    end = pos + N[m] - 1
    sigma[m] ~ cauchy(0, 2.5);
    for (n in start:end):
        mean = 
        y[n] ~ normal(X[n, "-b[n]"]*beta["-b[n]", e[n]], sigma[m]); 
    pos = pos + N[m];
  }
}
"""