# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:56:10 2021

@author: PernilleV
"""


model = """
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
    // Recentered coefficients
    matrix[D,E] beta;
    
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
               gamma[d,i] ~ normal(mu[d], tau[d]);       // Non-centered hierarchical model
               }             
       for (n in 1:N)
       y[n] ~ normal(X[n, :]*beta[:, e[n]], 1);          // Observational model
}
"""