
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
