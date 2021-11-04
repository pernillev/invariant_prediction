
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
    mu[d] ~ normal(0, 1);           // Prior model
    tau[d] ~ cauchy(0, 1);         // Prior model
    gamma[d, :] ~ std_normal();   // Non-centered hierarchical model
  }             
  for (n in 1:N)
    y[n] ~ normal(X[n, :]*beta[:, e[n]], 1);  // Observational model
}
