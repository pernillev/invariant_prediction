# Models
# hierarchical models
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
# non centered coefficients
model_NC = """
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
"""
# Missing data
model_missing = """
data {
  int<lower=1> N_total;              // observations in total
  int<lower=1> D;                   // number of covariates
  int<lower=2,upper=D+1> B;         // number of blocks
  int<lower=1> E;                   // number of environments
  int<lower=0> N[E,B];              // block sizes in each environment
  int<lower=1,upper=D> M[B-1];      // index of missing covariate
  matrix[N_total,D] X;              // covariate matrix
  vector[N_total] y;                // target vector
}
parameters {
  matrix[E,D] beta;                //  coefficients   
  real mu[D];                       // global means
  real<lower=0> tau[D];             // global scales
  real<lower=0> sigma[B];           // local (block) scale
}

model {  
  vector[N_total] location;
  int pos;

  // Prior model
  for (d in 1:D){
    mu[d] ~ normal(0, 1);
    tau[d] ~ cauchy(0, 1);
    for (i in 1:E)
      beta[d,i] ~ normal(mu[d], tau[d]); 
  }
  sigma ~ cauchy(0, 1);

  pos = 1;
  for (e in 1:E) {
    location[1:N[e,1]] = block(X, pos, 1, N[e,M[1]], D)*col(beta, e);
    segment(y, pos, N[e,1]) ~ normal(location[1:N[e,1]], sigma[1]);
    pos = pos + N[e,1];
    for (b in 2:B) { 
      //if (N[e,b] > 0) continue;
      if (M[b] == 1) {
        // sum X_j*beta_j for j = 2,...,D        
        location[1:N[e,b]] = block(X, pos, 2, N[e,b], D-1)*head(col(beta, e), D-1); 
      }
      else if (M[b] == D) {
        // sum X_j*beta_j for j = 1,...,D-1
        location[1:N[e,b]] = block(X, pos, 1, N[e,b], D-1)*tail(col(beta, e), D-1); 
      }
      else {
        // sum X_j*beta_j for j = 1,...,m-1,m+1,...D-1
        location[1:N[e,b]] = block(X, pos, 1, N[e,b], M[b]-1)*head(col(beta, e), M[b]-1) + block(X, pos, M[b]+1, N[e,b], D-M[b])*tail(col(beta, e), D-M[b]); 
      }
      segment(y, pos, N[e,b]) ~ normal(location[1:N[e,b]], sigma[b]);
      pos = pos + N[e,b];
    }
  }    
} 
"""
# logistic regression non centered coefficients
model_log_NC = """
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
    y[n] ~ bernoulli(inv_logit(X[n, :]*beta[:, e[n]]));  // Observational model
}
"""

model_log = """
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
    y[n] ~ bernoulli(inv_logit(X[n, :]*beta[:, e[n]]));  // Observational model
}
"""

model = """
data {
    int<lower=1> B;             // number of blocks
    int<lower=1> E;             // number of environments
    int<lower=0> N[B,E];        // number of observations
    int<lower=1> D[B];             // number of covariates
    int<lower=1,upper=E> e[N];  // associated environment
    int<lower=1,upper=E> b[N];  // associated block
    vector[B] S                 //
    matrix[B,E]  X[N[B,E],D[B]];        // covariate matrix
    vector[N] y;                // target vector
}
parameters {
    real mu[D];                 // global mean
    real<lower=0> tau[D];       // global scale
    matrix[B,E] beta[D];        // column of coefficients for each environment
} 
model {

    for (b in 1:B)
    
    for (d in 1:D){
    mu[d] ~ normal(0, 1);
    tau[d] ~ cauchy(0, 1);
    
        for (i in 1:E)
            beta[b,i] ~ normal(mu[d], tau[d]); 
  }

  for (n in 1:N)
    y[n] ~ normal(X[n, :]*beta[:, e[n]], 1);
}
"""


# Opening files
file1 = open('model_regular.stan', 'w')
file2 = open('model_NC.stan', 'w')
file3 = open('model_missing.stan', 'w')
file4 = open('model_logistic.stan', 'w')
file5 = open('model_logistic_NC.stan', 'w')
# Writing strings to file
file1.write(model)
file2.write(model_NC)
file3.write(model_missing)
file4.write(model_log)
file5.write(model_log_NC)
