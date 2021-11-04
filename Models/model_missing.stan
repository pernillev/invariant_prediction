
data {
  int<lower=1> N_total              // observations in total
  int<lower=2,upper=D+1> B;         // number of blocks
  int<lower=1> E;                   // number of environments
  int<lower=0> N[E,B];              // block sizes in each environment
  int<lower=1> D;                   // number of covariates
  int<lower=1,upper=D> M            // index of missing covariate  
  matrix[N_total,D] X;              // covariate matrix
  vector[N_total] y;                // target vector
}

parameters {
  matrix[E,D] gamma;                // Non-centered coefficients   
  real mu[D];                       // global means
  real<lower=0> tau[D];             // global scales
  real<lower=0> sigma[B];           // local (block) scale
}
transformed parameters {
  matrix[D,E] beta;                 // Recentered coefficients
  // Recentering             
  for (d in 1:D){
    for (i in 1:E){
      beta[d,i] = mu[d] + tau[d]*gamma[d,i];
    }
  }
}
model {
  // Prior model
  for (d in 1:D){
    mu[d] ~ normal(0, 1);               
    tau[d] ~ cauchy(0, 1);                   
    gamma[d, :] ~ std_normal();            
  } 

  sigma ~ cauchy(0, 2.5);          

  int start;
  start = 1;
  for (e in 1:E){
    b = 1  
    mean = block(X, start, 1, N[e,m], D)*col(beta, e);
    segment(y, start, N[e,b]) ~ normal(mean, sigma[b]);
    start = start + N[e,b];
    for (b in 2:B){
      m = M[b-1]
      N_eb = N[e,b] 
      if (N_eb > 0) continue;
      if (m == 1) {
        // sum X_j*beta_j for j = 2,...,D        
        mean = block(X, start, 2, N_eb, D-1)*head(col(beta, e), D-1); 
      }
      else if (m == D) {
        // sum X_j*beta_j for j = 1,...,D-1
        mean = block(X, start, 1, N_eb, D-1)*tail(col(beta, e), D-1); 
      }
      else {
        // sum X_j*beta_j for j = 1,...,m-1,m+1,...D-1
        mean = block(X, start, 1, N_eb, m-1)*head(col(beta, e), m-1);
        mean += block(X, start, m+1, N_eb, D-m)*tail(col(beta, e), D-m); // sum X_j*beta_j for j = 1,...,D-1
      }
      segment(y, start, N_eb) ~ normal(mean, sigma[b]);
      start = start + N[e,b];
    }
  }   
} 
