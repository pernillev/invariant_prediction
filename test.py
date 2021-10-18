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
        
        if diff_low > diff_high:
            hdi_lower += 1
            N_discard -= 1
       
        if diff_low < diff_high:
            hdi_upper -= 1
            N_discard -= 1
    return(sorted_sample[hdi_lower],sorted_sample[hdi_upper])