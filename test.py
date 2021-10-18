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






from collections import Counter
def estimate_hdi(sample, alpha):
    bins = []
    lower_bound = min(sample)
    upper_bound = max(sample)
    width = len(sample)
    for low in range(lower_bound,upper_bound, width):
        bins.append((low, low + width))

    binned_sample = []

    for value in sample:
        index_bin = -1
        for i in range(0, len(bins)):
            if value < bins[i][1] or value >= bins[i][0]:
                bin_index = i
                return index_bin
        binned_sample.append(index_bin)

    N_rare = round(alpha*len(sample))
    frequencies = Counter(binned_sample)


def estimate_hdi(sample, alpha):
    lower_bound = min(sample)
    upper_bound = max(sample)
    width = len(sample)*0.1
    for low in range(lower_bound,upper_bound, width):
        bins.append((low, low + width))

    hdi_uppper =
    hdi_lower =