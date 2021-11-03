import random
import pickle
import time
from datetime import datetime
import numpy as np
import sempler
import sempler.generators
import pandas as pd


####### Functions #########
def gen_scms(G, p, k, w_min, w_max, m_min, m_max, v_min, v_max, random_state):
    """
    Generate random experimental cases (ie. linear SEMs). Parameters:
      - G: total number of cases
      - p: number of variables in the SCMs
      - k: average node degree
      - w_min, w_max: Weights of the SCMs are sampled at uniform between w_min and w_max
      - v_min, v_max: Variances of the variables are sampled at uniform between v_min and v_max
      - m_min, m_max: Intercepts of the variables of the SCMs are sampled at uniform between m_min and m_max
      - random_state: to fix the random seed for reproducibility
    """
    random.seed(random_state)
    cases = list()
    while len(cases) < G:
        W = sempler.generators.dag_avg_deg(p, k, w_min, w_max)
        W *= np.random.choice([-1, 1], size=W.shape)
        scm = sempler.LGANM(W, (m_min, m_max), (v_min, v_max))
        cases.append(scm)
    return cases


def generate_interventions(scm, no_ints, int_size, m_min, m_max, v_min, v_max, include_obs=False, exclude_target=False):
    """Generate a set of interventions for a given scm, randomly sampling
    no_ints sets of targets of size int_size, and sampling the
    intervention means/variances uniformly from [m_min,m_max], [v_min, v_max].

    If include_obs is True, include an empty intervention to represent
    the reference or observational environment.

    If exclude_target is True, exclude target X = 0 from intervention
    """
    interventions = [None] if include_obs else []
    # For each intervention
    for _ in range(no_ints):
        # sample targets
        if exclude_target:
            # without intervening on  node 0 (= target Y)
            if int_size == scm.p:
                int_size -= 1
            targets = np.random.choice(range(1, scm.p), int_size, replace=False)
        else:
            targets = np.random.choice(range(scm.p), int_size, replace=False)
        # sample parameters
        means = np.random.uniform(m_min, m_max, len(targets)) if m_min != m_max else [m_min] * len(targets)
        variances = np.random.uniform(v_min, v_max, len(targets)) if v_min != v_max else [v_min] * len(targets)
        # assemble intervention
        intervention = dict((t, (mean, var)) for (t, mean, var) in zip(targets, means, variances))
        interventions.append(intervention)
    return interventions


##### Generate SCM and interventions ######
N_scenarios = 50
list_of_scm = list()
for _ in range(N_scenarios):
    D = random.randint(5, 15)  # number of nodes
    deg = random.randint(1, 4)  # average degree of graph
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

N_intervention = 50
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
                                              include_obs=True)
        interventions.append(intervention)
    cases.append((scm, interventions))


# Generate data and save it
def pick_coef(lb, ub):
    coef = np.round(np.random.uniform(lb, ub, 1) * random.choice([-1, 1]), 2)
    return coef[0]

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


print("\n\nSampling data for %d test cases %s\n\n" % (len(cases), datetime.now()))
start = time.time()
debug = True
list_of_df = list()
for i, (scm, interventions) in enumerate(cases):
    D = scm.p
    XXX = []
    start_case = time.time()
    print('\nGenerating data for test case %d' % i)
    print('  D=%d, Interventions=%s' % (D, interventions)) if debug else None
    for intervention in interventions:
        N_obs = random.choice([100, 200, 300, 400, 500])
        N_int = random.choice([100, 200, 300, 400, 500])
        # Sample interventional data
        XX = []
        for dict in intervention:
            if dict is None:
                X = scm.sample(n=N_obs)
            else:
                keys = list(dict.keys())
                # Intervene on nodes in A
                scm_intervention = scm
                for key in keys:
                    # possibly sample new coefficient
                    if random.choice(range(3)) == 0:
                        deg = random.randint(1, 4)
                        # lower and upper bound of new coefficients
                        coef_bounds = np.random.uniform(0.1, 2, 2)
                        if coef_bounds[0] < coef_bounds[1]:
                            lbe = coef_bounds[0]
                            ube = coef_bounds[1]
                        else:
                            lbe = coef_bounds[1]
                            ube = coef_bounds[0]
                        for parent in range(D):
                            if scm.W[parent][key] != 0:
                                new_coef = pick_coef(lbe, ube)
                                scm_intervention.W[parent][key] = new_coef
                # sample interventional data
                X = scm_intervention.sample(N_int, shift_interventions=dict)
            # append interventional data for each environment
            XX.append(X)
        # append simulated data
        XXX.append(XX)
    # save SCM, list of interventions and list of simulations XXX
    filename = 'data/experimentA/sim_vs2_' + str(i) + '.pkl'
    save_object((scm, interventions, XXX), filename)
    print('Done (%0.2f seconds)' % (time.time() - start_case))
end = time.time()
print("\n\nFinished at %s (elapsed %0.2f seconds)" % (datetime.now(), end - start))
