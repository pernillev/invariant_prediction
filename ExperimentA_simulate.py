import copy
import random
import pickle
import time
from datetime import datetime
import numpy as np


from invariant_prediction.simulate_functions import generate_scms
from invariant_prediction.simulate_functions import generate_shift_interventions

random.seed(1010)
##### Generate SCM and interventions ######
N_scenarios = 50
list_of_scm = list()
for _ in range(N_scenarios):
    lb1 = random.uniform(0.1, 2)  # lower bound on linear coefficients
    ub1 = lb1 + random.uniform(0.1, 1)  # upper bound of coefficient values
    sigmas = np.random.uniform(0.1, 2, 2)  # two noise variance
    if sigmas[0] < sigmas[1]:
        sigma_min = sigmas[0]
        sigma_max = sigmas[1]
    else:
        sigma_min = sigmas[1]
        sigma_max = sigmas[0]
    scm = generate_scms(G=1,
                        p_min=5, p_max=20,
                        k_min=1, k_max=4,
                        w_min=lb1, w_max=ub1,
                        m_min=0, m_max=0,
                        v_min=sigma_min, v_max=sigma_max,
                        random_state=101)
    list_of_scm.append(scm[0])

# A1
parents = [set([i for i in range(len(SCM.W)) if SCM.W[i][0] != 0]) for SCM in list_of_scm]
ps = [scm.p for scm in list_of_scm]
lst_1 = [list_of_scm[i] for i in range(N_scenarios) if len(parents[i]) > 0]
lst_2 = [list_of_scm[i] for i in range(N_scenarios) if len(parents[i]) == 0]
listA1 = lst_1 + lst_2[0:round(len(lst_1)*0.25)]

N_intervention = 50
cases = list()
for scm in listA1:
    interventions = list()
    for _ in range(N_intervention):
        inv_theta = [scm.p if (random.choice(range(6)) == 0) else np.random.uniform(1.1, 3)]
        int_size = round(scm.p / inv_theta[0])
        a_min = random.uniform(0.1, 4)
        a_Delta = [0 if (random.choice(range(3)) == 0) else np.random.uniform(0.1, 2)]
        a_max = a_min + a_Delta[0]

        if random.choice(range(2)) == 0:
            intervention = generate_shift_interventions(scm=scm,
                                                        no_ints=1, int_size=int_size,
                                                        m_max=0, m_min=0,
                                                        v_min=a_min, v_max=a_max,
                                                        include_obs=True, exclude_target=False)
        else:
            intervention = generate_shift_interventions(scm=scm,
                                                        no_ints=1, int_size=int_size,
                                                        m_max=0, m_min=0,
                                                        v_min=a_min, v_max=a_max,
                                                        include_obs=True, exclude_target=True)

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
for i, (scm, interventions) in enumerate(cases):
    D = scm.p
    XXX = []
    start_case = time.time()
    print('\nGenerating data for test case %d' % i)
    print('  D=%d, Interventions=%s' % (D, interventions)) if debug else None
    for intervention in interventions:
        # Sample interventional data
        XX = []
        for shift_dict in intervention:
            if shift_dict is None:
                N_obs = random.choice([100, 200, 300, 400, 500])
                X = scm.sample(n=N_obs)
            else:
                keys = list(shift_dict.keys())
                # Intervene on nodes in A
                scm_intervention = copy.deepcopy(scm)
                for key in keys:
                    # possibly sample new coefficient
                    if random.choice(range(3)) == 0:
                        # lower and upper bound of new coefficients
                        coef_bounds = np.random.uniform(0.1, 2, 2)
                        if coef_bounds[0] < coef_bounds[1]:
                            lbe = coef_bounds[0]
                            ube = coef_bounds[1]
                        else:
                            lbe = coef_bounds[1]
                            ube = coef_bounds[0]
                        for parent in range(D):
                            if scm.W[parent, key] != 0:
                                scm_intervention.W[parent, key] = pick_coef(lbe, ube)
                # sample interventional data
                N_int = random.choice([100, 200, 300, 400, 500])
                X = scm_intervention.sample(N_int, shift_interventions=shift_dict)
            # append interventional data for each environment
            XX.append(X)
        # append simulated data
        XXX.append(XX)
    # save SCM, list of interventions and list of simulations XXX
    filename = 'data/experimentA/simulation_A' + str(i) + '.pkl'
    save_object((scm, interventions, XXX), filename)
    print('Done (%0.2f seconds)' % (time.time() - start_case))
end = time.time()
print("\n\nFinished at %s (elapsed %0.2f seconds)" % (datetime.now(), end - start))
