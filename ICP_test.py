import pickle
import causalicp

N_scm = 50
N_data = 50

icp_results = list()
for i in range(N_scm):
    filename_scm = 'data/experimentA/sim_vs2_' + str(i) + '.pkl'
    with open(filename_scm, 'rb') as inp:
        scenario = pickle.load(inp)
    print('scenario', i)
    # Simulated SCM
    SCM = scenario[0]
    # true parental and child set
    true_pa = set([i for i in range(len(SCM.W)) if SCM.W[i][0] != 0])
    true_ch = set([i for i in range(len(SCM.W)) if SCM.W[0][i] != 0])
    estimate = list()
    for j in range(N_data):
        print('data', j)
        # Simulated data
        data = scenario[2][j]
        ICP = causalicp.fit(data, 0, alpha=0.05, sets=None, precompute=False, verbose=False, color=True)
        estimate.append(ICP.estimate)

    icp_results.append((true_pa, true_ch, estimate))

with open('data/experimentA/icp_results.pkl', 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(icp_results, outp, pickle.HIGHEST_PROTOCOL)


#results2 = [results[i] for i in range(N_test) if results[i][2] != None] # non rejected models
#results3 = [results2[i] for i in range(len(results2)) if len(results2[i][0]) > 0] # non empty true parental set
