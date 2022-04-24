import pickle
import causalicp

# N_scm = 43
# N_data = 30
#
# icp_results = list()
# for i in range(25, 35):
#     filename_scm = 'data/experimentB/df1' + str(i) + '.pkl'
#     with open(filename_scm, 'rb') as inp:
#         scenario = pickle.load(inp)
#     print('scenario', i)
#     # Simulated SCM
#     SCM = scenario[0]
#     # true parental and child set
#     true_pa = set([i for i in range(len(SCM.W)) if SCM.W[i][0] != 0])
#     true_ch = set([i for i in range(len(SCM.W)) if SCM.W[0][i] != 0])
#     estimate = list()
#     for j in range(N_data):
#         print('data', j)
#         # Simulated data
#         data = scenario[2][j]
#         ICP = causalicp.fit(data, target=0, alpha=0.05, sets=None, precompute=False, verbose=False, color=True)
#         estimate.append(ICP.estimate)
#
#     icp_results.append((true_pa, true_ch, estimate))

# with open('data/experimentB/ICP_results_simB3_0_5.pkl', 'wb') as outp:  # Overwrites any existing file.
#     pickle.dump(icp_results, outp, pickle.HIGHEST_PROTOCOL)


# results2 = [results[i] for i in range(N_test) if results[i][2] != None] # non rejected models
# results3 = [results2[i] for i in range(len(results2)) if len(results2[i][0]) > 0] # non empty true parental set

filename = 'data/experimentB/df_3.pkl'
with open(filename, 'rb') as inp:
    list_of_df = pickle.load(inp)

filename = 'data/experimentB/scm_3.pkl'
with open(filename, 'rb') as inp:
    list_of_scm = pickle.load(inp)
#
#S = [1,2,3,4,5,9]
for el in range(12):
    lst = list_of_df[el]
    for i in range(len(lst)):
        path = 'data/experimentB/' + str(el) + '_' + str(i) +'.csv'
        lst[i].to_csv(path,index=True,index_label='e')


def df_to_array(df):
    array = []
    E = len(df.index.unique())
    for e in range(E):
        data_e = df[df.index == e + 1].to_numpy()
        array.append(data_e)
    return array
#
#
# icp_results = list()
#
# for i in range(6,7):
#     print('scenario', i)
#     # Simulated SCM
#     SCM = list_of_scm[i]
#     # true parental and child set
#     true_pa = set([i for i in range(len(SCM.W)) if SCM.W[i][0] != 0])
#     true_ch = set([i for i in range(len(SCM.W)) if SCM.W[0][i] != 0])
#     estimate = list()
#     ldf = list_of_df[i]
#     for j in range(5):
#         print('data', j)
#         # Simulated data
#         data = df_to_array(ldf[j])
#         ICP = causalicp.fit(data, target=0, alpha=0.05, sets=None, precompute=False, verbose=False, color=True)
#         estimate.append(ICP.estimate)
#
#     icp_results.append((true_pa, true_ch, estimate))


