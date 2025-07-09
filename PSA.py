import numpy as np
def PSA(w,s,node_num,DATA):
    w = w
    s = s
    if DATA == 'UCLA':
        length = 145
        subjects_timeseries_with_confounds = np.load('C:\\model\\DSF-GNN_dianke\\UCLA_subjects_timeseries_with_confounds.npy',allow_pickle=True)
    if DATA == 'reUCLA':
        length = 115
        subjects_timeseries_with_confounds = np.load('C:\\model\\DSF-GNN_dianke\\reUCLA_subjects_timeseries_with_confounds.npy',allow_pickle=True)
    if DATA == 'COBRE':#cobre
        length = 145
        subjects_timeseries_with_confounds = np.load('C:\\model\\DSF-GNN_dianke\\subjects_timeseries_with_confounds_Balance_COBRE.npy')
    if DATA == 'COBRE+UCLA':
        length = 145
        subjects_timeseries_with_confounds = np.load('C:\\model\\DSF-GNN_dianke\\COBRE+UCLA_subjects_timeseries_with_confounds.npy')
    if DATA == 'SRPBS':
        length = 160
        subjects_timeseries_with_confounds = np.load('C:\\model\\DSF-GNN_SRPBS\\shuffled_SRPBS_dataset.npy')
    if DATA == 'dianke':
        length = 245
        subjects_timeseries_with_confounds = np.load('C:\\model\\DSF-GNN_dianke\\shuffled_dianke_dataset.npy')

    # subjects_timeseries_with_confounds is the raw BOLD signal for each brain region after preprocessing

    rang = (length-w)/s+1
    print(rang)
    rang = int(rang)
    num = subjects_timeseries_with_confounds.shape[0]

    Dy_subjects_timeseries_with_confounds = np.array([[[0]*node_num]*(rang*w)]*num, dtype=np.float64)
    for j in range(num):
        for i in range(rang):
            Dy_subjects_timeseries_with_confounds[j][(w*i):(w*(i+1))] = subjects_timeseries_with_confounds[j][(0+i*s):(w+i*s)]

    node_feature = np.array([[[0]*node_num]*rang]*num, dtype=np.float64)

    for i in range(rang):
        subjects_timeseries = Dy_subjects_timeseries_with_confounds[:,(w*i):(w*(i+1))]
        node_feature[:,i] = np.mean(subjects_timeseries,axis = 1)
    return node_feature
