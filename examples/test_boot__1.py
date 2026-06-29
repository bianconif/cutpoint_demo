"""
Testing CutpointCalculator.boot() 
-----------------------
Compute the optimal cut-point value of feature `radius1` on the
Diagnostic Wisconsin Breast Cancer Database.
"""

from cutpointpy.core import CutpointCalculator
from cutpointpy_demo.utils import ci_l, ci_u, load_dwbcd

import numpy as np
import pandas as pd
from tabulate import tabulate

all_features, feature_names, labels = load_dwbcd()

selected_feature = 'radius1'
feature_idx = feature_names.index(selected_feature)
features = all_features[:, feature_idx]

#Consider the following combinations of parameters
combinations = [
    {'target': 'youdenj', 'interpolation': None},
    {'target': 'youdenj', 'interpolation': 'linear', 'num_points': 1000},
    {'target': 'eucdist', 'interpolation': None},
    {'target': 'eucdist', 'interpolation': 'linear', 'num_points': 1000}    
]

for combination in combinations:
    
    cpcalc = CutpointCalculator(**combination)
    
    cutpoints, cutpoints_idxs, thresholds, accs, ses, sps, aucs_train,\
        aucs_test, performance_train, performance_test,\
        performance_whole = cpcalc.bootstrap(features, labels)
    
    #=============================================================
    #===== Pack results info dataframes and show the results =====
    #=============================================================
    data_dict = {'Cutpoint'  : cutpoints, 
                 'AUC_train' : aucs_train,
                 'AUC_test'  : aucs_test,
                 'Acc_train' : performance_train[:,[0]],
                 'Se_train'  : performance_train[:,[1]],
                 'Sp_train'  : performance_train[:,[2]],
                 'Acc_test'  : performance_test[:,[0]],
                 'Se_test'   : performance_test[:,[1]],
                 'Sp_test'   : performance_test[:,[2]],
                 'Acc_whole' : performance_whole[:,[0]],
                 'Se_whole'  : performance_whole[:,[1]],
                 'Sp_whole'  : performance_whole[:,[2]],                 
                 }
    df = pd.DataFrame(data = np.hstack(list(data_dict.values())),
                      columns = data_dict.keys())
    #=============================================================
    #=============================================================
    #=============================================================
        
        
    print(f'Combination: {combination}')
    print(tabulate(df, headers=df.columns))
    print()
    
    #Show aggregated results  
    df_agg = df.aggregate(func=['mean', 'std', ci_l, ci_u]).T
    print(tabulate(df_agg, headers=df_agg.columns))
    
        
