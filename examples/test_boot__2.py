"""
Testing CutpointCalculator.boot() 
-----------------------
Compute the optimal cut-point value of feature `Glucose` on the 
Pima Indians Diabetes Database.
"""

from cutpointpy.core import CutpointCalculator

import pandas as pd
from tabulate import tabulate

#Get the data
src = f'https://raw.githubusercontent.com/bianconif/cascara/refs/heads/master/data/pima/cleaned/diabetes.csv'
df = pd.read_csv(filepath_or_buffer=src, comment='#')
print(f'Number of records: {df.shape[0]}')

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
        performance_whole = cpcalc.bootstrap(
            features=df.Glucose, labels=df.Outcome
        )
    
    #=============================================================
    #===== Pack results info dataframes and show the results =====
    #=============================================================
    df_res = pd.DataFrame()
    
    #Acc, se and sp at each repetition
    for key, val in {'_train': performance_train, 
                     '_test': performance_test, 
                     '_whole': performance_whole
                     }.items():
        df_res_ = pd.DataFrame(
            index=range(val.shape[0]), 
            data=val, 
            columns=[item + key for item in ['Acc', 'Se', 'Sp']]
        )
        df_res = pd.concat((df_res, df_res_), axis=1)
    
    #AUC at each repetition    
    for key, val in {'_train': aucs_train, 
                     '_test': aucs_test 
                     }.items():
        df_res_ = pd.DataFrame(
            index=range(val.shape[0]), 
            data=val, 
            columns=[item + key for item in ['AUC']]
        )
        df_res = pd.concat((df_res, df_res_), axis=1)
    #=============================================================
    #=============================================================
    #=============================================================
        
        
    print(f'Combination: {combination}')
    print(tabulate(df_res, headers=df_res.columns))
    print()
