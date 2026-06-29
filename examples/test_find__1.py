"""
Test CutpointCalculator.find() 
-----------------------
Dataset: Diagnostic Wisconsin Breast Cancer Database.

Objective: Compute the optimal cut-point value for feature `radius1` to 
discriminate benign vs. malignant cases. 
"""

from cutpointpy.core import CutpointCalculator
from cutpointpy_test.utils import load_dwbcd, plot_roc, plot_se_sp

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
    
    opt_cutpoint, cutpoint_idx, thresholds, acc, se, sp, _ =\
        cpcalc.find(features=features, labels=labels)
    
    print(f'Combination: {combination}')
    print(f'Cut-point: {opt_cutpoint:.2f}, acc: {100*acc[cutpoint_idx,0]:.1f}, '
          f'se: {100*se[cutpoint_idx,0]:.1f}, sp: {100*sp[cutpoint_idx,0]:.1f}')

    #Plot sensitivity and specificity as a function of threshold
    plot_se_sp(thresholds, se, sp, cutpoint_idx,
               feature_name=selected_feature)

    #Plot ROC curve
    plot_roc(thresholds, se, sp, cutpoint_idx)