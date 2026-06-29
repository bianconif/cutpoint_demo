import matplotlib.pyplot as plt
import numpy as np

from ucimlrepo import fetch_ucirepo

def load_dwbcd():
    """
    Import the Diagnostic Wisconsin Breast Cancer Database
    
    Returns
    -------
    features : ndarray of float
        The feature values.
    feature_names : list of str
        The feature names.
    labels : list of int
        The malignancy labels (1 = positive, 0 = negative).
    """

    #Fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    #Data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features 
    y = breast_cancer_wisconsin_diagnostic.data.targets
    
    #Convert labels to 0/1
    labels = np.where(y['Diagnosis'] == 'M', 1, 0)
    
    #Convert feature values to numpy array
    features = X.to_numpy(dtype=float)
    
    feature_names = X.columns.to_list()
    
    return features, feature_names, labels

def plot_se_sp(thresholds, se, sp, cutpoint_idx, feature_name=None):
    """
    Plot sensitivity and specificity as a function of threshold.
    
    Parameters
    ----------
    thresholds : ndarray of numeric (N,1)
        The thresholds tested.
    se : ndarray of numeric (N,1)
        Sensitivity as a function of the thresholds.
    sp : ndarray of numeric (N,1)
        Specificity as a function of the thresholds.
    cutpoint_idx : int
        The index corresponding to optimal cut-point value.
    feature_name : str [optional]
        The feature name
    """
    grid_spec = {'hspace': 0.3}
    fig, axes = plt.subplots(nrows=2, gridspec_kw=grid_spec)

    metrics = [se, sp]
    colours = ['tab:blue', 'tab:orange']
    ylabels = ['Sensitivity', 'Specificity']
    for ax, metric, colour, ylabel in zip(axes, metrics, colours, ylabels):
        ax.plot(thresholds, metric, marker='o', color=colour)

        ax.set_ylabel(ylabel)
        ax.grid(visible=True)
        ax.vlines(x=thresholds[cutpoint_idx, 0], ymin=ax.get_ylim()[0],
                  ymax=ax.get_ylim()[1], color='tab:gray')

    ax.set_xlabel('Thresholds tested')
    
    axes[0].set_title(f'Feature: `{feature_name}`, '
                      f'opt. cut-off: {thresholds[cutpoint_idx, 0]:.1f} '
                      f'num. thresholds tested: {thresholds.size}')
    
    plt.show()
    

def plot_roc(thresholds, se, sp, cutpoint_idx, feature_name=None):
    """
    Plot Receiver-operating characteristic curve.
    
    Parameters
    ----------
    thresholds : ndarray of numeric (N,1)
        The thresholds tested.
    se : ndarray of numeric (N,1)
        Sensitivity as a function of the thresholds.
    sp : ndarray of numeric (N,1)
        Specificity as a function of the thresholds.
    cutpoint_idx : int
        The index corresponding to optimal cut-point value.
    """
    fig, ax = plt.subplots()
    ax.plot(1 - sp, se)

    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 - specificity')
    ax.grid(visible=True)

    #Place a marker corresponding to the point where the tarket is maximum
    ax.plot(1 - sp[cutpoint_idx, 0], se[cutpoint_idx, 0], marker='o')
    
    plt.show()
    
#Estimate extremes of 95% confidence intervals
def ci_l(x):
    return np.percentile(a=x, q=5.0)

def ci_u(x):
    return np.percentile(a=x, q=95.0)
    
