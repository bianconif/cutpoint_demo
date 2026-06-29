import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    import pandas as pd
    import seaborn as sns

    return Rectangle, np, pd, plt, sns


@app.cell
def _():
    from mpl_ornaments.titles import set_title_and_subtitle

    return


@app.cell
def _():
    from cutpointpy.core import CutpointCalculator

    return (CutpointCalculator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimal cut-point value of glucose level for predicting diabetes with bootstrap estimation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In previous notebooks of this series ([cutpointboot__glucose_cutoff_for_diabetes.py](cutpointboot__glucose_cutoff_for_diabetes.py), [cutpointfind__ibmi_cutoff_for_diabetes.py](cutpointfind__ibmi_cutoff_for_diabetes.py)) we showed how to use `cutpointpy` for estimating the optimal cut-off value of a predictor variable for a binary outcome.

    Herein we illustrate how to use `cutpointpy` for cut-point estimation with bootstrapping. Simply put, booststrapping consists of repeatedly re-sampling the data a given number of times (*runs* or *repetitions*) and recalculating the optimal cut-point and the performance parameters at each run. This enables determining the stability, variability and confidence intervals both of the the estimated cut-point and of the performance metrics.

    The data come from the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), available on Kaggle ([CC0: Public Domain](https://creativecommons.org/public-domain/)).

    The first part of the notebook (data retrieval) replicates the one available in [cutpointboot__glucose_cutoff_for_diabetes.py](cutpointboot__glucose_cutoff_for_diabetes.py) and is therefore presented with no further comments here. For simplicity, the strip/box plots are also omitted in the present notebook, but can be reproduced with code available in [cutpointboot__glucose_cutoff_for_diabetes.py](cutpointboot__glucose_cutoff_for_diabetes.py).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get the data
    """)
    return


@app.cell
def _(pd):
    src = 'https://raw.githubusercontent.com/bianconif/cascara/refs/heads/master/data/pima/cleaned/diabetes.csv'
    df = pd.read_csv(filepath_or_buffer=src, comment='#')
    print(f'Number of records: {df.shape[0]}')
    return (df,)


@app.cell
def _():
    feature_name = 'Glucose'
    units = {feature_name: 'mg/dL'}
    return feature_name, units


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Perform optimal cut-point analysis with bootstrapping
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As usual we start by creating an instance of `CutpointCalculator`. We refer the reader to the other notebooks of this series ([cutpointboot__glucose_cutoff_for_diabetes.py](cutpointboot__glucose_cutoff_for_diabetes.py), [cutpointfind__ibmi_cutoff_for_diabetes.py](cutpointfind__ibmi_cutoff_for_diabetes.py)) and to the `cutpointpy` documentation  for the available options.
    """)
    return


@app.cell
def _(CutpointCalculator):
    cpcalc = CutpointCalculator(target='youdenj', polarity=True, interpolation=None)
    return (cpcalc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now use `CutpointCalculator.bootstrap()` to perform optimal cut-point analysis with bootstrapping. Let's have a look at the function's arguments first.

    * `features`: the value of the predictor variable for each datapoint.
    * `labels`: the class label of each datapoint.

    The above arguments have the same meaning as in `CutpointCalculator.find()`. The following are specific to `CutpointCalculator.bootstrap()`.

    * `method`: a string indicating the sampling strategy used for generating the bootstrap repetitions — that is, how to split the original data into a *train* (or *in-bag*) and *test* (or *out-of-bag*) group at each run.
    * `train_ratio`: the fraction of the original data that goes to the train group (the complement to one goes to the test group). Say we have 100 datapoints and we set `train_ratio=0.6`, then at each repetition 60 datapoints will go to the train group and the remaining 40 to the test group.
    * `num_reps`: the number of bootstrap repetitions.
    * `random_state`: controls the randomness of the repetitions produced.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We invoke the method with the default parameters. Observe that:
    * `method='sss'` selects label-aware stratified shuffle splitting, meaning that at each repetion the ratio of positive vs. negative samples in the train and test set is the same as in the whole dataset;
    * `random_state=0` guarantees that the subdivisions into train and test set (repetitions) are the same across multiple calls (see also [What is Scikit-learn Random State in Splitting Dataset?](https://www.geeksforgeeks.org/machine-learning/what-is-scikit-learn-random-state-in-splitting-dataset/) on this).
    """)
    return


@app.cell
def _(cpcalc, df, feature_name):
    num_reps = 30
    train_ratio = 0.6

    cutpoints, cutpoints_idxs, thresholds, acc, ses, sps, aucs_train, aucs_test, performance_train, performance_test, performance_whole = cpcalc.bootstrap(
        features=df[feature_name], labels=df.Outcome, method='sss', train_ratio=train_ratio, num_reps=num_reps, random_state=0)
    return (
        aucs_test,
        aucs_train,
        cutpoints,
        num_reps,
        performance_test,
        performance_train,
        performance_whole,
        ses,
        sps,
        thresholds,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's have a look at the returned values and their meaning.

    * `cutpoints`: ndarray of float *(num_reps, 1)* contaning the optimal cut-point value for each repetition estimated on the train set.
    * `cutpoints_idxs` : ndarray of float *(num_reps, 1)* containing, for each repetition, the index of the optimal cut-point value in the `thresholds` array.
    * `thresholds` : ndarray of float *(num_reps, N)* containing the values of the thresholds tested on the train set at each repetition. N = floor(len(`features`) * `train_ratio`) if the `interpolation` argument passed to `CutpointCalculator()` is None, otherwise N is the value of `num_points` passed to `CutpointCalculator()`.
    * `accs`, `ses`, `sps` : one ndarray of float *(num_reps, N)* each, respectively storing accuracy, sensistivity and specificity on the train set for each repetition and threshold value tested.
    * `aucs_train` and `aucs_test`: two ndarrays of float, each of shape *(num_reps, 1)*, containing the area under the curve for each repetition estimated on the train and test set respectively.
    * `performance_train`, `performance_test` and `performance_whole` : three ndarrays of float, each of shape *(num_reps, 3)*, conatining, in column-wise order, the accuracy, sensitivity and specificity yielded by the optimal cut-point value when applied respectively to the train set, to the test set and to the whole dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For convenience, we pack the results into a pandas dataframe to show them in a compact way.
    """)
    return


@app.cell
def _(
    aucs_test,
    aucs_train,
    cutpoints,
    np,
    pd,
    performance_test,
    performance_train,
    performance_whole,
):
    results_dict = {'Cutpoint'  : cutpoints, 
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
    df_results = pd.DataFrame(data = np.hstack(list(results_dict.values())),
                              columns = results_dict.keys())
    df_results.head()
    return (df_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we present the results in aggregated form by showing mean, standard deviation and 95% confidence intervals.
    """)
    return


@app.cell
def _(df_results, np):
    #Estimate extremes of 95% confidence intervals
    def ci_l(x):
        return np.percentile(a=x, q=5.0)

    def ci_u(x):
        return np.percentile(a=x, q=95.0)

    df_results_agg = df_results.aggregate(func=['mean', 'std', ci_l, ci_u]).T
    df_results_agg.head(df_results_agg.shape[0])
    return (df_results_agg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, `auc` is a scalar storing the area under the receiver-operating characteristic (ROC) curve.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Show the results graphically
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot sensitivity and specificity vs. glucose level

    NOTE: as an example we generate the plots for the results obtained on the train set. Of course similar charts can be created, with the same procedure, for the results obatined on the test set and on the whole dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Colours and *y*-axis labels for each metric shown in the chart.
    """)
    return


@app.cell
def _(ses, sps):
    metrics = [ses, sps]
    colours = ['tab:blue', 'tab:orange']
    ylabels = ['Sensitivity', 'Specificity']
    return colours, metrics, ylabels


@app.cell
def _(
    Rectangle,
    colours,
    df_results_agg,
    feature_name,
    metrics,
    num_reps,
    plt,
    sns,
    thresholds,
    units,
    ylabels,
):
    _grid_spec = {'hspace': 0.3}
    _fig, _axes = plt.subplots(nrows=2, gridspec_kw=_grid_spec)

    cutpoint_mean, cutpoint_ci_l, cutpoint_ci_u = [
        df_results_agg.loc['Cutpoint', x] for x in ['mean', 'ci_l', 'ci_u']
    ]

    for _ax, _metric, _colour, _ylabel in zip(_axes, metrics, colours, ylabels):

        sns.lineplot(x=thresholds.flatten(), y=_metric.flatten(), color=_colour, errorbar=('pi', 95), ax=_ax)
        _ax.set_ylabel(_ylabel)
        _ax.set_xlabel(None)
        _ax.grid(visible=True)

        #Highlight the average optimal cut-point value with a vertical bar
        _ax.vlines(x=cutpoint_mean, 
                   ymin=_ax.get_ylim()[0], 
                   ymax=_ax.get_ylim()[1], 
                   color='tab:gray'
                  )

        #Draw confidence interval for optimal cut-point
        _bottom_left = (cutpoint_ci_l, _ax.get_ylim()[0])
        _width = cutpoint_ci_u - cutpoint_ci_l
        _height = _ax.get_ylim()[1] - _ax.get_ylim()[0]
        _ax.add_patch(Rectangle(_bottom_left, _width, _height, facecolor = 'lightgray'))

    _axes[1].set_xlabel(f'{feature_name} [{units[feature_name]}]')

    _axes[0].set_title(f'Optimal cut-point: {cutpoint_mean:.1f} {units[feature_name]} [{cutpoint_ci_l:.1f}-{cutpoint_ci_u:.1f}], '
                       f'Num. runs: {num_reps}')

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The shaded areas represent the 95% interval for sensitivity, specificity and optimal cut-point value over the bootstrap runs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot ROC curve and show AUC estimated on the train set
    """)
    return


@app.cell
def _(df_results_agg, plt, ses, sns, sps):
    _fig, _ax = plt.subplots()

    auc_mean, auc_ci_l, auc_ci_u =  [
        df_results_agg.loc['AUC_train', x] for x in ['mean', 'ci_l', 'ci_u']
    ]

    sns.lineplot(x=1-sps.flatten(), y=ses.flatten(), errorbar=('pi', 95), ax=_ax)

    _ax.set_title(f'Receiver-operating characteristic curve (AUC = {auc_mean:.3f} [{auc_ci_l:.3f}-{auc_ci_u:.3f}])')
    _ax.set_ylabel('Sensitivity')
    _ax.set_xlabel('1 - specificity')
    _ax.grid(visible=True)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The shaded areas represent the 95% interval for the AUC estimated on ther train set over the bootstrap runs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.
    2. Baratloo, A., Hosseini, M., Negida, A., El Ashal, G. (2015). [Part 1: Simple definition and calculation of accuracy, sensitivity and specificity](https://pmc.ncbi.nlm.nih.gov/articles/PMC4614595/). Emergency; 3(2):48-49
    3. Bianconi, F. (2024). [Data and process visualisation for graphic communication: A hands-on approach with Python](https://doi.org/10.1007/978-3-031-57051-3). Springer Cham, Switzerland.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Licence
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ©2026 [Francesco Bianconi](www.bianconif.net). Unless otherwise specified the material presented in this notebook is available under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.txt]) (code) and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en) (Text & media).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
