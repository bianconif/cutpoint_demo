import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    return np, pd, plt, sns


@app.cell
def _():
    from mpl_ornaments.titles import set_title_and_subtitle

    return (set_title_and_subtitle,)


@app.cell
def _():
    import os
    print(os.getcwd())
    return


@app.cell
def _():
    from cutpointpy.core import CutpointCalculator

    return (CutpointCalculator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimal cut-point value of glucose level for predicting diabetes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook shows how to use `cutpointpy` for estimating the optimal cut-off value of plasma glucose concentration at two hours in an oral glucose tolerance test (GTIT) to predict diabetes mellitus. The data come from the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), available on Kaggle ([CC0: Public Domain](https://creativecommons.org/public-domain/)).
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
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Store units here
    """)
    return


@app.cell
def _():
    units = {'Glucose': 'mg/dL'}
    return (units,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Show the data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's use combined strip and plots to show the distribution of the glucose level in the positive and negative class.
    """)
    return


@app.cell
def _(df, plt, set_title_and_subtitle, sns, units):
    _fig, _ax = plt.subplots(figsize=(6,4))
    feature = 'Glucose'

    chart_kw = {'data': df, 'y': 'Outcome', 'x': feature, 'hue': 'Outcome', 'orient': 'h', 'linewidth': 1, 'legend': False, 'ax': _ax}

    sns.boxplot(**chart_kw, showfliers=False, boxprops={'alpha': 0.4})
    sns.stripplot(**chart_kw, edgecolor='auto', jitter=0.15)

    _ax.grid(visible=True, axis='x')
    _ax.spines[['top', 'bottom', 'right']].set_visible(False)
    _ax.set_xlabel(f'{feature} [{units[feature]}]')

    set_title_and_subtitle(fig=_fig, title='Glucose level in the two groups', subtitle='Positive (1) vs. negative (0) cases.', h_offset=30)

    _fig
    return (feature,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: Perform optimal cut-point analysis with the default settings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As a first step we crate an instance of `CutpointCalculator` by invoking the constructor with the default parameter values. Let's have a look at their meaning.

    * `target`: selects the objective function to maximise or minimise. The default parameter is `youdenj`, meaning that the optimal cut-point is the threshold value that maximises Youden's J index — that is, sensitivity + specificity - 1.
    * `polarity`: defines the direction of the inequality. If `True` (`False`) all datapoints with value *greater than* (*less than*) or equal to the tested threshold are flagged as positive, and the others as negative.
    * `interpolation`: determines the set of thresholds to be tested as optimal cut-points. If `None` the thresholds tested are the values of each datapoint.

    NOTE: from now on we use `_dft` postfix to indicate variables generated with the default settings.
    """)
    return


@app.cell
def _(CutpointCalculator):
    cpcalc_dft = CutpointCalculator(target='youdenj', polarity=True, interpolation=None)
    return (cpcalc_dft,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we use `CutpointCalculator.find()` to determine the cut-off value that best separates the positive (diabetes) from negative (not-diabetes) cases. The accepted parameters are:

    * `features`: the value of the predictor variable for each datapoint.
    * `labels`: the class label of each datapoint.
    """)
    return


@app.cell
def _(cpcalc_dft, df):
    opt_cutpoint_dft, cutpoint_idx_dft, thresholds_dft, acc_dft, se_dft, sp_dft, auc_dft = cpcalc_dft.find(features=df.Glucose, labels=df.Outcome)
    return (
        acc_dft,
        auc_dft,
        cutpoint_idx_dft,
        opt_cutpoint_dft,
        se_dft,
        sp_dft,
        thresholds_dft,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's have a look at the returned values. For a start, `opt_cutpoint` and `cutpoint_idx` are scalars respectively representing the optimal cut-off value we are after and the index of the `thresholds` array where the target function reached the optimum:
    """)
    return


@app.cell
def _(cutpoint_idx_dft, opt_cutpoint_dft):
    for _item in [opt_cutpoint_dft, cutpoint_idx_dft]:
        print(_item)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we have `thresholds`, `acc`, `se` and `se`, which respectively store the thresholds tested, and the accuracy, sensitivity and specificity for each threshold tested. Note that these are all `numpy.ndarray` of shape *(N,1)*, where *N* is the number of thresholds tested. Let's pack these returned values into a `pd.DataFrame` for a better look.
    """)
    return


@app.cell
def _(acc_dft, np, pd, se_dft, sp_dft, thresholds_dft):
    df_res_dft = pd.DataFrame(data=np.concatenate((thresholds_dft, acc_dft, se_dft, sp_dft), axis=1),
                              columns=['Threshold', 'Acc', 'Se', 'Sn'])
    df_res_dft.head()
    return (df_res_dft,)


@app.cell
def _(df_res_dft):
    df_res_dft.tail()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now show the accuracy, sensitivity and specificity yielded by the optimal cut-point.
    """)
    return


@app.cell
def _(acc_dft, cutpoint_idx_dft, se_dft, sp_dft):
    for _key, _val in {'Acc': acc_dft, 'Se': se_dft, 'Sp': sp_dft}.items():
        print(f'{_key}: {_val[cutpoint_idx_dft]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, `auc` is a scalar storing the area under the receiver-operating characteristic (ROC) curve.
    """)
    return


@app.cell
def _(auc_dft):
    print(auc_dft)
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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Colours and *y*-axis labels for each metric shown in the chart.
    """)
    return


@app.cell
def _(se_dft, sp_dft):
    metrics = [se_dft, sp_dft]
    colours = ['tab:blue', 'tab:orange']
    ylabels = ['Sensitivity', 'Specificity']
    return colours, metrics, ylabels


@app.cell
def _(
    colours,
    cutpoint_idx_dft,
    feature,
    metrics,
    plt,
    thresholds_dft,
    units,
    ylabels,
):
    _grid_spec = {'hspace': 0.3}
    _fig, _axes = plt.subplots(nrows=2, gridspec_kw=_grid_spec)

    for _ax, _metric, _colour, _ylabel in zip(_axes, metrics, colours, ylabels):
        _ax.plot(thresholds_dft, _metric, marker='o', color=_colour, markeredgecolor='grey')

        _ax.set_ylabel(_ylabel)
        _ax.grid(visible=True)

        #Highlight the optimal cut-point value with a vertical bar
        _ax.vlines(x=thresholds_dft[cutpoint_idx_dft, 0], 
                   ymin=_ax.get_ylim()[0], 
                   ymax=_ax.get_ylim()[1], 
                   color='tab:gray'
                  )

        _axes[1].set_xlabel(f'{feature} [{units[feature]}]')

        _axes[0].set_title(f'Optimal cut-point: {thresholds_dft[cutpoint_idx_dft, 0]:.1f}, '
                           f'num. thresholds tested: {thresholds_dft.size}')

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot ROC curve and show AUC
    """)
    return


@app.cell
def _(auc_dft, cutpoint_idx_dft, plt, se_dft, sp_dft):
    _fig, _ax = plt.subplots()
    _ax.plot(1 - sp_dft, se_dft)

    _ax.set_title(f'Receiver-operating characteristic curve (AUC = {auc_dft:.3f})')
    _ax.set_ylabel('Sensitivity')
    _ax.set_xlabel('1 - specificity')
    _ax.grid(visible=True)


    #Place a marker corresponding to the point where the tarket is maximum
    _ax.plot(1 - sp_dft[cutpoint_idx_dft, 0], se_dft[cutpoint_idx_dft, 0], marker='o')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compare the results with those obtained with a third-party calculator
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check our results with those generated through [MetricGate](https://metricgate.com)'s [Optimal Cutpoint Analysis](https://metricgate.com/calculator/optimal-cutpoint-analysis) tool (please find the results [here](https://metricgate.com/shared/cutpointbootglucosecutofffordiabetesdft-f274d2)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2: Perform optimal cut-point analysis with custom settings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Again, we start by creating an instance of `CutpointCalculator`, but we invoke the constructor with custom parameters in this case. Let's have a look at the values.

    * `target='eucdist'`: the optimal cut-point is the value that minimises the distance between the receiver-operating characteristic curve and the *(0,1)* point in the ROC space.
    * `interpolation='linear'`: the thresholds tested are obtained by piecewise linear, uniform interpolation over the original feature values.
    * `num_points=1000`: the number of points returned by the interpolation — i.e., the interpolated thresholds to be tested as optimal cut-points. If `num_points` is greater than the number of the original datapoints we are upsampling the original data, otherwise we are downsampling them.

    NOTE: from now on we use the `_cst` postfix to indicate variables generated with the above custom settings.
    """)
    return


@app.cell
def _(CutpointCalculator):
    cpcalc_cst = CutpointCalculator(target='eucdist', polarity=True, interpolation='linear', num_points=1000)
    return (cpcalc_cst,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now proceed as in the first part to perform the cut-point analysis and show the results.
    """)
    return


@app.cell
def _(cpcalc_cst, df):
    opt_cutpoint_cst, cutpoint_idx_cst, thresholds_cst, acc_cst, se_cst, sp_cst, auc_cst = cpcalc_cst.find(features=df.Glucose, labels=df.Outcome)
    return (
        acc_cst,
        auc_cst,
        cutpoint_idx_cst,
        opt_cutpoint_cst,
        se_cst,
        sp_cst,
        thresholds_cst,
    )


@app.cell
def _(auc_cst, cutpoint_idx_cst, opt_cutpoint_cst, se_cst, sp_cst):
    opt_cutpoint_cst, cutpoint_idx_cst, auc_cst, se_cst[cutpoint_idx_cst], sp_cst[cutpoint_idx_cst]
    return


@app.cell
def _(acc_cst, np, pd, se_cst, sp_cst, thresholds_cst):
    df_res_cst = pd.DataFrame(data=np.concatenate((thresholds_cst, acc_cst, se_cst, sp_cst), axis=1),
                              columns=['Threshold', 'Acc', 'Se', 'Sn'])
    df_res_cst.head()
    return (df_res_cst,)


@app.cell
def _(df_res_cst):
    df_res_cst.tail()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compare the results with those obtained with a third-party calculator
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check the results with those provided by [MetricGate](https://metricgate.com) (available [here](https://metricgate.com/shared/glucosecutpointdiabeteseucdist-33474f))
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Optionally, reuse the code presented in Part 1 for generating the plots.
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
    1. Baratloo, A., Hosseini, M., Negida, A., El Ashal, G. (2015). Part 1: Simple definition and calculation of accuracy, sensitivity and specificity. Emergency; 3(2):48-49
    2. Bianconi, F. (2024). [Data and process visualisation for graphic communication: A hands-on approach with Python](https://doi.org/10.1007/978-3-031-57051-3). Springer Cham, Switzerland.
    3. Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.
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
