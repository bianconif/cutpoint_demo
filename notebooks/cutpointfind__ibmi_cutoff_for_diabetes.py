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
    # Optimal cut-point value of inverse body-mass index (iBMI) level for predicting diabetes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook shows how to use `cutpointpy` for estimating the optimal iBMI to predict diabetes mellitus. The data come from the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), available on Kaggle ([CC0: Public Domain](https://creativecommons.org/public-domain/)).
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
    Add `iBMI` column (see [2] for a definition of inverse body mass index).
    """)
    return


@app.cell
def _(df):
    df['iBMI'] = 1000/df['BMI']
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Store units here.
    """)
    return


@app.cell
def _():
    units = {'iBMI': 'cm^2/kg'}
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
    Use combined strip and plots to show the distribution of the glucose level in the positive and negative class.
    """)
    return


@app.cell
def _(df, plt, set_title_and_subtitle, sns, units):
    _fig, _ax = plt.subplots(figsize=(6,4))
    feature = 'iBMI'

    chart_kw = {'data': df, 'y': 'Outcome', 'x': feature, 'hue': 'Outcome', 'orient': 'h', 'linewidth': 1, 'legend': False, 'ax': _ax}

    sns.boxplot(**chart_kw, showfliers=False, boxprops={'alpha': 0.4})
    sns.stripplot(**chart_kw, edgecolor='auto', jitter=0.15)

    _ax.grid(visible=True, axis='x')
    _ax.spines[['top', 'bottom', 'right']].set_visible(False)
    _ax.set_xlabel(f'{feature} [{units[feature]}]')

    set_title_and_subtitle(fig=_fig, title=f'{feature} in the two groups', subtitle='Diabetes (1) vs. not-diabetes (0) cases.', h_offset=30)

    _fig
    return (feature,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Perform optimal cut-point analysis (inverted polarity)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In a previous notebook of this series *link_here* we discussed the usage of `CutpointCalculator` and the arguments accepted by the constructor (please see the cited document for details). We shall now invoke the constructor with the default parameters except for the `polarity` argument.

    In this case study the polarity is indeed inverted — that is, higher values of iBMI are associated with a lower chance of getting diabetes and vice-versa. Hence we shall set `polarity=False` to reflect this.
    """)
    return


@app.cell
def _(CutpointCalculator):
    cpcalc = CutpointCalculator(polarity=False)
    return (cpcalc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now proceed as in the other notebook for the optimal cut-point analysis.
    """)
    return


@app.cell
def _(cpcalc, df):
    opt_cutpoint, cutpoint_idx, thresholds, acc, se, sp, auc = cpcalc.find(features=df.iBMI, labels=df.Outcome)
    return acc, auc, cutpoint_idx, opt_cutpoint, se, sp, thresholds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As usual we have a look at the returned values, starting with the scalar ones first.
    """)
    return


@app.cell
def _(auc, cutpoint_idx, opt_cutpoint):
    opt_cutpoint, cutpoint_idx, auc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the arry-like ones.
    """)
    return


@app.cell
def _(acc, np, pd, se, sp, thresholds):
    df_res = pd.DataFrame(data=np.concatenate((thresholds, acc, se, sp), axis=1),
                          columns=['Threshold', 'Acc', 'Se', 'Sn'])
    df_res.head()
    return (df_res,)


@app.cell
def _(df_res):
    df_res.tail()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's have a look at the accuracy, sensitivity and specificity yielded by the optimal cut-point.
    """)
    return


@app.cell
def _(acc, cutpoint_idx, se, sp):
    for key, val in {'Acc': acc, 'Se': se, 'Sp': sp}.items():
        print(f'{key}: {val[cutpoint_idx]}')
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
    #### Plot sensitivity and specificity vs. iBMI
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Colours and *y*-axis labels for each metric shown in the chart.
    """)
    return


@app.cell
def _(se, sp):
    metrics = [se, sp]
    colours = ['tab:blue', 'tab:orange']
    ylabels = ['Sensitivity', 'Specificity']
    return colours, metrics, ylabels


@app.cell
def _(
    colours,
    cutpoint_idx,
    feature,
    metrics,
    plt,
    thresholds,
    units,
    ylabels,
):
    _grid_spec = {'hspace': 0.3}
    _fig, _axes = plt.subplots(nrows=2, gridspec_kw=_grid_spec)

    for _ax, _metric, _colour, _ylabel in zip(_axes, metrics, colours, ylabels):
        _ax.plot(thresholds, _metric, marker='o', color=_colour, markeredgecolor='grey')

        _ax.set_ylabel(_ylabel)
        _ax.grid(visible=True)

        #Highlight the optimal cut-point value with a vertical bar
        _ax.vlines(x=thresholds[cutpoint_idx, 0], 
                   ymin=_ax.get_ylim()[0], 
                   ymax=_ax.get_ylim()[1], 
                   color='tab:gray'
                  )

        _axes[1].set_xlabel(f'{feature} [{units[feature]}]')

        _axes[0].set_title(f'Optimal cut-point: {thresholds[cutpoint_idx, 0]:.1f}, '
                           f'num. thresholds tested: {thresholds.size}')

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Observe that the shape of the sensitivity/specificity vs. feature value curves is mirrored compared to the ones we got for the glucose level analysis ([cutpointboot__glucose_cutoff_for_diabetes.py](cutpointboot__glucose_cutoff_for_diabetes.py)). This is what we expected, given that the polarity of the problem is inverted here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plot ROC curve and show AUC
    """)
    return


@app.cell
def _(auc, cutpoint_idx, plt, se, sp):
    _fig, _ax = plt.subplots()
    _ax.plot(1 - sp, se)

    _ax.set_title(f'Receiver-operating characteristic curve (AUC = {auc:.3f})')
    _ax.set_ylabel('Sensitivity')
    _ax.set_xlabel('1 - specificity')
    _ax.grid(visible=True)


    #Place a marker corresponding to the point where the tarket is maximum
    _ax.plot(1 - sp[cutpoint_idx, 0], se[cutpoint_idx, 0], marker='o')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check the results with a third-party calculator
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a comparison we can check our results with those generated through [MetricGate](https://metricgate.com)'s [Optimal Cutpoint Analysis](https://metricgate.com/calculator/optimal-cutpoint-analysis) tool (please find the results [here](https://metricgate.com/shared/cutpointfindibmicutofffordiabetes-7da98e)).
    """)
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
    ## References
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Baratloo, A., Hosseini, M., Negida, A., El Ashal, G. (2015). Part 1: Simple definition and calculation of accuracy, sensitivity and specificity. Emergency; 3(2):48-49
    2. Bianconi, F. (2024). [Data and process visualisation for graphic communication: A hands-on approach with Python](https://doi.org/10.1007/978-3-031-57051-3). Springer Cham, Switzerland.
    3. Nevill, A.M., Stavropoulos-Kalinoglou, A., Metsios, G.S., Koutedakis, Y., Holder, R.L., Kitas, G.D., Mohammed, M.A. (2011). Inverted BMI rather than BMI is a better proxy for percentage of body fat.  Ann Hum Biol, Nov;38(6):681-4.
    4. Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.
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


if __name__ == "__main__":
    app.run()
