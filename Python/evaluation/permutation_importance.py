import pandas as pd
import os
import json
from datetime import datetime
import numpy as np
import scipy.stats as sps
import re
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

distribution = 'JSU'
baseline_path = '/home/ahaas/BachelorThesis/forecasts_probNN_jsu_q-Ens'
hourly = True
#function for loading data
def get_df(path):
    if not os.path.exists(path):
        print(f'Path does not exist: {path}\nReturning')
        return

    dist_file_list = sorted(os.listdir(path))
    dist_params = pd.DataFrame()
    for file in dist_file_list:
        with open(os.path.join(path, file), 'r') as f:
            fc_dict = json.load(f)

        fc_df = pd.DataFrame(fc_dict)
        fc_df.index = pd.date_range(start=file, periods=len(fc_df), freq='H')
        dist_params = pd.concat([dist_params, fc_df])
    return dist_params

quantile_array = np.arange(0.01, 1, 0.01)
def pinball_score(observed, pred_quantiles):
    observed = np.asarray(observed)
    pred_quantiles = np.asarray(pred_quantiles)

    if observed.ndim == 0:
        observed_expanded = observed[np.newaxis]
    else:
        observed_expanded = observed[:, np.newaxis]

    losses = np.maximum((1 - quantile_array) * (pred_quantiles - observed_expanded),
                        quantile_array * (observed_expanded - pred_quantiles))
    return np.mean(losses, axis=-1)

data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data.index = pd.to_datetime(data.index)

#load baseline model
baseline = pd.read_csv(os.path.join(baseline_path, 'predictions.csv'), index_col=0)
baseline = baseline.rename(columns={'0': 'forecast_quantiles'})
baseline[f'forecast_quantiles'] = baseline[f'forecast_quantiles'].apply(lambda x: re.sub(r'\[\s+', '[', x))
baseline[f'forecast_quantiles'] = baseline[f'forecast_quantiles'].apply(lambda x: x.replace(' ', ','))
baseline[f'forecast_quantiles'] = baseline[f'forecast_quantiles'].apply(lambda x: re.sub(',+', ',', x))
baseline[f'forecast_quantiles'] = baseline[f'forecast_quantiles'].apply(ast.literal_eval)
baseline[f'forecast_quantiles'] = baseline[f'forecast_quantiles'].apply(lambda x: np.array(x))
baseline.index = pd.to_datetime(baseline.index)

y = data.loc[baseline.index, 'Price']

if hourly:
    param_dfs = []
    for hour in range(1, 25):
        ens_dfs = []
        for trial in range(1, 5):
            path = f'/home/ahaas/BachelorThesis/distparams_probNN_{distribution.lower()}_{trial}_R_1{hour}'
            param_df = get_df(path)
            ens_dfs.append(param_df)
        param_dfs.append(ens_dfs)
else:
    param_dfs = []
    feature_groups = ['Y', 'L', 'R']
    for feature_group in feature_groups:
        ens_dfs = []
        for trial in range(1, 5):
            path = f'/home/ahaas/BachelorThesis/distparams_probNN_{distribution.lower()}_{trial}_{feature_group}'
            param_df = get_df(path)
            ens_dfs.append(param_df)
        param_dfs.append(ens_dfs)

if distribution.lower() == 'jsu':
    #generate ensemble forecasts
    ensembles = np.empty((len(param_dfs), len(y), 99))
    for num, ens in enumerate(param_dfs):

        #gettin quantiles for each ensemble member
        data_3d = np.array([df.to_numpy() for df in ens])
        quantiles_3d = np.empty((*data_3d.shape[:2], len(quantile_array)))
        for i, q in enumerate(quantile_array):
            quantiles_3d[:, :, i] = sps.johnsonsu.ppf(q, loc=data_3d[:, :, 0], scale=data_3d[:, :, 1],
                                                      a=data_3d[:, :, 3], b=data_3d[:, :, 2])
        #averaging quantiles to gain q-ens
        ensembles[num, :, :] = quantiles_3d.mean(axis=0)

    # calculate baseline crps
    crps_baseline = [pinball_score(observed, quantiles_row) for observed, quantiles_row in
                     zip(y, baseline[f'forecast_quantiles'])]
    print(f'Baseline CRPS: {np.mean(crps_baseline)}\n')

    #get crps for each ensemble prediction
    crps_ensembles = np.empty(ensembles.shape[:2])
    for i, ens in enumerate(ensembles):
        crps_ensembles[i, :] = pinball_score(y, ens)
        print(f'Ens {i+1} CRPS: {np.mean(crps_ensembles[i, :])}')
        pmi = (np.mean(crps_ensembles[i, :]) - np.mean(crps_baseline))/ np.mean(crps_baseline)
        print(f'PMI: {pmi}\n')

    #get dataframe with relative permutation importances
    crps_ensembles_df = pd.DataFrame(crps_ensembles.T, index=baseline.index)
    crps_baseline_df = pd.DataFrame(crps_baseline, index=baseline.index)

    #remove outliers
    outliers = [
        '2020-09-16 18:00:00',
        '2020-09-16 19:00:00',
        '2020-09-16 20:00:00',
        '2020-09-21 19:00:00',
        '2020-09-22 18:00:00',
        '2020-09-22 19:00:00',
        '2020-09-22 20:00:00',
        '2020-09-24 19:00:00'
    ]
    outliers = pd.to_datetime(outliers)

    crps_baseline_df = crps_baseline_df.loc[~crps_baseline_df.index.isin(outliers)]
    crps_ensembles_df = crps_ensembles_df.loc[~crps_ensembles_df.index.isin(outliers)]

    crps_baseline_df = crps_baseline_df.groupby(crps_baseline_df.index.hour).mean()
    crps_ensembles_df = crps_ensembles_df.groupby(crps_ensembles_df.index.hour).mean()


    pmi = crps_ensembles_df.sub(crps_baseline_df.iloc[:, 0], axis=0)
    pmi = pmi.div(crps_baseline_df.iloc[:, 0], axis=0)
    pmi = pmi.iloc[::-1]

    #define normalizer
    class PiecewiseNorm(mcolors.Normalize):
        def __init__(self, threshold, vmin=None, vmax=None):
            super().__init__(vmin, vmax)
            self.threshold = threshold

        def __call__(self, value, clip=None):
            # Skaliere die Daten stückweise: linear unterhalb des Schwellwerts, logarithmisch darüber
            x, y = [self.vmin, self.threshold, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))


    norm = PiecewiseNorm(15, vmin=0, vmax=75)

    #plot heatmap
    plt.figure(figsize=(9, 9), dpi=300)
    ax = sns.heatmap(pmi.values*100, annot=False, cmap='flare', fmt=".1f", norm=norm, square=True, cbar_kws={'shrink': 0.75})
    ax.set_xticks(np.arange(0, pmi.values.shape[1], 6))
    ax.set_yticks(np.arange(0, pmi.values.shape[1], 6))

    ax.set_xticklabels(np.arange(0, pmi.values.shape[1], 6), fontsize=14)
    ax.set_yticklabels(np.arange(pmi.values.shape[0], 0, -6), fontsize=14)

    plt.xlabel('Inputs by lead time', fontsize=14)
    plt.ylabel('Relative Permutation Importance in %', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.show()

    #plot line plot
    plt.figure(figsize=(16, 9), dpi=450)
    pmi = pmi.iloc[::-1]

    plt.plot(range(24), pmi.iloc[:, 0]*100, label='Lagged DA Price', linewidth=2, marker='o', linestyle='--')
    plt.plot(range(24), pmi.iloc[:, 1]*100, label='Load Forecast', linewidth=2, marker='o', linestyle='--')
    plt.plot(range(24), pmi.iloc[:, 2]*100, label='Renewables Forecast', linewidth=2, marker='o', linestyle='--')
    plt.xlabel('Lead Time', fontsize=14)
    plt.ylabel('Relative Permutation Importance in %', fontsize=14)
    plt.xticks(range(24), fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', axis='y', color='lightgray', linestyle='--', linewidth=1.5)
    legend = plt.legend(fontsize=14)

    avg_values = [pmi.iloc[:, 0].mean() * 100, pmi.iloc[:, 1].mean() * 100, pmi.iloc[:, 2].mean() * 100]
    labels = ['Lagged DA Price Avg:', 'Load Forecast Avg:', 'Renewables Forecast Avg:']
    values = [f'{value:.2f}%' for value in avg_values]
    max_label_length = max(len(label) for label in labels)
    max_value_length = max(len(value) for value in values)

    avg_text_lines = [f'{label.rjust(max_label_length)} {value.rjust(max_value_length)}' for label, value in
                      zip(labels, values)]
    avg_text = '\n'.join(avg_text_lines)

    frame = legend.get_frame()
    props = {'boxstyle': 'round',
             'facecolor': frame.get_facecolor(),
             'edgecolor': frame.get_edgecolor(),
             'linestyle': frame.get_linestyle(),
             'linewidth': frame.get_linewidth(),
             'alpha': frame.get_alpha()}

    plt.gca().text(0.01, 0.99, avg_text, transform=plt.gca().transAxes, fontsize=14,
                   bbox=props, verticalalignment='top', horizontalalignment='left', family='monospace', linespacing=1.5)

    #plt.tight_layout()
    #plt.show()









