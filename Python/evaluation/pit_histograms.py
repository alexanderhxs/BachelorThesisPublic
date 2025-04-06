import numpy as np
import os
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import json
import re
import ast

#define evaluated models
fcs = {'JSU': f'/home/ahaas/BachelorThesis/distparams_singleNN1_jsu_4',
       #'q-Ens leadNN-JSU':  f'/home/ahaas/BachelorThesis/forecasts_leadNN_jsu_q-Ens',
       #'q-Ens singleNN-JSU': '/home/ahaas/BachelorThesis/forecasts_singleNN_jsu_q-Ens',
        #'q-Ens BQN': '/home/ahaas/BachelorThesis/forecasts_probNN_BQN_q-Ens'
       }
agg_window = 1
#get data
data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data.index = pd.to_datetime(data.index)
quantile_array = np.arange(0.01, 1, 0.01)

def find_bin(value, quantiles):
    for i, q in enumerate(quantiles):
        if value <= q:
            return i
    return len(quantiles)

B = np.zeros((12+1, 99))
for d in range(12+1):
    B[d, :] = sps.binom.pmf(d, 12, quantile_array)
def bern_quants(alphas):
    return np.dot(alphas, B)

def winkler_score(y, lower_bound, upper_bound, alpha):
    score = np.where(y > upper_bound, (upper_bound - lower_bound) + 2/alpha * (y - upper_bound),
                     np.where(y < lower_bound, (upper_bound - lower_bound) + 2/alpha * (lower_bound - y),
                              upper_bound - lower_bound))
    return score

fig, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=300, sharey=True)
axs = axs.flatten()
#get data from models, stored in df
for idx, (model, filepath) in enumerate(fcs.items()):

    if model.startswith('BQN'):
        dist_file_list = sorted(os.listdir(filepath))
        df = pd.DataFrame()
        for day, file in enumerate(dist_file_list):
            with open(os.path.join(filepath, file)) as f:
                fc = pd.read_csv(f, index_col=0)
            df = pd.concat([df, fc], axis=0)
        df[f'alphas'] = df[f'alphas'].apply(lambda x: re.sub(r'\[\s+', '[', x))
        df[f'alphas'] = df[f'alphas'].apply(lambda x: x.replace(' ', ','))
        df[f'alphas'] = df[f'alphas'].apply(lambda x: re.sub(',+', ',', x))
        df[f'alphas'] = df[f'alphas'].apply(ast.literal_eval)
        df[f'alphas'] = df[f'alphas'].apply(lambda x: np.array(x))
        df[f'forecast_quantiles'] = df[f'alphas'].apply(bern_quants)

    elif model.startswith('Normal'):
        df = pd.DataFrame()
        dist_file_list = sorted(os.listdir(filepath))
        for day, file in enumerate(dist_file_list):
            with open(os.path.join(filepath, file)) as f:
                fc_dict = json.load(f)
            fc_df = pd.DataFrame(fc_dict)
            fc_df.index = pd.date_range(pd.to_datetime(file), periods=len(fc_df), freq='H')
            df = pd.concat([df, fc_df], axis=0)

        quantiles = df.apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc'], scale=x[f'scale']), axis=1)
        df[f'forecast_quantiles'] = quantiles
        df = df.drop(['loc', 'scale'], axis=1)

    elif model.startswith('JSU'):
        df = pd.DataFrame()
        dist_file_list = sorted(os.listdir(filepath))
        for day, file in enumerate(dist_file_list):
            with open(os.path.join(filepath, file)) as f:
                fc_dict = json.load(f)
            fc_df = pd.DataFrame(fc_dict)
            fc_df.index = pd.date_range(pd.to_datetime(file), periods=len(fc_df), freq='H')
            df = pd.concat([df, fc_df], axis=0)

        quantiles = df.apply(
            lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc'], scale=x[f'scale'], a=x[f'skewness'],
                                        b=x[f'tailweight']), axis=1)
        df[f'forecast_quantiles'] = quantiles
        df = df.drop(['loc', 'scale'], axis=1)
    elif model.startswith('q-Ens'):
        df = pd.read_csv(os.path.join(filepath, 'predictions.csv'), index_col=0)
        df = df.rename(columns={'0': 'forecast_quantiles'})
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: re.sub(r'\[\s+', '[', x))
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: x.replace(' ', ','))
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: re.sub(',+', ',', x))
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(ast.literal_eval)
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: np.array(x))
    else:
        raise ValueError('ERROR: could not find ModelType')

    bin_counts = np.zeros(100)

    df.index = pd.to_datetime(df.index)
    for y, quantiles in zip(data.loc[df.index, 'Price'], df[f'forecast_quantiles']):
        bin_index = find_bin(y, quantiles)
        bin_counts[bin_index] += 1

    normalized_bin_counts = bin_counts/sum(bin_counts)
    agg_bin_counts = [sum(normalized_bin_counts[i:i + agg_window]) for i in range(0, len(normalized_bin_counts), agg_window)]

    bar_width = agg_window * 0.8
    offset = agg_window/2
    axs[idx].bar(np.arange(0, 100, agg_window) + offset, agg_bin_counts, width=bar_width, color='skyblue')
    axs[idx].set_ylabel('Share of bin', fontsize=14)
    axs[idx].set_xlabel('Percentile-level bin', fontsize=14)
    axs[idx].set_title(model, fontsize=14, fontweight='bold')
    axs[idx].tick_params(axis='both', which='major', labelsize=14)
    axs[idx].tick_params(axis='both', which='minor', labelsize=14)
    axs[idx].axhline(y=1/len(agg_bin_counts), color='coral', linestyle='--')

    ks_stat, p_val = sps.kstest(np.cumsum(agg_bin_counts), 'uniform')

    forecast_quantiles = df[f'forecast_quantiles']
    w_score_05 = winkler_score(y, forecast_quantiles.apply(lambda x: x[24]).values, forecast_quantiles.apply(lambda x: x[74]).values, 0.5)
    w_score_01 = winkler_score(y, forecast_quantiles.apply(lambda x: x[4]).values, forecast_quantiles.apply(lambda x: x[94]).values, 0.1)
    w_score_002 = winkler_score(y, forecast_quantiles.apply(lambda x: x[0]).values, forecast_quantiles.apply(lambda x: x[-1]).values, 0.02)

    print(f'\n\nTest for uniform distribution of model {model}:\np-value: {p_val}\nTest statistik: {ks_stat}\n')
    print(f'Coverage of the PI: {1-(agg_bin_counts[0] + agg_bin_counts[-1])}')
    print(f'Average Winkler score for nievau alpha = 0.5: {np.mean(w_score_05)}')
    print(f'Average Winkler score for nievau alpha = 0.1: {np.mean(w_score_01)}')
    print(f'Average Winkler score for nievau alpha = 0.02: {np.mean(w_score_002)}')


plt.tight_layout()
plt.show()
