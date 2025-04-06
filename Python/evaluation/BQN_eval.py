import numpy as np
import os
import pandas as pd
import scipy.stats as sps
import properscoring as ps
import json
import sys
import ast
import re

print('\n\n')
print(sys.executable)
try:
    data = pd.read_csv('../../Datasets/DE.csv', index_col=0)
except:
    data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)

distribution = 'Normal'
num_runs = 4
d_degree = 12
quantile_array = np.arange(0.01, 1, 0.01)

def pinball_score(observed, pred_quantiles):
    quantiles = np.arange(0.01, 1, 0.01)
    scores = pd.Series(np.maximum((1 - quantiles) * (pred_quantiles - observed), quantiles * (observed - pred_quantiles)))
    return scores.mean()

B = np.zeros((d_degree+1, 99))
for d in range(d_degree+1):
    B[d, :] = sps.binom.pmf(d, d_degree, quantile_array)
def bern_quants(alphas):
    return np.dot(alphas, B)

quant_dfs = []

#load data
for num in range(num_runs):

    if num_runs == 1:
        file_path = f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_2'
    else:
        file_path = f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_{num+1}'
    dist_file_list = sorted(os.listdir(file_path))
    print(file_path)

    fc_df = pd.DataFrame()
    for day, file in enumerate(dist_file_list):
        with open(os.path.join(file_path, file)) as f:
            fc = pd.read_csv(f, index_col=0)
        fc_df = pd.concat([fc_df, fc], axis=0)

    quant_dfs.append(fc_df.add_suffix(f'_{num+1}'))

#format data right
for num, df in enumerate(quant_dfs):
    df[f'alphas_{num + 1}'] = df[f'alphas_{num + 1}'].apply(lambda x: re.sub(r'\[\s+', '[', x))
    df[f'alphas_{num + 1}'] = df[f'alphas_{num + 1}'].apply(lambda x: x.replace(' ', ','))
    df[f'alphas_{num + 1}'] = df[f'alphas_{num + 1}'].apply(lambda x: re.sub(',+', ',', x))
    df[f'alphas_{num + 1}'] = df[f'alphas_{num + 1}'].apply(ast.literal_eval)
    df[f'alphas_{num + 1}'] = df[f'alphas_{num + 1}'].apply(lambda x: np.array(x))
    quant_dfs[num] = df

data.index = pd.to_datetime(data.index)
#compute individual scores
for num, df in enumerate(quant_dfs):
    mean_series = df[f'alphas_{num+1}'].apply(lambda x: np.mean(x))
    y = data.loc[df.index, 'Price']

    quantiles = df[f'alphas_{num+1}'].apply(bern_quants)

    crps_obs = [pinball_score(obs, np.array(pred)) for obs, pred in zip(y, quantiles)]
    CRPS = np.mean(crps_obs)
    median_series = quantiles.apply(lambda x: x[50])
    mae = np.abs(y.values - median_series).mean()
    rmse = np.sqrt((y.values - mean_series) ** 2).mean()
    print(f'\n\nCRPS for trial {num+1}: {CRPS}')
    print(f'MAE for trial {num+1}: {mae}')
    print(f'RMSE for trial {num+1}: {rmse}')

#horizontal (quantile) averaging (q-Ens)
all_df = quant_dfs[0]
for df in quant_dfs[1:]:
    all_df = pd.concat([all_df, df], axis=1)

qEns_alphas = all_df.apply(np.mean, axis=1)
qEns_quantiles = qEns_alphas.apply(bern_quants)
qEns_crps_obs = [pinball_score(obs, np.array(pred)) for obs, pred in zip(y, qEns_quantiles.values)]
qEns_CRPS = np.mean(qEns_crps_obs)
median_series = qEns_quantiles.apply(lambda x: x[49])
rmse = np.sqrt(((y.values - qEns_alphas.apply(np.mean)) ** 2).mean())
mae = np.abs(y.values - median_series).mean()

print('\n\nq-Ens MAE: ' + str(mae))
print('q-Ens RMSE: ' + str(rmse))
print('q-Ens CRPS: ' + str(np.mean(qEns_CRPS)) )

#if not os.path.exists(f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_q-Ens'):
#    os.mkdir(f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_q-Ens')

#qEns_quantiles.to_csv(f'/home/ahaas/BachelorThesis/forecasts_probNN_BQN_q-Ens/predictions.csv')

#vertical (probability) averaging (p-Ens)
def inverse_cdf_interpolated(p, quantiles, values):
    return np.interp(p, quantiles, values)

sample_size = 500

samples_list = []

for num, df in enumerate(quant_dfs):
    uniform_samples = np.random.rand(sample_size)
    quantiles = df[f'alphas_{num + 1}'].apply(bern_quants)

    sample = np.empty((sample_size, len(quantiles)))

    for i, pred in enumerate(quantiles.values):
        samp = inverse_cdf_interpolated(uniform_samples, quantile_array, pred)
        sample[:, i] = samp

    sample = sample.T
    samples_list.append(sample)

samples = np.concatenate(samples_list, axis=1)
p_EnsQuantiles = np.percentile(samples, quantile_array*100, axis=1)
median_array = p_EnsQuantiles[49, :]
mean_array = np.mean(samples, axis=1)

pEns_crps_obs = [pinball_score(obs, pred) for obs, pred in zip(y, p_EnsQuantiles.T)]
mae = np.abs(y.values - median_array).mean()
rmse = np.sqrt(((y.values - mean_array) ** 2).mean())
print(f'\n\np-Ens MAE: {mae}')
print(f'p-Ens RMSE: {rmse}')
print(f'p-Ens CRPS: {np.mean(pEns_crps_obs)}')
