import numpy as np
import os
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import json
import re
import ast

#define evaluated models
fcs = {'q-Ens probNN-JSU': '/home/ahaas/BachelorThesis/forecasts_probNN_jsu_q-Ens',
       'q-Ens leadNN-JSU':  f'/home/ahaas/BachelorThesis/forecasts_leadNN_jsu_q-Ens',
       'q-Ens singleNN-JSU 4 trials': '/home/ahaas/BachelorThesis/forecasts_singleNN_jsu_q-Ens',
       'q-Ens BQN': '/home/ahaas/BachelorThesis/forecasts_probNN_BQN_q-Ens',
       'q-Ens singleNN-JSU 3 trials':  '/home/ahaas/BachelorThesis/forecasts_singleNN2_jsu_q-Ens',
       }
agg = 'lead'
#get data
data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data.index = pd.to_datetime(data.index)
quantile_array = np.arange(0.01, 1, 0.01)

def pinball_score(observed, pred_quantiles):
    quantiles = np.arange(0.01, 1, 0.01)
    scores = pd.Series(np.maximum((1 - quantiles) * (pred_quantiles - observed), quantiles * (observed - pred_quantiles)))
    return scores.mean()


fig, axs = plt.subplots(1, len(fcs), figsize=(20, 6), dpi=300, sharey=True)
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

    df.index = pd.to_datetime(df.index)
    y = data.loc[df.index, 'Price']

    #calculate scores and cap them
    crps_observations = np.array([pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, df[f'forecast_quantiles'])])
    ae_observations = np.abs(y - df[f'forecast_quantiles'].apply(lambda x: x[49]))
    crps_observations[crps_observations > 100] = 100
    ae_observations[ae_observations > 250] = 250

    df['crps'] = crps_observations
    df['mae'] = ae_observations

    if agg.lower() == 'lead':
        crps_lead = df['crps'].groupby(df.index.hour).mean()
        mae_lead = df['mae'].groupby(df.index.hour).mean()
        y_mean = y.groupby(y.index.hour).mean()
        y_std = y.groupby(y.index.hour).std()
    elif agg.lower() == 'day':
        crps_lead = df['crps'].groupby(df.index.weekday).mean()
        mae_lead = df['mae'].groupby(df.index.weekday).mean()
        y_mean = y.groupby(y.index.weekday).mean()
        y_std = y.groupby(y.index.weekday).std()
    else:
        raise ValueError('Wrong Aggregation scheme')

    bars = np.arange(len(crps_lead))
    bar_width = 0.3
    bar_offset = np.arange(len(crps_lead)) * (1)

    axs[idx].bar(bar_offset - bar_width / 2, mae_lead, width=bar_width, label='MAE', color='lightgrey')
    axs[idx].bar(bar_offset + bar_width / 2, crps_lead, width=bar_width, label='CRPS', color='blue')
    axs[idx].set_title(model, fontsize=14, fontweight='bold')

    if agg.lower() == 'day':
        wochentage = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axs[idx].set_xticks(bar_offset)
        axs[idx].set_xticklabels(wochentage)

    ax2 = axs[idx].twinx()
    ax2.plot(bars, y_mean, label='Mean price', linestyle='--', color='red')
    ax2.plot(bars, y_std, label='Standard deviation price', linestyle='--', color='orange')
    ax2.set_ylim(bottom=0)

handles1, labels1 = axs[idx].get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()


combined_handles = handles1 + handles2
combined_labels = labels1 + labels2

fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.legend(combined_handles, combined_labels, loc='lower center', ncol=len(combined_handles), bbox_to_anchor=(0.5, -0.01), fontsize=14)

plt.show()

fix, ax = plt.subplots(dpi=300, figsize=(16, 9))
for idx, (model, filepath) in enumerate(fcs.items()):
    if model.startswith('q-Ens singleNN') or model.startswith('q-Ens probNN'):
        df = pd.read_csv(os.path.join(filepath, 'predictions.csv'), index_col=0)
        df = df.rename(columns={'0': 'forecast_quantiles'})
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: re.sub(r'\[\s+', '[', x))
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: x.replace(' ', ','))
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: re.sub(',+', ',', x))
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(ast.literal_eval)
        df[f'forecast_quantiles'] = df[f'forecast_quantiles'].apply(lambda x: np.array(x))

        df.index = pd.to_datetime(df.index)
        y = data.loc[df.index, 'Price']

        crps_observations = np.array(
            [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, df[f'forecast_quantiles'])])
        crps_observations[crps_observations > 100] = 100
        df['crps'] = crps_observations

        crps_lead = df['crps'].groupby(df.index.hour).mean()


        ax.plot(range(24), crps_lead, label = model, linewidth=2, marker='o', linestyle='--')

plt.xlabel('Lead Time', fontsize=14)
plt.ylabel('CRPS value', fontsize=14)
plt.xticks(range(24), fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which='both', axis='y', color='lightgray', linestyle='--', linewidth=1.5)
legend = plt.legend(fontsize=14)
plt.tight_layout()
plt.show()