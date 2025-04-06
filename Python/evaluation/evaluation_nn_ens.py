import numpy as np
import os
import pandas as pd
import scipy.stats as sps
import properscoring as ps
import json
import sys

print('\n\n')
print(sys.executable)
try:
    data = pd.read_csv('../../Datasets/DE.csv', index_col=0)
except:
    data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)

distribution = 'JSU'
num_runs = 1
outlier_threshold_crps = 100
outlier_threshold_mae = 250
outlier_threshold_rmse = outlier_threshold_mae**2
quantile_array = np.arange(0.01, 1, 0.01)

def pinball_score(observed, pred_quantiles):
    quantiles = np.arange(0.01, 1, 0.01)
    scores = pd.Series(np.maximum((1 - quantiles) * (pred_quantiles - observed), quantiles * (observed - pred_quantiles)))
    return scores.mean()

param_dfs = []

#load data
for num in range(num_runs):
    try:
        if num_runs == 1:
            file_path = f'/home/ahaas/BachelorThesis/distparams_singleNN1_{distribution.lower()}_4'
        else:
            file_path = f'/home/ahaas/BachelorThesis/distparams_singleNN1_{distribution.lower()}_{num + 1}'

        dist_file_list = sorted(os.listdir(file_path))
    except:
        if num_runs == 1:
            file_path = f'/home/ahaas/BachelorThesis/distparams_probNN1_{distribution.lower()}_1'
        else:
            file_path = f'/home/ahaas/BachelorThesis/distparams_leadNN3.2_{distribution.lower()}_{num + 1}'
        dist_file_list = sorted(os.listdir(file_path))
    print(file_path)

    dist_params = pd.DataFrame()
    for day, file in enumerate(dist_file_list):
        with open(os.path.join(file_path, file)) as f:
            fc_dict = json.load(f)

        fc_df = pd.DataFrame(fc_dict)
        fc_df.index = pd.date_range(pd.to_datetime(file), periods=len(fc_df), freq='H')
        dist_params = pd.concat([dist_params, fc_df])

    param_dfs.append(dist_params.add_suffix(f'_{num+1}'))

data.index = pd.to_datetime(data.index)

#set test period
for num, df in enumerate(param_dfs):
    param_dfs[num] = df.iloc[-24*554:, :]

if distribution.lower() == 'normal':
    for num, df in enumerate(param_dfs):
        num += 1

        quantiles = df.apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
        median_series = df.apply(lambda x: sps.norm.median(loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
        y = data.loc[df.index, 'Price']
        crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles)]

        mae = np.abs(y.values - median_series).mean()
        rmse = np.sqrt(((y.values - df[f'loc_{num}']) ** 2).mean())
        print(f'Run Nr {num}')
        print('Observations: ' + str(len(y)) + '\n')
        print('MAE: ' + str(mae) + '\n' + 'RMSE: ' + str(rmse))
        print('CRPS: ' + str(np.mean(crps_observations)) + '\n\n')


    #q-Ens averaging (horizontal) via parameter averaging
    all_df = param_dfs[0]
    for df in param_dfs[1:]:
        all_df = pd.merge(all_df, df, how='inner', left_index=True, right_index=True)

    average_dist_params = pd.DataFrame({'loc': all_df.iloc[:, ::2].mean(axis=1),
                                       'scale': all_df.iloc[:, 1::2].mean(axis=1)},
                                       index=all_df.index)
    y = data.loc[average_dist_params.index, 'Price']
    quantiles = average_dist_params.apply(lambda x: sps.norm.ppf(quantile_array, loc=x['loc'], scale=x['scale']), axis=1)
    median_series = average_dist_params.apply(lambda x: sps.norm.median(loc=x['loc'], scale=x['scale']), axis=1)
    qEns_crps_observations = np.array([pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles)])

    crps_accurate = ps.crps_gaussian(y, mu=average_dist_params['loc'], sig=average_dist_params['scale'])
    qEns_mae = np.abs(y.values - median_series.values).mean()
    qEns_rmse = np.sqrt(((y - average_dist_params['loc'].values) ** 2).mean())

    print('q-Ens MAE: ' + str(qEns_mae) + '\n' + 'q-Ens RMSE: ' + str(qEns_rmse))
    print('q-Ens CRPS: ' + str(np.mean(qEns_crps_observations)))
    print('q-Ens Accurate CRPS: ' + str(np.mean(crps_accurate)) + str('\n\n'))

    #q-Ens averaging (horizontal) via quantile averaging
    quantiles_runs = param_dfs[0].apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_1'], scale=x[f'scale_1']), axis=1)
    quantiles_runs = pd.DataFrame({'run_1': quantiles_runs})
    loc_runs = pd.DataFrame(param_dfs[0]['loc_1'])
    for num, df in enumerate(param_dfs[1:], start=2):
        qs = df.apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
        qs = pd.DataFrame({f'run_{num}': qs})
        quantiles_runs = pd.merge(quantiles_runs, qs, left_index=True, right_index=True, how='inner')
        loc_runs = pd.merge(loc_runs, df[f'loc_{num}'], left_index=True, right_index=True, how='inner')

    qEns_quantiles = quantiles_runs.apply(np.mean, axis=1)
    qEns_crps_observations2 = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, qEns_quantiles)]
    median_series = qEns_quantiles.apply(lambda x: x[49])
    mean_series = loc_runs.apply(np.mean, axis=1)

    # remove outliers
    print(qEns_crps_observations[qEns_crps_observations > outlier_threshold_crps])
    qEns_crps_observations[qEns_crps_observations > outlier_threshold_crps] = outlier_threshold_crps
    print(f'Outliers: {len(qEns_crps_observations[qEns_crps_observations >= outlier_threshold_crps])}')
    qEns_mae_obs = np.abs(y.values - median_series)
    qEns_mae_obs[qEns_mae_obs > outlier_threshold_mae] = outlier_threshold_mae
    qEns_mae = np.mean(qEns_mae_obs)
    qEns_rmse_obs = ((y - mean_series) ** 2)
    qEns_rmse_obs[qEns_rmse_obs > outlier_threshold_rmse] = outlier_threshold_rmse
    qEns_rmse = np.sqrt(qEns_rmse_obs.mean())

    print('q-Ens MAE: ' + str(qEns_mae))
    print('q-Ens RSME:' + str(qEns_rmse))
    print('q-Ens CRPS: ' + str(np.mean(qEns_crps_observations)))

    #p-Ens averaging (vertical) via sampling
    for sample_size in [50, 100, 250, 500, 1000, 2500]:
        samples = param_dfs[0].apply(lambda x: sps.norm.rvs(size=sample_size, loc=x['loc_1'], scale=x['scale_1']), axis=1)
        for num, df in enumerate(param_dfs[1:], start=2):
            sample = df.apply(lambda x: sps.norm.rvs(size=sample_size, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
            samples = pd.concat([samples, sample], join='inner', axis=1)

        samples.columns = [str(i) for i in range(len(samples.columns))]
        samples = samples.apply(np.concatenate, axis=1)
        pEns_quantiles = samples.apply(lambda x: np.percentile(x, quantile_array*100))
        pEns_crps_observations = np.array([pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, pEns_quantiles)])
        median_series = pEns_quantiles.apply(lambda x: x[49])
        mean_series = samples.apply(np.mean)

        pEns_crps_observations[pEns_crps_observations > outlier_threshold_crps] = outlier_threshold_crps
        pEns_mae_obs = np.abs(y.values - median_series)
        pEns_mae_obs[pEns_mae_obs > outlier_threshold_mae] = outlier_threshold_mae
        pEns_mae = np.mean(pEns_mae_obs)
        pEns_rmse_obs = ((y - mean_series) ** 2)
        pEns_rmse_obs[pEns_rmse_obs > outlier_threshold_rmse] = outlier_threshold_rmse
        pEns_rmse = np.sqrt(pEns_rmse_obs.mean())

        print(f'Results based on sample size of {sample_size} per distribution')
        print(f'p-Ens MAE: {pEns_mae}')
        print(f'p-Ens RMSE: {pEns_rmse}')
        print(f'p-Ens CRPS: {np.mean(pEns_crps_observations)} \n\n')

elif distribution.lower() == 'jsu':

    outliers = []
    for num, df in enumerate(param_dfs):
        num += 1
        y = data.loc[df.index, 'Price']
        quantiles = df.apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
        crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles)]

        #detect outliers
        df['crps'] = crps_observations
        outlier_mask = df['crps'] > outlier_threshold_crps
        print(df.loc[outlier_mask, 'crps'])
        df.loc[outlier_mask, 'crps'] = outlier_threshold_crps
        outliers.append(outlier_mask)

        #compute scores without outliers
        y = data.loc[df.index, 'Price']
        median_series = df.apply(lambda x: sps.johnsonsu.median(loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
        mean_series = df.apply(lambda x: sps.johnsonsu.mean(loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)

        #detect outliers for point forecasts
        mae_obs = np.abs(y.values - median_series)
        mae_obs[mae_obs > outlier_threshold_mae] = outlier_threshold_mae
        mae = np.mean(mae_obs)
        rmse_obs = ((y.values - mean_series) ** 2)
        rmse_obs[rmse_obs > outlier_threshold_rmse] = outlier_threshold_rmse
        rmse = np.sqrt(rmse_obs.mean())
        print(f'Run Nr {num}')
        print('Observations: ' + str(len(y)) + '\n')
        print('MAE: ' + str(mae) + '\n' + 'RMSE: ' + str(rmse))
        print(f'CRPS: {df["crps"].mean()}\n\n')

    #removing all outliers from runs
    '''
    outliers = np.any(outliers, axis=0)
    for i, df in enumerate(param_dfs):
        print(df.loc[outliers])
        param_dfs[i] = df.loc[~outliers]
    '''

    #q-Ens averaging (horizontal) via quantile averaging
    quantiles_runs = param_dfs[0].apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_1'], scale=x[f'scale_1'], a=x[f'skewness_1'], b=x['tailweight_1']),
                                        axis=1)
    quantiles_runs = pd.DataFrame({'run_1': quantiles_runs})
    mean_series = pd.DataFrame(param_dfs[0].apply(lambda x: sps.johnsonsu.mean(loc=x[f'loc_1'], scale=x[f'scale_1'], a=x[f'skewness_1'], b=x['tailweight_1']),
                                        axis=1))
    mean_series.columns = ['run_1']

    #collecting quantiles and means
    for num, df in enumerate(param_dfs[1:], start=2):
        qs = df.apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'],a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
        qs = pd.DataFrame({f'run_{num}': qs})
        quantiles_runs = pd.merge(quantiles_runs, qs, left_index=True, right_index=True, how='inner')
        means_trial = df.apply(lambda x: sps.johnsonsu.mean(loc=x[f'loc_{num}'], scale=x[f'scale_{num}'],a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
        means_trial_df = pd.DataFrame(means_trial, columns=[f'run_{num}'])
        mean_series = pd.merge(mean_series, means_trial_df, left_index=True, right_index=True, how='inner')

    #compute ensembles
    qEns_quantiles = quantiles_runs.apply(np.mean, axis=1)
    y = data.loc[qEns_quantiles.index, 'Price']
    qEns_crps_observations =np.array([pinball_score(observed, quantiles_row) for observed, quantiles_row in
                              zip(y, qEns_quantiles)])
    median_series = qEns_quantiles.apply(lambda x: x[49])
    mean_series = np.mean(mean_series, axis=1)

    #remove outliers
    print(qEns_crps_observations[qEns_crps_observations > outlier_threshold_crps])
    qEns_crps_observations[qEns_crps_observations > outlier_threshold_crps] = outlier_threshold_crps
    print(f'Outliers: {len(qEns_crps_observations[qEns_crps_observations >= outlier_threshold_crps] )}')
    qEns_mae_obs = np.abs(y.values - median_series)
    qEns_mae_obs[qEns_mae_obs > outlier_threshold_mae] = outlier_threshold_mae
    qEns_mae = np.mean(qEns_mae_obs)
    qEns_rmse_obs = ((y - mean_series) ** 2)
    qEns_rmse_obs[qEns_rmse_obs > outlier_threshold_rmse] = outlier_threshold_rmse
    qEns_rmse = np.sqrt(qEns_rmse_obs.mean())

    print('q-Ens MAE: ' + str(qEns_mae))
    print('q-Ens RSME: ' + str(qEns_rmse))
    print('q-Ens CRPS: ' + str(np.mean(qEns_crps_observations)))

    # p-Ens averaging (vertical) via sampling
    for sample_size in [50, 100, 250, 500, 1000, 2500]:
        samples = param_dfs[0].apply(lambda x: sps.johnsonsu.rvs(size=sample_size, loc=x[f'loc_1'], scale=x[f'scale_1'], a=x[f'skewness_1'],
                                                                 b=x['tailweight_1']), axis=1)
        for num, df in enumerate(param_dfs[1:], start=2):
            sample = df.apply(lambda x: sps.johnsonsu.rvs(size=sample_size, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'],
                                                     b=x[f'tailweight_{num}']), axis=1)
            samples = pd.concat([samples, sample], join='inner', axis=1)

        samples.columns = [str(i) for i in range(len(samples.columns))]
        samples = samples.apply(np.concatenate, axis=1)
        pEns_quantiles = samples.apply(lambda x: np.percentile(x, quantile_array * 100))
        pEns_crps_observations = np.array([pinball_score(observed, quantiles_row) for observed, quantiles_row in
                                  zip(y, pEns_quantiles)])

        median_series = pEns_quantiles.apply(lambda x: x[49])
        mean_series = samples.apply(np.mean)

        #remove outliers
        pEns_crps_observations[pEns_crps_observations > outlier_threshold_crps] = outlier_threshold_crps
        pEns_mae_obs = np.abs(y.values - median_series)
        pEns_mae_obs[pEns_mae_obs > outlier_threshold_mae] = outlier_threshold_mae
        pEns_mae = np.mean(pEns_mae_obs)
        pEns_rmse_obs = ((y - mean_series) ** 2)
        pEns_rmse_obs[pEns_rmse_obs > outlier_threshold_rmse] = outlier_threshold_rmse
        pEns_rmse = np.sqrt(pEns_rmse_obs.mean())

        print(f'Results based on sample size of {sample_size} per distribution')
        print(f'p-Ens MAE: {pEns_mae}')
        print(f'p-Ens RMSE: {pEns_rmse}')
        print(f'p-Ens CRPS: {np.mean(pEns_crps_observations)} \n\n')
else:
    print('Could not calculate scores: Wrong distribution')

if not os.path.exists(f'/home/ahaas/BachelorThesis/forecasts_singleNN2_{distribution.lower()}_q-Ens'):
    os.mkdir(f'/home/ahaas/BachelorThesis/forecasts_singleNN2_{distribution.lower()}_q-Ens')

qEns_quantiles.to_csv(f'/home/ahaas/BachelorThesis/forecasts_singleNN2_{distribution.lower()}_q-Ens/predictions.csv')