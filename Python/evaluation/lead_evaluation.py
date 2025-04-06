import numpy as np
import os
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
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
num_runs2 = 0
outlier_threshold = 1000
quantile_array = np.arange(0.01, 1, 0.01)

def pinball_score(observed, pred_quantiles):
    quantiles = np.arange(0.01, 1, 0.01)
    scores = pd.Series(np.maximum((1 - quantiles) * (pred_quantiles - observed), quantiles * (observed - pred_quantiles)))
    return scores.mean()

#collect data
param_dfs = []
param_dfs2 = []
#load data
for num in range(num_runs):
    if num_runs == 1:
        file_path = f'/home/ahaas/BachelorThesis/distparams_singleNN1_{distribution.lower()}_4'
    else:
        file_path = f'/home/ahaas/BachelorThesis/distparams_leadNN3.2_{distribution.lower()}_{num+1}'
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
for num in range(num_runs2):
    if num_runs2 == 1:
        file_path = f'/home/ahaas/BachelorThesis/distparams_leadNN3.1_{distribution.lower()}_4'
    else:
        file_path = f'/home/ahaas/BachelorThesis/distparams_probNN_{distribution.lower()}_{num + 1}'
    dist_file_list = sorted(os.listdir(file_path))
    print(file_path)
    dist_params = pd.DataFrame()
    for day, file in enumerate(dist_file_list):
        with open(os.path.join(file_path, file)) as f:
            fc_dict = json.load(f)

        fc_df = pd.DataFrame(fc_dict)
        fc_df.index = pd.date_range(pd.to_datetime(file), periods=len(fc_df), freq='H')
        dist_params = pd.concat([dist_params, fc_df])

    param_dfs2.append(dist_params.add_suffix(f'_{num+1}'))
data.index = pd.to_datetime(data.index)

#set test period right
for num, df in enumerate(param_dfs):
    param_dfs[num] = df.iloc[-24*554:, :]
for num, df in enumerate(param_dfs2):
    param_dfs2[num] = df.iloc[-24*554:, :]

#function for plot
def plotting(y, crps_observations, crps_observations2=None):

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    # CRPS
    ax1.plot(y[~outliers].index, pd.Series(crps_observations).rolling(window=24 * 7).mean(),
             label='CRPS leadNN-JSU', color='dimgrey', linewidth=2)
    if not (crps_observations2 is None):
        ax1.plot(y.index, pd.Series(crps_observations2).rolling(window=24 * 7).mean(),
                 label='CRPS probNN-JSU', color='darkgrey', linewidth=2)


    # Intraday Variance with scaling for visibility
    intraday_std = y.groupby(y.index.date).std()

    ax2 = ax1.twinx()
    ax2.plot(intraday_std.index, intraday_std.rolling(window=7).mean(),
             label='Smoothed Intraday Standard Deviation of DA Price', color='coral', linewidth=2, linestyle='--', zorder=2)

    # Aesthetics
    plt.grid(axis='y', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Smoothed CRPS value', fontsize=14)
    ax2.set_ylabel('Smoothed Intraday Variance', fontsize=14)
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    plt.xticks(rotation=45, fontsize=14)
    plt.xlabel('Test Period', fontsize=14)
    plt.subplots_adjust(bottom=0.2, top=0.9)

    # Set the x-axis limits
    start_date = pd.to_datetime('2020-01-01')
    end_date = pd.to_datetime('2020-06-30')
    #ax1.set_xlim(start_date, end_date)
    #ax2.set_xlim(start_date, end_date)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(top=28)

    # Legend for models
    lines, labels = ax1.get_legend_handles_labels()
    legend_models = ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02),
                               fontsize=14, ncol=len(lines), frameon=False)

    # Legend for Intraday Variance
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend_variance = ax2.legend(lines2, labels2, loc='lower center', bbox_to_anchor=(0.5, 1.08),
                                 fontsize=14, ncol=len(lines2), frameon=False)

    # Manually adjust legend position
    plt.setp(legend_models.get_title(), multialignment='center')
    plt.setp(legend_variance.get_title(), multialignment='center')

    # Ensure the legends are drawn above the plot area
    #plt.gca().add_artist(legend_models)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


#compute scores for plotting
if distribution.lower() == 'normal':
    for num, df in enumerate(param_dfs):
        num += 1
        quantiles = df.apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
        median_series = df.apply(lambda x: sps.norm.median(loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
        y = data.loc[df.index, 'Price']
        crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles)]

        df['crps'] = crps_observations
        df['crps'] = df['crps'].rolling(24*7).mean()
        outlier_mask = df['crps'] > outlier_threshold
        print(df.loc[outlier_mask])

        mae = np.abs(y.values - median_series).mean()
        rmse = np.sqrt((y.values - df[f'loc_{num}']) ** 2).mean()
        print(f'Overall results for run Nr {num}')
        print('Observations: ' + str(len(y)) + '\n')
        print('MAE: ' + str(mae) + '\n' + 'RMSE: ' + str(rmse))
        print('CRPS: ' + str(np.mean(crps_observations)) + '\n\n')

        #plotting(y, df[f'loc_{num}'], crps_observations, quantiles)
        for num2, df2 in enumerate(param_dfs2):
            quantiles2 = df2.apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
            crps_observations2 = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles2)]

            #plotting(y, crps_observations, crps_observations2=crps_observations2)





    if num_runs > 1:
        #q-Ens averaging (horizontal) via quantile averaging
        quantiles_runs = param_dfs[0].apply(
            lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_1'], scale=x[f'scale_1']), axis=1)
        quantiles_runs = pd.DataFrame({'run_1': quantiles_runs})
        loc_runs = pd.DataFrame(param_dfs[0]['loc_1'])

        for num, df in enumerate(param_dfs[1:], start=2):
            qs = df.apply(lambda x: sps.norm.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']), axis=1)
            qs = pd.DataFrame({f'run_{num}': qs})
            quantiles_runs = pd.merge(quantiles_runs, qs, left_index=True, right_index=True, how='inner')
            loc_runs = pd.merge(loc_runs, df[f'loc_{num}'], left_index=True, right_index=True, how='inner')

        qEns_quantiles = quantiles_runs.apply(np.mean, axis=1)
        qEns_crps_observations2 = [pinball_score(observed, quantiles_row) for observed, quantiles_row in
                                   zip(y, qEns_quantiles)]
        median_series = qEns_quantiles.apply(lambda x: x[49])
        loc_series = loc_runs.apply(np.mean, axis=1)

        qEns_mae2 = np.abs(y.values - median_series).mean()
        qEns_rmse2 = np.sqrt(((y - loc_series) ** 2).mean())

        print('q-Ens MAE: ' + str(qEns_mae2))
        print('q-Ens RSME:' + str(qEns_rmse2))
        print('q-Ens CRPS: ' + str(np.mean(qEns_crps_observations2)))

        plotting(y, qEns_crps_observations2)
        #plot_lead_bar(y, loc_series, qEns_quantiles)

        # p-Ens averaging (vertical) via sampling
        for sample_size in [50, 100, 250, 500, 1000, 2500]:
            samples = param_dfs[0].apply(lambda x: sps.norm.rvs(size=sample_size, loc=x['loc_1'], scale=x['scale_1']), axis=1)
            for num, df in enumerate(param_dfs[1:], start=2):
                sample = df.apply(lambda x: sps.norm.rvs(size=sample_size, loc=x[f'loc_{num}'], scale=x[f'scale_{num}']),
                                  axis=1)
                samples = pd.concat([samples, sample], join='inner', axis=1)

            samples.columns = [str(i) for i in range(len(samples.columns))]
            samples = samples.apply(np.concatenate, axis=1)
            pEns_quantiles = samples.apply(lambda x: np.percentile(x, quantile_array * 100))
            pEns_crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in
                                      zip(y, pEns_quantiles)]
            median_series = pEns_quantiles.apply(lambda x: x[49])
            mean_series = samples.apply(np.mean)

            pEns_mae = np.abs(y.values - median_series).mean()
            pEns_rmse = np.sqrt(((y.values - mean_series) ** 2).mean())

            print(f'Results based on sample size of {sample_size} per distribution')
            print(f'p-Ens MAE: {pEns_mae}')
            print(f'p-Ens RMSE: {pEns_rmse}')
            print(f'p-Ens CRPS: {np.mean(pEns_crps_observations)} \n\n')

            if sample_size == 500:
                plotting(y, mean_series, pEns_crps_observations, pEns_quantiles)



if distribution.lower() == 'jsu':

    outliers = []
    for num, df in enumerate(param_dfs):
        num += 1
        quantiles = df.apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
        median_series = df.apply(lambda x: sps.johnsonsu.median(loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'],
                                           b=x[f'tailweight_{num}']), axis=1)
        y = data.loc[df.index, 'Price']
        crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in zip(y, quantiles)]

        mae = np.abs(y.values - median_series).mean()
        rmse = np.sqrt((y.values - df[f'loc_{num}']) ** 2).mean()

        #detect outliers
        df['crps'] = crps_observations
        outlier_mask = df['crps'] > outlier_threshold
        df = df.loc[~outlier_mask]
        outliers.append(outlier_mask)

        print(f'Overall results for run Nr {num}')
        print('Observations: ' + str(len(y)) + '\n')
        print('MAE: ' + str(mae) + '\n' + 'RMSE: ' + str(rmse))
        print('CRPS: ' + str(np.mean(crps_observations)) + '\n\n')

        outliers = outlier_mask
        plotting(y, crps_observations)


    if num_runs > 1:
        # removing all outliers from runs
        outliers = np.any(outliers, axis=0)
        for i, df in enumerate(param_dfs):
            param_dfs[i] = df.loc[~outliers]
        y = y.loc[~outliers]

        #q-Ens averaging (horizontal) via quantile averaging
        quantiles_runs = param_dfs[0].apply(
            lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_1'], scale=x[f'scale_1'], a=x[f'skewness_1'], b=x[f'tailweight_1']), axis=1)
        quantiles_runs = pd.DataFrame({'run_1': quantiles_runs})
        mean_series = pd.DataFrame(param_dfs[0].apply(
            lambda x: sps.johnsonsu.mean(loc=x[f'loc_1'], scale=x[f'scale_1'], a=x[f'skewness_1'], b=x['tailweight_1']),
            axis=1))
        mean_series.columns = ['run_1']

        for num, df in enumerate(param_dfs[1:], start=2):
            qs = df.apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
            qs = pd.DataFrame({f'run_{num}': qs})
            quantiles_runs = pd.merge(quantiles_runs, qs, left_index=True, right_index=True, how='inner')
            means_trial = df.apply(
                lambda x: sps.johnsonsu.mean(loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'],
                                             b=x[f'tailweight_{num}']), axis=1)
            means_trial_df = pd.DataFrame(means_trial, columns=[f'run_{num}'])
            mean_series = pd.merge(mean_series, means_trial_df, left_index=True, right_index=True, how='inner')

        qEns_quantiles = quantiles_runs.apply(np.mean, axis=1)
        qEns_crps_observations = [pinball_score(observed, quantiles_row) for observed, quantiles_row in
                                   zip(y, qEns_quantiles)]
        median_series = qEns_quantiles.apply(lambda x: x[49])
        mean_series = np.mean(mean_series, axis=1)

        qEns_mae2 = np.abs(y.values - median_series).mean()
        qEns_rmse2 = np.sqrt(((y - mean_series) ** 2).mean())

        print('q-Ens MAE: ' + str(qEns_mae2))
        print('q-Ens RSME:' + str(qEns_rmse2))
        print('q-Ens CRPS: ' + str(np.mean(qEns_crps_observations)))


        # q-Ens averaging (horizontal) via quantile averaging for 2. model
        quantiles_runs = param_dfs2[0].apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_1'], scale=x[f'scale_1'], a=x[f'skewness_1'],
                                        b=x[f'tailweight_1']), axis=1)
        quantiles_runs = pd.DataFrame({'run_1': quantiles_runs})
        mean_series = pd.DataFrame(param_dfs[0].apply(
            lambda x: sps.johnsonsu.mean(loc=x[f'loc_1'], scale=x[f'scale_1'], a=x[f'skewness_1'], b=x['tailweight_1']),
            axis=1))
        mean_series.columns = ['run_1']

        for num, df in enumerate(param_dfs2[1:], start=2):
            qs = df.apply(lambda x: sps.johnsonsu.ppf(quantile_array, loc=x[f'loc_{num}'], scale=x[f'scale_{num}'],
                                                      a=x[f'skewness_{num}'], b=x[f'tailweight_{num}']), axis=1)
            qs = pd.DataFrame({f'run_{num}': qs})
            quantiles_runs = pd.merge(quantiles_runs, qs, left_index=True, right_index=True, how='inner')
            means_trial = df.apply(
                lambda x: sps.johnsonsu.mean(loc=x[f'loc_{num}'], scale=x[f'scale_{num}'], a=x[f'skewness_{num}'],
                                             b=x[f'tailweight_{num}']), axis=1)
            means_trial_df = pd.DataFrame(means_trial, columns=[f'run_{num}'])
            mean_series = pd.merge(mean_series, means_trial_df, left_index=True, right_index=True, how='inner')

        y = data.loc[df.index, 'Price']
        qEns_quantiles = quantiles_runs.apply(np.mean, axis=1)
        qEns_crps_observations2 = [pinball_score(observed, quantiles_row) for observed, quantiles_row in
                                  zip(y, qEns_quantiles)]
        median_series = qEns_quantiles.apply(lambda x: x[49])
        mean_series = np.mean(mean_series, axis=1)

        qEns_mae2 = np.abs(y.values - median_series).mean()
        qEns_rmse2 = np.sqrt(((y - mean_series) ** 2).mean())

        print('q-Ens MAE: ' + str(qEns_mae2))
        print('q-Ens RSME:' + str(qEns_rmse2))
        print('q-Ens CRPS: ' + str(np.mean(qEns_crps_observations2)))

        plotting(y, qEns_crps_observations, qEns_crps_observations2)