import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns

#load data
data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
plt_data = data.iloc[:, 1].groupby(data.index.hour).agg(list)
plt_data = plt_data.apply(lambda x: np.array(x)/1000)
q_05_data = plt_data.apply(lambda x: np.percentile(x, 5))
q_95_data = plt_data.apply(lambda x: np.percentile(x, 95))
q_25_data = plt_data.apply(lambda x: np.percentile(x, 25))
q_75_data = plt_data.apply(lambda x: np.percentile(x, 75))
q_50_data = plt_data.apply(lambda x: np.percentile(x, 50))
q_99_data = plt_data.apply(lambda x: np.percentile(x, 99.9))
q_01_data = plt_data.apply(lambda x: np.percentile(x, 0.1))
var_data = plt_data.apply(lambda x: np.std(x))
plt.figure(figsize=(10, 6), dpi=600)
plt.violinplot(plt_data,
                   showmeans=False,
                   showextrema=False)
plt.plot(range(1, 25), plt_data.apply(np.mean), linewidth=1, label='Mean')
plt.plot(range(1, 25), q_05_data, linewidth=.8, color='grey', linestyle='--', label='90% percentile band')
plt.plot(range(1, 25), q_95_data, linewidth=.8, color='grey', linestyle='--')

plt.title('Violonplot of Load DA Forecast by Hour', fontsize=18)
plt.legend(loc='upper left', fontsize=14)
#plt.xticks(range(1, 8), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=14)
plt.xticks(range(1, 25, 2), fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Hour', fontsize=14)
plt.ylabel('Load DA Forecast\n(EUR/GWh)', fontsize=14)
#plt.ylim(top=125, bottom=-25)
plt.show()

print(str(q_75_data.mean() - q_50_data.mean()))
print(str(q_99_data.mean() - q_50_data.mean()))
print(str(q_50_data.mean() - q_25_data.mean()))
print(str(q_50_data.mean() - q_01_data.mean()))
print(str(var_data))

#load params
filepath = '/home/ahaas/BachelorThesis/distparams_leadNN_normal_4'
#load data
def load_data(filepath):
    dist_params = pd.DataFrame()
    for file in sorted(os.listdir(filepath)):
        with open(os.path.join(filepath, file)) as f:
            fc_dict = json.load(f)

        fc_df = pd.DataFrame(fc_dict)
        fc_df.index = pd.date_range(pd.to_datetime(file), periods=len(fc_df), freq='H')
        dist_params = pd.concat([dist_params, fc_df])
    return dist_params

#function to plot boxplots for every hour
def violon_per_hour(dist_params, parameter):
    data = dist_params[parameter].groupby(dist_params.index.hour).agg(list)
    plt.violinplot(data,
                   showmeans=True,
                   showextrema=True,
                   quantiles=[[.05, .25, .75, .95] for _ in range(len(data))])
    plt.plot(range(1, 25), dist_params[parameter].groupby(dist_params.index.hour).mean(), linewidth= 1)
    plt.show()
    #if not os.path.exists('/home/ahaas/BachelorThesis/Plots'):
    #    os.mkdir('/home/ahaas/BachelorThesis/Plots')
    #plt.savefig('/home/ahaas/BachelorThesis/Plots/prob_normal_1.svg')


#violon_per_hour(load_data(filepath), 'loc')
