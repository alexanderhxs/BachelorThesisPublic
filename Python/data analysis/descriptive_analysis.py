import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('/home/ahaas/BachelorThesis/Datasets/DE.csv', index_col=0)
data.index = pd.to_datetime(data.index)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(data.describe())
print(data.skew())
print(data.kurtosis())


line_width = 0.8

vertical_line_date = data.index[-554*24]
vertical_line_date_2 = data.index[-736*24]
left_text_x = vertical_line_date_2 - pd.Timedelta(days=(1456)/2)
middle_text_x = vertical_line_date - pd.Timedelta(days=(736-554)/2)
right_text_x = vertical_line_date + pd.Timedelta(days=554/2)

fig, axs = plt.subplots(nrows=5, sharex=True, figsize=(10, 6), dpi=600)

# Plot-Befehle
axs[0].plot(data.index, data['Price'].rolling(24*7).mean(), linewidth=line_width)
axs[0].set_ylabel('Price\n(EUR/MWh)', fontsize=8)
text_height = axs[0].get_ylim()[1] * 1.05
axs[0].text(left_text_x, text_height, 'Training Period', ha='center', va='bottom', fontsize=8)
#axs[0].text(middle_text_x, text_height, '(QRA Calibration)', ha='center', va='bottom', fontsize=8)
axs[0].text(right_text_x, text_height, 'Out-of-sample testing', ha='center', va='bottom', fontsize=8)

axs[1].plot(data.index, data['Load_DA_Forecast'].rolling(24*7).mean()/1000, linewidth=line_width)
axs[1].set_ylabel('Load DA\nForecast\n(GWh)', fontsize=8)
axs[2].plot(data.index, data['Renewables_DA_Forecast'].rolling(24*7).mean()/1000, linewidth=line_width)
axs[2].set_ylabel('RES DA\nForecast\n(GWh)', fontsize=8)
axs[3].plot(data.index, data['EUA'], linewidth=line_width)
axs[3].set_ylabel('EUA Price\n(EUR/tCO2)', fontsize=8)
axs[4].plot(data.index, data['API2_Coal'], label='API2 Coal (EUR/t)', linewidth=line_width)
axs[4].plot(data.index, data['TTF_Gas'], label='TTF Gas (EUR/MWh)', linewidth=line_width)
axs[4].plot(data.index, data['Brent_oil'], label='Brent Oil (EUR/bbl.)', linewidth=line_width)
axs[4].set_ylabel('Fuel Prices', fontsize=8)
axs[4].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),ncol=3, fontsize='small')
for ax in axs:
    ax.axvline(x=vertical_line_date, color='grey', linestyle='--', linewidth=line_width)
#for ax in axs:
#    ax.axvline(x=vertical_line_date_2, color='grey', linestyle='--', linewidth=line_width)
for ax in axs:
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
plt.subplots_adjust(hspace=0.2)
plt.xlim(pd.Timestamp('2015-01-01'), pd.Timestamp('2021-01-01'))
plt.show()



