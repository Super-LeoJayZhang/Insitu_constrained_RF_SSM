"""
--------------------------
File Name:  Draw_TimeSeries_evaluation.py
Author: zhang (FZJ/IBG3)
Contact: leojayak@gmail.com
Date: 05.09.21

Description: Evaluate RF SSM (compared with in-situ and ESA-CCI SSM time-series over the evaluation period).
--------------------------
"""
# Packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def cal_ubrmse(a, b):
    """ This function is used to calculate the unbiased Root Mean Square Error"""
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    part_sq = ((a - a_mean) - (b - b_mean)) ** 2
    ubrmse = np.sqrt(np.mean(part_sq))
    return ubrmse, a_mean, b_mean


# Work directory.
work_dir = r'/media/zhang/Samsung_T5/Thesis_Data/L2_RF/RF_regional_with_geographic_info_v2/all'
os.chdir(work_dir)

folder_in = r'Predicted_SM_time_series'
folder_out = r'TimeSeries_Evaluation_v2'
if not os.path.exists(folder_out):
    os.mkdir(folder_out)

file_lc = r'station_climate_lc_new.csv'
df_lc = pd.read_csv(file_lc, index_col=0)


def obtain_result(file_name, file_lc=file_lc):
    df_lc = pd.read_csv(file_lc, index_col=0)

    # Obtain the name of network and station.
    network_name = file_name.split('_')[0]  # Network name.
    station_name = file_name.split('_')[1].split('.csv')[0]  # Station name
    series_lc = df_lc[df_lc.index == file_name.strip('.csv')]
    lc_2010 = series_lc['LC_2010_STR'].values[0]  # land cover.
    climate = series_lc['climate_insitu'].values[0]  # climate zone.

    file = os.path.join(folder_in, file_name)
    df = pd.read_csv(file, index_col=0)

    rmse_RF = np.sqrt(mean_squared_error(df['in_situ'], df['estimated']))
    ubrmse_RF = cal_ubrmse(df['in_situ'], df['estimated'])[0]
    p_r_RF = pearsonr(df['in_situ'], df['estimated'])[0]
    rmse_cci = np.sqrt(mean_squared_error(df['in_situ'], df['esa_cci']))
    ubrmse_cci = cal_ubrmse(df['in_situ'], df['esa_cci'])[0]
    p_r_cci = pearsonr(df['in_situ'], df['esa_cci'])[0]
    number_n = len(df['in_situ'])

    df.columns = ['In_situ', 'RF_predicted', 'ESA_CCI']

    return df, network_name, station_name, [rmse_RF, ubrmse_RF, p_r_RF, rmse_cci, ubrmse_cci, p_r_cci, number_n], \
           climate, lc_2010


# test
# file_name = 'USCRN_Versailles-3-NNW.csv'
file_name_list = ['USCRN_Versailles-3-NNW.csv', 'SCAN_GoodwinCreekPasture.csv',
                  'SOILSCAPE_node403.csv', 'SNOTEL_DRYLAKE.csv']  # choose four stations.


def draw_TimeSeries(idx_i, idx_j, file_name, title, file_lc=file_lc, legend_on=0):
    ax = axs[idx_i, idx_j]
    df, network_name, station_name, metrics, climate, landcover = obtain_result(file_name)
    if legend_on:
        im = ax.plot(df.index, df['In_situ'], 'k-', label='In situ')
        im = ax.plot(df.index, df['RF_predicted'], 'g-.', label='RF SSM')
        im = ax.plot(df.index, df['ESA_CCI'], 'b:', label='ESA CCI')

    else:
        im = ax.plot(df.index, df['In_situ'], 'k-')
        im = ax.plot(df.index, df['RF_predicted'], 'g-.')
        im = ax.plot(df.index, df['ESA_CCI'], 'b:')
    # Set the figure parameters.
    ax.set_xticks(ax.get_xticks()[::int(metrics[-1] / 5)])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_ylabel(r'Surface soil moisture ($\mathregular{m^3m^{-3}}$)', fontsize=12)
    ax.set_title(f'({title})', fontsize=14)
    ax.set_xlabel(f'Date', fontsize=12)
    ax.text(x=0.05, y=0.47, s=str("Network:  %s.   Station:  %s." % (network_name, station_name)), fontsize=12)
    ax.text(x=0.05, y=0.44, s=str('Climate Zone: %s' % climate), fontsize=12)
    # ax.text(x=0.05, y=0.51, s=str('Land cover: %s' % landcover), fontsize=12)

    if legend_on:
        ax.legend(fontsize=12)


fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
draw_TimeSeries(0, 0, file_name_list[0], 'a')
draw_TimeSeries(0, 1, file_name_list[1], 'b')
draw_TimeSeries(1, 0, file_name_list[2], 'c')
draw_TimeSeries(1, 1, file_name_list[3], 'd', legend_on=1)
plt.savefig('Evaluation_TimeSeries.eps', dpi=300)
plt.savefig('Evaluation_TimeSeries.pdf', dpi=300)
plt.savefig('Evaluation_TimeSeries.png', dpi=300)
plt.savefig('Evaluation_TimeSeries.jpg', dpi=300)
