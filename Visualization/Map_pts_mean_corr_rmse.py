"""
Map_pts_mean_corr_rmse.py
date: 22-Sep-2021
author: leojay
Contact: l.zhang-8@student.utwente.nl
-------------------------------------
Description: 
"""
# libraries
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# -----------------------------------------
# Self defined function to draw the figure.
# -----------------------------------------
def draw_metrics_pts_regional(ax, df, vmin, vmax, metrics_name, title, colorbar_name, legend_on=1):
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.set_extent([-125, -66, 24, 46])

    # Add the Grid info
    ax.set_xticks([-120, -100, -80])
    ax.set_xticklabels([r'$\mathregular{120^o}$ W', '$\mathregular{100^o}$ W', '$\mathregular{80^o}$ W'])
    ax.set_yticks([50, 40, 30])
    ax.set_yticklabels([r'$\mathregular{50^o}$ N', '$\mathregular{40o}$ N', '$\mathregular{30^o}$ N'])
    ax.set_title(f'{title}', fontsize=12)
    plt.scatter(x=df['lon'], y=df['lat'], c=df[metrics_name], vmin=vmin, vmax=vmax,
                edgecolors='black', cmap='gnuplot2',
                data=df[metrics_name])
    if legend_on:
        cb = plt.colorbar(ax=ax, extend='both', aspect=50, shrink=0.7, fraction=0.046, pad=0.04)
        if 'RMSE' in colorbar_name:
            cb.set_label(colorbar_name + r' $(m^3m^{-3})$')
        else:
            cb.set_label(colorbar_name)


work_path = r'a:\Thesis_Data\L2_RF\sychronized\RF_GEE_Dataset_v02'
os.chdir(work_path)

''' Metrics '''
file_metrics = r'a:\Thesis_Data\L2_RF\RF_regional_with_geographic_info_v2\all\Evaluation_metrics.csv'
df_metrics = pd.read_csv(file_metrics, index_col=0)
file_lon_lat = r'a:\Thesis_Data\L2_RF\sychronized\station_lon_lat.csv'
df_lon_lat = pd.read_csv(file_lon_lat, index_col=0)
df_metrics_2 = pd.merge(df_metrics, df_lon_lat, on='station', how='inner')

data_crs = ccrs.PlateCarree()

plt.subplots(3, 2, figsize=(13.5, 11))
# RMSE: RF SSM
ax0 = plt.subplot(3, 2, 1, projection=data_crs)
draw_metrics_pts_regional(ax0, df_metrics_2, vmin=0, vmax=0.2, metrics_name='RMSE_RF',
                          title='(a) RF SSM', colorbar_name='RMSE')

ax1 = plt.subplot(3, 2, 2, projection=data_crs)
draw_metrics_pts_regional(ax1, df_metrics_2, vmin=0, vmax=0.2, metrics_name='RMSE_cci',
                          title='(b) ESA CCI', colorbar_name='RMSE')

ax2 = plt.subplot(3, 2, 3, projection=data_crs)
draw_metrics_pts_regional(ax2, df_metrics_2, vmin=0, vmax=0.1, metrics_name='ubRMSE_RF',
                          title='(c) RF SSM', colorbar_name='ubRMSE')

ax3 = plt.subplot(3, 2, 4, projection=data_crs)
draw_metrics_pts_regional(ax3, df_metrics_2, vmin=0, vmax=0.1, metrics_name='ubRMSE_RF',
                          title='(d) ESA CCI', colorbar_name='ubRMSE')

ax4 = plt.subplot(3, 2, 5, projection=data_crs)
draw_metrics_pts_regional(ax4, df_metrics_2, vmin=-0.2, vmax=1, metrics_name='r_RF',
                          title='(f) RF SSM', colorbar_name='Pearson Correlation')

ax5 = plt.subplot(3, 2, 6, projection=data_crs)
draw_metrics_pts_regional(ax5, df_metrics_2, vmin=-0.2, vmax=1, metrics_name='r_cci',
                          title='(f) ESA CCI', colorbar_name='Pearson Correlation')

#
# def draw_metrics_pts_regional(ax, df, vmin, vmax, metrics_name):
#     data_crs = ccrs.PlateCarree()
#     fig, ax = plt.subplots(1, 1)
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.coastlines(linewidth=0.5)
#     ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
#     ax.set_extent([-125, -66, 24, 46])
#
#     # Grid
#     ax.set_xticks([-120, -100, -80])
#     ax.set_xticklabels(['$\mathregular{120^o}$ W', '$\mathregular{100^o}$ W', '$\mathregular{80^o}$ W'])
#     ax.set_yticks([50, 40, 30])
#     ax.set_yticklabels(['$\mathregular{50^o}$ N', '$\mathregular{40o}$ N', '$\mathregular{30^o}$ N'])
#
#     plt.scatter(x=df['lon'], y=df['lat'], c=df[metrics_name], vmin=vmin, vmax=vmax,
#                 edgecolors='black', cmap='gnuplot2',
#                 data=df[metrics_name])
#     cb = plt.colorbar(ax=ax, aspect=50, shrink=0.6, fraction=0.046, pad=0.04)
#     if 'RMSE' in metrics_name:
#         cb.set_label(metrics_name + ' [$\mathregular{cm^3}$/$\mathregular{cm^3}$]')
#     else:
#         cb.set_label(metrics_name)
#

# output_folder
folder_output = r'a:\Thesis_Data'
plt.savefig(os.path.join(folder_output, 'map_pts_rmse_corr_rmse.pdf'), dpi=300)
