"""
 Map_pts_mean_corr_rmse
date: 10-Jul-2020
 author: L.Zhang
Contact: leojayak@gmail.com
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

# time (start running)
time_0 = datetime.now()
start_time = time_0.strftime("%Y-%m-%d %H:%M:%S")
print('Program start running at: %s. ' % start_time)

# Set the working path
work_path = r'a:\Thesis_Data\L2_RF\sychronized\RF_GEE_Dataset_v02'
os.chdir(work_path)
files = os.listdir()

df_insitu_mean_ssm = pd.DataFrame(columns=['mean_ssm', 'lon', 'lat'])


def cal_Mean_insitu_SSM(file):
    df_station = pd.read_csv(file, index_col=0)

    if len(df_station) > 1:
        df_station.index = pd.to_datetime(df_station.index)
        date_duration = (df_station.index[-1] - df_station.index[0]).days
        if date_duration > 365:
            ssm_mean = df_station.mean()['soil moisture']
            lon = df_station.iloc[0]['lon']
            lat = df_station.iloc[0]['lat']

            series_insitu_mean_ssm = pd.Series(index=['mean_ssm', 'lon', 'lat'], dtype='float')
            series_insitu_mean_ssm['mean_ssm'] = df_station.mean()['soil moisture']
            series_insitu_mean_ssm['lon'] = df_station.iloc[0]['lon']
            series_insitu_mean_ssm['lat'] = df_station.iloc[0]['lat']
            return series_insitu_mean_ssm
    else:
        pass


def draw_Mean_insitu_SSM_regional(df):
    data_crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.set_extent([-125, -66, 24, 46])

    # Grid
    ax.set_xticks([-120, -100, -80])
    ax.set_xticklabels(['$\mathregular{120^o}$ W', '$\mathregular{100^o}$ W', '$\mathregular{80^o}$ W'])
    ax.set_yticks([50, 40, 30])
    ax.set_yticklabels(['$\mathregular{50^o}$ N', '$\mathregular{40o}$ N', '$\mathregular{30^o}$ N'])

    plt.scatter(x=df['lon'], y=df['lat'], c=df['mean_ssm'], vmin=0, vmax=0.35, edgecolors='black', cmap='YlGnBu',
                data=df['mean_ssm'])
    cb = plt.colorbar(ax=ax, aspect=50, shrink=0.6, fraction=0.046, pad=0.04)
    cb.set_label('Surface Soil Moisture [$\mathregular{cm^3}$/$\mathregular{cm^3}$]')


def draw_Mean_insitu_SSM_global(df):
    data_crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.set_extent([-180, 180, -60, 75])

    # Grid
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(['$\mathregular{180^o}$ W', '$\mathregular{90^o}$ W', '$\mathregular{0^o}$',
                        '$\mathregular{90^o}$ E', '$\mathregular{180^o}$ E'])
    ax.set_yticks([60, 30, 0, -30, -60])
    ax.set_yticklabels(
        ['$\mathregular{60^o}$ N', '$\mathregular{30^o}$ N', '$\mathregular{0^o}$ N',
         '$\mathregular{30^o}$ S', '$\mathregular{60^o}$ S'])
    plt.scatter(x=df['lon'], y=df['lat'], c=df['mean_ssm'], vmin=0, vmax=0.35, edgecolors='black', cmap='YlGnBu',
                data=df['mean_ssm'])
    cb = plt.colorbar(ax=ax, aspect=50, shrink=0.6, fraction=0.046, pad=0.04)
    cb.set_label('Surface Soil Moisture [$\mathregular{cm^3}$/$\mathregular{cm^3}$]')


for idx, file in enumerate(files):
    # Mark the procedure status.
    print('In the processing: %d of %d' % (idx + 1, len(files)))
    df_insitu_mean_ssm.loc[idx] = cal_Mean_insitu_SSM(file)

# Draw
a = df_insitu_mean_ssm.copy()
a[a['mean_ssm'] < 0] = np.nan
b = a.dropna()

''' Regional '''
draw_Mean_insitu_SSM_regional(b)

''' Global '''
draw_Mean_insitu_SSM_global(b)

''' Metrics '''
file_metrics = r'a:\Thesis_Data\L2_RF\RF_regional_with_geographic_info_v2\all\Evaluation_metrics.csv'
df_metrics = pd.read_csv(file_metrics, index_col=0)
file_lon_lat = r'a:\Thesis_Data\L2_RF\sychronized\station_lon_lat.csv'
df_lon_lat = pd.read_csv(file_lon_lat, index_col=0)

df_metrics_2 = pd.merge(df_metrics, df_lon_lat, on='station', how='inner')


def draw_metrics_pts_regional(df, vmin, vmax, metrics_name):
    data_crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.set_extent([-125, -66, 24, 46])

    # Grid
    ax.set_xticks([-120, -100, -80])
    ax.set_xticklabels(['$\mathregular{120^o}$ W', '$\mathregular{100^o}$ W', '$\mathregular{80^o}$ W'])
    ax.set_yticks([50, 40, 30])
    ax.set_yticklabels(['$\mathregular{50^o}$ N', '$\mathregular{40o}$ N', '$\mathregular{30^o}$ N'])

    plt.scatter(x=df['lon'], y=df['lat'], c=df[metrics_name], vmin=vmin, vmax=vmax,
                edgecolors='black', cmap='gnuplot2',
                data=df[metrics_name])
    cb = plt.colorbar(ax=ax, aspect=50, shrink=0.6, fraction=0.046, pad=0.04)
    if 'RMSE' in metrics_name:
        cb.set_label(metrics_name + ' [$\mathregular{cm^3}$/$\mathregular{cm^3}$]')
    else:
        cb.set_label(metrics_name)
# output_folder
folder_output = r'a:\Thesis_Data\L3_RF_gridded\US_GRIDDED_0707\L3_metrics_maps'

draw_metrics_pts_regional(df_metrics_2, vmin=0, vmax=0.2, metrics_name='RMSE_RF')
plt.savefig(os.path.join(folder_output, 'RMSE_RF_pts_region.jpg'), dpi=300)
plt.close()

draw_metrics_pts_regional(df_metrics_2, vmin=0, vmax=0.2, metrics_name='RMSE_cci')
plt.savefig(os.path.join(folder_output, 'RMSE_cci_pts_region.jpg'), dpi=300)
plt.close()

draw_metrics_pts_regional(df_metrics_2, vmin=0, vmax=0.1, metrics_name='ubRMSE_RF')
plt.savefig(os.path.join(folder_output, 'ubRMSE_RF_pts_region.jpg'), dpi=300)
plt.close()

draw_metrics_pts_regional(df_metrics_2, vmin=0, vmax=0.1, metrics_name='ubRMSE_cci')
plt.savefig(os.path.join(folder_output, 'ubRMSE_cci_pts_region.jpg'), dpi=300)
plt.close()

draw_metrics_pts_regional(df_metrics_2, vmin=-0.2, vmax=1, metrics_name='r_RF')
plt.savefig(os.path.join(folder_output, 'r_RF_pts_region.jpg'), dpi=300)
plt.close()

draw_metrics_pts_regional(df_metrics_2, vmin=-0.2, vmax=1, metrics_name='r_cci')
plt.savefig(os.path.join(folder_output, 'r_CCI_pts_region.jpg'), dpi=300)
plt.close()


def draw_metrics_pts_global(df, vmin, vmax, metrics_name):
    data_crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.set_extent([-180, 180, -60, 75])

    # Grid
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(['$\mathregular{180^o}$ W', '$\mathregular{90^o}$ W', '$\mathregular{0^o}$',
                        '$\mathregular{90^o}$ E', '$\mathregular{180^o}$ E'])
    ax.set_yticks([60, 30, 0, -30, -60])
    ax.set_yticklabels(
        ['$\mathregular{60^o}$ N', '$\mathregular{30^o}$ N', '$\mathregular{0^o}$ N',
         '$\mathregular{30^o}$ S', '$\mathregular{60^o}$ S'])
    plt.scatter(x=df['lon'], y=df['lat'], c=df[metrics_name], vmin=vmin, vmax=vmax,
                edgecolors='black', cmap='gnuplot2',
                data=df[metrics_name])
    cb = plt.colorbar(ax=ax, aspect=50, shrink=0.6, fraction=0.046, pad=0.04)
    if 'RMSE' in metrics_name:
        cb.set_label(metrics_name + ' [$\mathregular{cm^3}$/$\mathregular{cm^3}$]')
    else:
        cb.set_label(metrics_name)


draw_metrics_pts_global(df_metrics_2, vmin=0, vmax=0.3, metrics_name='RMSE_RF')
plt.savefig(os.path.join(folder_output, 'RMSE_RF_pts_global.jpg'), dpi=300)
plt.close()

draw_metrics_pts_global(df_metrics_2, vmin=0, vmax=0.3, metrics_name='RMSE_cci')
plt.savefig(os.path.join(folder_output, 'RMSE_cci_pts_global.jpg'), dpi=300)
plt.close()


draw_metrics_pts_global(df_metrics_2, vmin=0, vmax=0.1, metrics_name='ubRMSE_RF')
plt.savefig(os.path.join(folder_output, 'ubRMSE_RF_pts_global.jpg'), dpi=300)
plt.close()

draw_metrics_pts_global(df_metrics_2, vmin=0, vmax=0.1, metrics_name='ubRMSE_cci')
plt.savefig(os.path.join(folder_output, 'ubRMSE_cci_pts_global.jpg'), dpi=300)
plt.close()

draw_metrics_pts_global(df_metrics_2, vmin=-0.2, vmax=1, metrics_name='r_RF')
plt.savefig(os.path.join(folder_output, 'r_RF_pts_global.jpg'), dpi=300)
plt.close()

draw_metrics_pts_global(df_metrics_2, vmin=-0.2, vmax=1, metrics_name='r_cci')
plt.savefig(os.path.join(folder_output, 'r_CCI_pts_global.jpg'), dpi=300)
plt.close()





# time (end running)
time_1 = datetime.now()
end_time = time_1.strftime("%Y-%m-%d %H:%M:%S")
print('program end running at: %s.' % end_time)
time_consuming = (time_1 - time_0).seconds
print('It takes %d seconds to run this scripts' % int(time_consuming))
