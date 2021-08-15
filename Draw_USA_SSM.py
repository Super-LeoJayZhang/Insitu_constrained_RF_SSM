"""
Thesis_MSc_Lijie Draw_USA_SSM
date: 05-Jul-2021
author: leojay
Contact: leojayak@gmail.com
-------------------------------------
Description: 
"""
# libraries
import os
from datetime import datetime
import netCDF4 as nc
import numpy as np
import gdal
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# part1 = 1  # Extract the data [time, lat, lon] of the US continental.
# part2 = 1  # Draw the US figures.
# part3 = 1  # Draw the temporal-lon, temporal-lat.

# Part 1
file_rf = f'a:\Thesis_Data\L3_RF_gridded\RF_predicted_global_nc\RF_SSM_global.nc'
nc_rf = nc.Dataset(file_rf, 'r')
ssm_rf = nc_rf['ssm']
time = nc_rf['time']
date_rf_e = list(time[:]).index(20180630)

x_rf_0 = int((180 - 125) / 0.25)
x_rf_1 = int((180 - 66) / 0.25)
y_rf_0 = int((70 - 50) / 0.25)
y_rf_1 = int((70 - 24) / 0.25)
ssm_rf_us_18yrs = ssm_rf[0:date_rf_e + 1, y_rf_0: y_rf_1, x_rf_0:x_rf_1]

folder_cci = 'a:\Thesis_Data\L3_RF_gridded\ESA_CCI_processing\Merged_ESA_CCI_years'
x_cci_0 = int((180 - 125) / 0.25)
x_cci_1 = int((180 - 66) / 0.25)
y_cci_0 = int((90 - 50) / 0.25)
y_cci_1 = int((90 - 24) / 0.25)

for year in np.arange(2000, 2019):
    print(f'Processing: {year}')
    file_cci = os.path.join(folder_cci, 'ESA_CCI_MERGED_' + str(year) + '.tif')
    gdal_file = gdal.Open(file_cci)
    if year == 2000:
        ssm_cci_us = gdal_file.ReadAsArray()[:, y_cci_0:y_cci_1, x_cci_0:x_cci_1]
    else:
        ssm_cci_us = np.concatenate((ssm_cci_us, gdal_file.ReadAsArray()[:, y_cci_0:y_cci_1, x_cci_0:x_cci_1]), axis=0)


def draw_US_ssm(data):
    data_crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.set_extent([-125, -66, 24, 50])
    ax.add_feature(cfeature.LAKES,edgecolor='k', facecolor='w')
    # Grid
    ax.set_xticks([-120, -100, -80])
    ax.set_xticklabels(['$\mathregular{120^o}$ W', '$\mathregular{100^o}$ W', '$\mathregular{80^o}$ W'])
    ax.set_yticks([50, 40, 30])
    ax.set_yticklabels(['$\mathregular{50^o}$ N', '$\mathregular{40o}$ N', '$\mathregular{30^o}$ N'])

    # norm = Normalize(vmin=0, vmax=0.35)
    plt.imshow(data, cmap='YlGnBu', origin='upper', vmin=0, vmax=0.4, extent=[-125, -66, 24, 50])
    # Add the ColorBar
    cb = plt.colorbar(ax=ax, orientation="horizontal", aspect=50, shrink=0.6, extend='both')
    cb.set_label('Surface Soil Moisture [$\mathregular{cm^3}$/$\mathregular{cm^3}$]')
def draw_US_ssm_diff(data):
    data_crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.set_extent([-125, -66, 24, 50])
    ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='w')
    # Grid
    ax.set_xticks([-120, -100, -80])
    ax.set_xticklabels(['$\mathregular{120^o}$ W', '$\mathregular{100^o}$ W', '$\mathregular{80^o}$ W'])
    ax.set_yticks([50, 40, 30])
    ax.set_yticklabels(['$\mathregular{50^o}$ N', '$\mathregular{40o}$ N', '$\mathregular{30^o}$ N'])

    # norm = Normalize(vmin=0, vmax=0.35)
    plt.imshow(data, cmap='RdBu', origin='upper', extent=[-125, -66, 24, 50])
    # Add the ColorBar
    cb = plt.colorbar(ax=ax, orientation="horizontal", aspect=50, shrink=0.6, extend='both')
    cb.set_label('Surface Soil Moisture [$\mathregular{cm^3}$/$\mathregular{cm^3}$]')


draw_US_ssm(np.nanmean(ssm_rf_us_18yrs, 0))
plt.savefig('US_ssm_rf.pdf', dpi=600)
draw_US_ssm(np.nanmean(ssm_cci_us, 0))
plt.savefig('US_ssm_cci.pdf', dpi=600)
draw_US_ssm_diff(np.nanmean(ssm_rf_us_18yrs, 0) -np.nanmean(ssm_cci_us, 0 ))
plt.savefig('US_ssm_diff.pdf', dpi=600)

idx0 = 0
idx1 = list(time[:]).index(20050101)
idx2 = list(time[:]).index(20100101)
idx3 = list(time[:]).index(20150101)

idx1_cci = 365*5+1 - 31-24
idx2_cci = idx1_cci+365*5+1
idx3_cci = idx2_cci+365*5+1

def draw_lat_temp(lat_temp_data, a1, a2, a3):
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), constrained_layout=True)
    # norm = Normalize(vmin=0, vmax=vmax)
    axs[0].imshow(lat_temp_data[:, 0: a1 - 1], cmap='YlGnBu', vmin=0, vmax=0.4)
    axs[1].imshow(lat_temp_data[:, a1: a2 - 1], cmap='YlGnBu', vmin=0, vmax=0.4)
    axs[2].imshow(lat_temp_data[:, a2: a3 - 1], cmap='YlGnBu', vmin=0, vmax=0.4)
    sc = axs[3].imshow(lat_temp_data[:, a3: -1], cmap='YlGnBu', vmin=0, vmax=0.4)
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.3)
    cb.set_label('Surface Soil Moisture [$\mathregular{cm^3}$/$\mathregular{cm^3}$]')
def draw_lon_temp(lon_temp_data, a1, a2, a3):
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 8), constrained_layout=True)
    axs[0].imshow(lon_temp_data[0: a1 - 1, :], cmap='YlGnBu', vmin=0, vmax=0.4)
    axs[1].imshow(lon_temp_data[a1: a2 - 1, :], cmap='YlGnBu', vmin=0, vmax=0.4)
    axs[2].imshow(lon_temp_data[a2: a3 - 1, :], cmap='YlGnBu', vmin=0, vmax=0.4)
    sc = axs[3].imshow(lon_temp_data[a3: -1, :], cmap='YlGnBu', vmin=0, vmax=0.4)
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.3)
    cb.set_label('Surface Soil Moisture [$\mathregular{cm^3}$/$\mathregular{cm^3}$]')
draw_lat_temp(np.nanmean(ssm_rf_us_18yrs, 2).transpose(), idx1, idx2, idx3)
draw_lat_temp(np.nanmean(ssm_cci_us[55:-1, ::], 2).transpose(), idx1_cci, idx2_cci, idx3_cci)

draw_lon_temp(np.nanmean(ssm_rf_us_18yrs, 1), idx1, idx2, idx3)
draw_lon_temp(np.nanmean(ssm_cci_us[55:-1, ::], 1), idx1_cci, idx2_cci, idx3_cci)


ssm_mean_diff = np.nanmean(ssm_rf_us_18yrs , 0) - np.nanmean(ssm_cci_us, 0)
draw_US_ssm_diff(ssm_mean_diff)


