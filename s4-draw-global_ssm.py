"""
 s4-draw-global_ssm
date: 26-Jun-2021
 author: L.Zhang
Contact: leojayak@gmail.com
-------------------------------------
Description: 
"""
# libraries
import os
from sklearn.externals import joblib
import gdal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
import netCDF4 as nc
import pandas as pd

# Set the working path
work_path = r'a:\Thesis_Data\L3_RF_gridded\RF_predicted_global_nc'
os.chdir(work_path)


def draw_global_ssm_map(data):
    # /update: 03-07-2021.
    data_crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.set_extent([-180, 180, -58, 70])

    # Grid
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(['$\mathregular{180^o}$ W', '$\mathregular{90^o}$ W', '$\mathregular{0^o}$',
                        '$\mathregular{90^o}$ E', '$\mathregular{180^o}$ E'])
    ax.set_yticks([70, 60, 30, 0, -30, -58])
    ax.set_yticklabels(
        ['$\mathregular{70^o}$ N', '$\mathregular{60^o}$ N', '$\mathregular{30^o}$ N', '$\mathregular{0^o}$ N',
         '$\mathregular{30^o}$ S', '$\mathregular{58^o}$ S'])
    ax.add_feature(cfeature.LAKES, facecolor='w')
    # norm = Normalize(vmin=0, vmax=0.4)
    # Add the Data
    plt.imshow(data, cmap='YlGnBu', origin='upper', vmin=0, vmax=0.4, extent=[-180, 180, -58, 70])
    # Add the ColorBar
    cb = plt.colorbar(ax=ax, orientation="horizontal", aspect=50, shrink=0.6,  extend='both')
    cb.set_label('Surface Soil Moisture [$\mathregular{cm^3}$/$\mathregular{cm^3}$]')
def draw_global_ssm_diff(data):
    # /update: 03-07-2021.
    data_crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.set_extent([-180, 180, -58, 70])

    # Grid
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(['$\mathregular{180^o}$ W', '$\mathregular{90^o}$ W', '$\mathregular{0^o}$',
                        '$\mathregular{90^o}$ E', '$\mathregular{180^o}$ E'])
    ax.set_yticks([70, 60, 30, 0, -30, -58])
    ax.set_yticklabels(
        ['$\mathregular{70^o}$ N', '$\mathregular{60^o}$ N', '$\mathregular{30^o}$ N', '$\mathregular{0^o}$ N',
         '$\mathregular{30^o}$ S', '$\mathregular{58^o}$ S'])
    ax.add_feature(cfeature.LAKES, facecolor='w')
    # norm = Normalize(vmin=-0.2, vmax=0.2)
    # Add the Data
    plt.imshow(data, cmap='RdBu',   origin='upper', extent=[-180, 180, -58, 70])
    # Add the ColorBar
    cb = plt.colorbar(ax=ax, orientation="horizontal", aspect=50, shrink=0.6, extend='both')
    cb.set_label('Surface Soil Moisture difference [$\mathregular{cm^3}$/$\mathregular{cm^3}$]')


file_esa = 'a:\Thesis_Data\L3_RF_gridded\ESA_CCI_processing\Merged_ESA_CCI_years\ESA_CCI_MERGED_2015.tif'

file_rf = 'a:\Thesis_Data\L3_RF_gridded\RF_predicted_global_nc\RF_global_2015.nc'
# Read ESA-CCI mean data.
gdal_esa = gdal.Open(file_esa)
esa_cci = gdal_esa.ReadAsArray()
# Draw ESA-CCI figure
ssm_mean_2015_cci = np.nanmean(esa_cci[:, 80: 720 - 128, :], 0)
draw_global_ssm_map(ssm_mean_2015_cci)
plt.savefig('Global_ESA-CCI_SSM_mean_2015.pdf', dpi=600)
plt.close()

nc_file = nc.Dataset(file_rf, 'r')
ssm = nc_file['ssm']
ssm_mean_2015_rf = np.nanmean(ssm, 0)
draw_global_ssm_map(ssm_mean_2015_rf)
plt.savefig('Global_RF_SSM_mean_2015.pdf', dpi=600)
plt.close()

draw_global_ssm_diff( ssm_mean_2015_rf - ssm_mean_2015_cci)
plt.savefig('Global_SSM_difference_RF-CCI_2015.pdf', dpi=600)
plt.close()



