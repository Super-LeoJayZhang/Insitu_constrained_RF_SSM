"""
Thesis_MSc_Lijie Draw_global_SSM.py
date: 22-Sep-2021
author: leojay
Contact: l.zhang-8@student.utwente.nl
-------------------------------------
Description: 
"""

# libraries
import os
import gdal
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc


# Set the working path
work_path = r'a:\Thesis_Data\L3_RF_gridded\RF_predicted_global_nc'
os.chdir(work_path)


def draw_global_ssm_map(ax, data, title='title', vmin=0, vmax=0.4, colorbar_name='SSM', cmap='YlGnBu'):
    """
    :param ax: ax
    :param data: SSM data
    :param title: The title of subplot.
    :param vmin: threshold of value (min)
    :param vmax: Threshold of value (max)
    :param colorbar_name: color bar name.
    :param cmap: cmap
    :return: a figure.
    """
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
    plt.imshow(data, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax, extent=[-180, 180, -58, 70])
    # Add the ColorBar
    cb = plt.colorbar(ax=ax, aspect=50, shrink=0.8, extend='both')
    cb.set_label(fr'{colorbar_name}' + ' $(m^3m^{-3})$', fontsize=12)
    plt.title(f'{title}', fontsize=12)


file_esa = 'a:\Thesis_Data\L3_RF_gridded\ESA_CCI_processing\Merged_ESA_CCI_years\ESA_CCI_MERGED_2015.tif'

file_rf = 'a:\Thesis_Data\L3_RF_gridded\RF_predicted_global_nc\RF_global_2015.nc'
# Read ESA-CCI mean data.
gdal_esa = gdal.Open(file_esa)
esa_cci = gdal_esa.ReadAsArray()

# Get the SSM mean value of 2015.
ssm_mean_2015_cci = np.nanmean(esa_cci[:, 80: 720 - 128, :], 0)

nc_file = nc.Dataset(file_rf, 'r')
ssm = nc_file['ssm']
ssm_mean_2015_rf = np.nanmean(ssm, 0)

data_crs = ccrs.PlateCarree()
plt.subplots(3, 1, figsize=(8, 12))

ax0 = plt.subplot(3, 1, 1, projection=data_crs)
draw_global_ssm_map(ax0, ssm_mean_2015_rf, title='(a) RF SSM')

ax1 = plt.subplot(3, 1, 2, projection=data_crs)
draw_global_ssm_map(ax1, ssm_mean_2015_cci, title='(b) ESA CCI ssm')

ax2 = plt.subplot(3, 1, 3, projection=data_crs)
draw_global_ssm_map(ax2, ssm_mean_2015_rf - ssm_mean_2015_cci, vmin=-0.2, vmax=0.1, cmap='RdBu',
                    title='(c) Difference between RF and ESA CCI SSM')

plt.savefig(r'a:\Thesis_Data\Global_SSM.pdf', dpi=300)
plt.savefig(r'a:\Thesis_Data\Global_SSM.png', dpi=300)
