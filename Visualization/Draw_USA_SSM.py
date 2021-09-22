"""
 Draw_USA_SSM
date: 05-Jul-2021
 author: L.Zhang
Contact: leojayak@gmail.com
-------------------------------------
Description: 
"""
# libraries
import os
import netCDF4 as nc
import numpy as np
import gdal
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

file_rf = r'a:\Thesis_Data\RF_SSM_USA.nc'
file_cci = r'a:\Thesis_Data\ESA_CCI_SSM_USA.nc'

nc_rf = nc.Dataset(file_rf, 'r')
ssm_rf = nc_rf['ssm']
ssm_rf_mean = np.nanmean(ssm_rf[:], 0)

nc_cci = nc.Dataset(file_cci, 'r')
ssm_cci = nc_cci['ssm']
ssm_cci_mean = np.nanmean(ssm_cci[:], 0)


# ------------------------
# Self-defined function
# -------------------------
def draw_US_ssm(ax, data, title='title', cmap='YlGnBu', colorbar_name='SSM', vmin=0, vmax=0.4):
    # data_crs = ccrs.PlateCarree()
    # fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    # ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')

    ax.set_extent([-125, -66, 24, 46])
    # ax.set_extent([-125, -66, 24, 50])
    ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='w')
    # Grid
    ax.set_xticks([-120, -100, -80])
    ax.set_xticklabels(['$\mathregular{120^o}$ W', '$\mathregular{100^o}$ W', '$\mathregular{80^o}$ W'])
    ax.set_yticks([50, 40, 30])
    ax.set_yticklabels(['$\mathregular{50^o}$ N', '$\mathregular{40o}$ N', '$\mathregular{30^o}$ N'])

    # norm = Normalize(vmin=0, vmax=0.35)
    plt.imshow(data, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, extent=[-125, -66, 24, 50])
    # Add the ColorBar
    cb = plt.colorbar(ax=ax, aspect=50, shrink=0.8, fraction=0.046, extend='both')
    cb.set_label(fr'{colorbar_name}' + ' $(m^3m{^-3}$)', fontsize=12)
    plt.title(f'{title}', fontsize=12)


data_crs = ccrs.PlateCarree()
plt.subplots(3, 1, figsize=(8, 12))
ax0 = plt.subplot(3, 1, 1, projection=data_crs)
draw_US_ssm(ax0, ssm_rf_mean, title='(a) RF SSM')
ax1 = plt.subplot(3, 1, 2, projection=data_crs)
draw_US_ssm(ax1, ssm_cci_mean, title='(b) ESA CCI SSM')
ax2 = plt.subplot(3, 1, 3, projection=data_crs)
draw_US_ssm(ax2, ssm_rf_mean - ssm_cci_mean,
            title='(c) Difference between RF and ESA CCI SS,', cmap='RdBu', vmin=-0.2, vmax=0.1)

plt.savefig(r'a:\Thesis_Data\USA_SSM.png', dpi=300)
plt.savefig(r'a:\Thesis_Data\USA_SSM.pdf', dpi=300)
