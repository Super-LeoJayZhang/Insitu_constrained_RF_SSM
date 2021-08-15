"""
Thesis_MSc_Lijie Draw_global_DEM_NDVI_precipitation
date: 09-Jul-2021
author: leojay
Contact: leojayak@gmail.com
-------------------------------------
Description: 
"""
# libraries
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import gdal
import numpy as np
import cartopy as cart

# Set the working path
work_path = r'a:\Thesis_Data\Research_area'
os.chdir(work_path)
folder_in = 'Downloaded_GEE'

file_p0 = os.path.join(folder_in, 'Precipitation0.tif')
file_p1 = os.path.join(folder_in, 'Precipitation1.tif')
file_p2 = os.path.join(folder_in, 'Precipitation2.tif')

p0 = gdal.Open(file_p0).ReadAsArray()
p1 = gdal.Open(file_p1).ReadAsArray()
p2 = gdal.Open(file_p2).ReadAsArray()

precipitation = np.concatenate((p0, p1, p2), 1)

file_ndvi0 = os.path.join(folder_in, 'NDVI_0.tif')
file_ndvi1 = os.path.join(folder_in, 'NDVI_1.tif')
file_ndvi2 = os.path.join(folder_in, 'NDVI_2.tif')

NDVI0 = gdal.Open(file_ndvi0).ReadAsArray()
NDVI1 = gdal.Open(file_ndvi1).ReadAsArray()
NDVI2 = gdal.Open(file_ndvi2).ReadAsArray()

NDVI = np.concatenate((NDVI0, NDVI1, NDVI2), 1)


def draw_global_map(data, cmap_unit, vmin, vmax, cmap):
    # /update: 03-07-2021.
    data_crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.set_extent([-180, 180, -60, 75])

    # Grid
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(['$\mathregular{180^o}$ W', '$\mathregular{90^o}$ W', '$\mathregular{0^o}$',
                        '$\mathregular{90^o}$ E', '$\mathregular{180^o}$ E'])
    ax.set_yticks([75, 60, 30, 0, -30, -60])
    ax.set_yticklabels(
        ['$\mathregular{75^o}$ N', '$\mathregular{60^o}$ N', '$\mathregular{30^o}$ N', '$\mathregular{0^o}$ N',
         '$\mathregular{30^o}$ S', '$\mathregular{60^o}$ S'])
    ax.add_feature(cart.feature.LAKES, edgecolor='k', facecolor='w')
    # ax.add_feature(cfeature.OCEAN, color='white')
    ax.add_feature(cart.feature.OCEAN, zorder=10, edgecolor='k', facecolor='w')
    # norm = Normalize(vmin=0, vmax=0.4)
    # Add the Data
    plt.imshow(data, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax, extent=[-180, 180, -60, 75])
    # Add the ColorBar
    cb = plt.colorbar(ax=ax, orientation="horizontal", aspect=50, shrink=0.6, extend='both')
    cb.set_label(cmap_unit)


draw_global_map(precipitation * 1000, 'Precipitation (mm)', vmin=0, vmax=1500, cmap='RdYlBu')
plt.savefig('Global_yearly_precipitation.pdf', doi=600)
draw_global_map(NDVI[30: 30+540, ::] * 0.0001, 'NDVI', vmin=-0.2, vmax=0.6, cmap='RdYlGn')
plt.savefig('Glboal_yearly_NDVI.pdf')