"""
date: 25-May-2020 / updated: 26-June-2021.
author: leojay
Contact: leojayak@gmail.com
-------------------------------------
Description: This script used to read ESA-CCI COMBINED\v04.4 soil moisture data into daily SSM in the format of tiff
    for every single year. but also the mean value from 2000-2018.
"""
# libraries
import os
from datetime import datetime
import gdal
import netCDF4 as nc
import numpy as np


def writeTiff(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
    """
    This function is used to save the multi-bands raster data into a geo-tiff file.
    """
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
        # Create the file
    driver = gdal.GetDriverByName("GTiff")
    data_set = driver.Create(path, im_width, im_height, im_bands, datatype)
    data_set.SetGeoTransform(im_geotrans)  # GeoTrans
    data_set.SetProjection(im_proj)  # Projection
    for i in range(im_bands):
        data_set.GetRasterBand(i + 1).WriteArray(im_data[i])


file_basemap = r'a:\Thesis_Data\L3_RF_gridded\sand_global.tif'
gdal_basemap = gdal.Open(file_basemap)
geo_trans = (-180.0, 0.25, 0, 90, 0, -0.25)
# geo_trans = gdal_basemap.GetGeoTransform()
proj = gdal_basemap.GetProjection()

# time (start running)
time_0 = datetime.now()
start_time = time_0.strftime("%Y-%m-%d %H:%M:%S")
print('Program start running at: %s. ' % start_time)

# Set the working path
work_path = r'a:\Thesis_Data\L1_ESA_CCI\ESA-CCI\COMBINED\v04.4'
os.chdir(work_path)
folder_daily = r'a:\Thesis_Data\L3_RF_gridded\ESA_CCI_MAP\daily_year'
folder_yearly_mean = r'a:\Thesis_Data\L3_RF_gridded\ESA_CCI_MAP\yearly_mean'

for year in np.arange(2000, 2019):
    # Mark the processing status.
    print('In the processing: %s' % str(year))
    folder_name = str(year)
    files = os.listdir(folder_name)
    #     # Number of bands.
    n = len(files)
    result_year = np.arange(n * 720 * 1440).reshape(n, 720, 1440).astype(float)
    result_year[::] = np.nan

    # Loop of every single bands
    for idx, file_name in enumerate(files):
        file = os.path.join(folder_name, file_name)
        # Open the file through NC
        file_obj = nc.Dataset(file)
        # lon = file_obj.variables['lon'][:].data
        # lat = file_obj.variables['lat'][:].data
        sm = file_obj.variables['sm'][:].data[0, ::]

        sm[sm == -9999] = np.nan
        result_year[idx, ::] = sm
    # Save the result
    file_out = os.path.join(folder_daily, 'ESA_CCI_MERGED_' + str(year) + '.tif')
    writeTiff(result_year, 1440, 720, n, geo_trans, proj, file_out)


# Calculate the mean value from 2000 to 2018
result_years = np.arange(720 * 1440).reshape(720, 1440).astype(float)
result_years[::] = 0
files_year = os.listdir(folder_daily)
n_days = 0
for idx, file_name in enumerate(files_year):
    print(idx)
    gdal_file = gdal.Open(os.path.join(folder_daily, file_name))
    ds_year = gdal_file.ReadAsArray()
    n_days = n_days + ds_year.shape[0]
    result_years = result_years + np.nansum(ds_year, axis=0)
result_mean = result_years / n_days

result_mean[result_mean == 0] = np.nan

# Save the mean Tiff on the local drive.
result_mean_name = r'a:\Thesis_Data\L3_RF_gridded\ESA_CCI_MAP\all_years_mean_value\ESA-CCI_Cv04.4_mean_2000-2008.tif'
writeTiff(result_mean, 1440, 720, 1, geo_trans, proj, result_mean_name)
