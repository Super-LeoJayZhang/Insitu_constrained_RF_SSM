"""
Thesis_MSc_Lijie s4-Draw_global_regional_SSM
date: 26-Jun-2021
author: leojay
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

file_rf = 'RF_SSM_global.nc'

# Read RF predicted data.
nc_rf = nc.Dataset(file_rf, 'r')
rf_ssm = nc_rf['ssm']
time = list(nc_rf['time'][:])

# Create a time index list to separate the data.
date_index_list = [0]
for idx, year in enumerate(np.arange(2001, 2020)):
    index = time.index(year * 10000 + 101)
    date_index_list.append(index)
date_index_list.append(len(time))


def write_nc(file, time_list, ssm_data):
    ncfile = nc.Dataset(file, mode='w', format='NETCDF4')

    # Create dimensions
    lat_dim = ncfile.createDimension('lat', 512)  # Latitude axis
    lon_dim = ncfile.createDimension('lon', 1440)  # Longitude axis
    time_dim = ncfile.createDimension('time', len(time_list))  # Unlimited axis (can be appended to).
    for dim in ncfile.dimensions.items():
        print(dim)

    # Create variables. (A varialbe has a name, a type, a shape, and some data values)
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees'
    lat.long_name = 'latitude'

    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees'
    lon.long_name = 'longitude'

    time = ncfile.createVariable('time', np.int_, ('time',))
    time.units = 'YYYYMMDD'
    time.long_name = 'time'

    time[:] = time_list
    lat[:] = 69.875 - np.arange(512) * 0.25

    # Define a 3D variable to hold the data.
    ssm = ncfile.createVariable('ssm', np.float32, ('time', 'lat', 'lon'))  # Note: unlimited dimension is leftmost.
    ssm.units = 'cm3/cm3'
    ssm.standard_name = 'Surface Soil Moisture'  # This is a CF standard name.
    ssm[:] = ssm_data

    ncfile.close()


for index in np.arange(len(date_index_list) - 1):
    index_s = date_index_list[index]
    index_e = date_index_list[index + 1]
    ssm_data = rf_ssm[index_s: index_e, :, :]
    year = 2000 + index
    file_name = 'RF_global_' + str(year) + '.nc'

    write_nc(file_name, time[index_s:index_e], ssm_data)


