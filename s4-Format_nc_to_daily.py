"""
Thesis_MSc_Lijie s4-Format_nc_to_daily
date: 23-May-2021
author: leojay
Contact: leojayak@gmail.com
-------------------------------------
Description: 
"""
# libraries
from netCDF4 import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt

work_dir = r'/home/yijian/RandomForest/RF_gridded/Global_27830_stripes'
os.chdir(work_dir)
folder_in = 'L3_RF_SSM_NC'
folder_out = 'L3_RF_SSM_NC_daily'

if not os.path.exists(folder_out):
    os.mkdir(folder_out)

ncfile = Dataset(os.path.join(folder_in, 'RF_SSM_global.nc'), mode='r')
print(ncfile)

time_all = ncfile['time'][:]
ssm_all = ncfile['ssm']


def Write_in_nc(file_name, file_data, file_time):
    # # Create dimensions
    nc_file = Dataset(file_name, mode='w', format='NETCDF4')
    # Create dimensions
    lat_dim = nc_file.createDimension('lat', 512)  # Latitude axis
    lon_dim = nc_file.createDimension('lon', 1440)  # Longitude axis
    time_dim = nc_file.createDimension('time', 1)  # Unlimited axis (can be appended to).

    lat = nc_file.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees'
    lat.long_name = 'latitude'
    lat[:] = 69.875 - np.arange(512) * 0.25

    lon = nc_file.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees'
    lon.long_name = 'longitude'
    lon[:] = np.arange(1440) * 0.25 - 179.875

    time = nc_file.createVariable('time', np.int_, ('time',))
    time.units = 'YYYYMMDD'
    time.long_name = 'time'
    time[:] = file_time

    ssm = nc_file.createVariable('ssm', np.float32, ('lat', 'lon'))  # Note: unlimited dimension is leftmost.
    ssm.units = 'cm3/cm3'
    ssm.standard_name = 'Surface Soil Moisture'  # This is a CF standard name.
    ssm = file_data

    nc_file.close()


# plt.imshow(ssm[100, :, :])
# plt.savefig(os.path.join(folder_out, 'test100.jpg'))


# for idx in np.arange(len(time)):
for idx in np.arange(1000, 1005):
    print('In the processing: ', idx)
    file_name = os.path.join(folder_out, str(time_all[idx]) + '.nc')
    print(np.nanmean(ssm_all[idx, :, :]))
    Write_in_nc(file_name, np.array(ssm_all[idx, :, :]), time_all[idx])

# nc = Dataset(r'c:\Users\leojay\Desktop\20021230.nc', 'r')
