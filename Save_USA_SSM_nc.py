"""
Thesis_MSc_Lijie Save_USA_SSM_nc
date: 22-Sep-2021
author: leojay
Contact: l.zhang-8@student.utwente.nl
-------------------------------------
Description: 
"""
# Library
import os
import netCDF4 as nc
import numpy as np
import gdal
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

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

def write_nc_USA_SSM(file,len_time, ssm_data):
    ncfile = nc.Dataset(file, mode='w', format='NETCDF4')

    # Create dimensions
    lat_dim = ncfile.createDimension('lat', 104)  # Latitude axis
    lon_dim = ncfile.createDimension('lon', 236)  # Longitude axis
    time_dim = ncfile.createDimension('time', len_time)  # Unlimited axis (can be appended to).
    for dim in ncfile.dimensions.items():
        print(dim)

    # Create variables. (A varialbe has a name, a type, a shape, and some data values)
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees'
    lat.long_name = 'latitude'

    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees'
    lon.long_name = 'longitude'

    # Define a 3D variable to hold the data.
    ssm = ncfile.createVariable('ssm', np.float32, ('time', 'lat', 'lon'))  # Note: unlimited dimension is leftmost.
    ssm.units = 'cm3/cm3'
    ssm.standard_name = 'Surface Soil Moisture'  # This is a CF standard name.
    ssm[:] = ssm_data

    ncfile.close()

write_nc_USA_SSM(r'a:\Thesis_Data\RF_SSM_USA.nc', ssm_rf_us_18yrs.shape[0], ssm_rf_us_18yrs)
write_nc_USA_SSM(r'a:\Thesis_Data\ESA_CCI_SSM_USA.nc', ssm_cci_us.shape[0], ssm_cci_us)
