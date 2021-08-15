"""
Thesis_MSc_Lijie s4-Format_to_nc
date: 19-May-2021
author: L.Zhang
Contact: leojayak@gmail.com
-------------------------------------
Description: Convert the RF_SSM from tiff into netCDF4 format.
"""
# libraries
from netCDF4 import Dataset
import numpy as np
import gdal
import pandas as pd
import os


def read_tif_simple(file):
    gdal_file = gdal.Open(file)
    ds = gdal_file.ReadAsArray()
    columns = gdal_file.RasterXSize
    lines = gdal_file.RasterYSize
    bands = gdal_file.RasterCount
    geo_trans = gdal_file.GetGeoTransform()
    proj = gdal_file.GetProjection()
    return ds, columns, lines, bands, geo_trans, proj


work_dir = r'/home/yijian/RandomForest/RF_gridded/Global_27830_stripes'
os.chdir(work_dir)

folder_in = 'L2_predicted_RF_SSM_v2'
file_date = r'L1_formated/dates_lst_list.csv'
folder_out = 'L3_RF_SSM_NC'

if not os.path.exists(folder_out):
    os.mkdir(folder_out)


# Open a file, create a new Dataset.
ncfile = Dataset(os.path.join(folder_out, 'RF_SSM_global.nc'), mode='w', format='NETCDF4')
print(ncfile)

df_dates = pd.read_csv(file_date, index_col=0)

# Create dimensions
lat_dim = ncfile.createDimension('lat', 512)  # Latitude axis
lon_dim = ncfile.createDimension('lon', None)  # Longitude axis
time_dim = ncfile.createDimension('time', 7192)  # Unlimited axis (can be appended to).
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

time[:] = df_dates['0'].values
lat[:] = 69.875 - np.arange(512) * 0.25

# Define a 3D variable to hold the data.
ssm = ncfile.createVariable('ssm', np.float32, ('time', 'lat', 'lon'))  # Note: unlimited dimension is leftmost.
ssm.units = 'cm3/cm3'
ssm.standard_name = 'Surface Soil Moisture'  # This is a CF standard name.

def Add_data_into_nc(file, ssm):
    ds, columns, lines, bands, geo_trans, proj = read_tif_simple(file)
    x0 = ssm.shape[2]
    ssm[:, :, x0:x0+40] = ds
    return ssm

for count_i in np.arange(0, 36):
    file = os.path.join(folder_in, 'Estimated_SSM_global'+str(count_i)+'.tif')
    print(file, os.path.exists(file))
    ssm = Add_data_into_nc(file, ssm)
ncfile.close()






# For juwels to calculate the mean.

