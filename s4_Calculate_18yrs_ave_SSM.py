"""
date: 03-Jul-2021
 author: L.Zhang
Contact: leojayak@gmail.com
-------------------------------------
Description: This script used to calculate the 18 yrs average of SSM from both ESA-CCI and RF-predicted.
"""
# libraries
import os
import numpy as np
import gdal
import matplotlib.pyplot as plt

# Set the working path
work_path = r'a:\Thesis_Data\L3_RF_gridded'
os.chdir(work_path)

# I set the full path for convenience of further organization.
folder_cci = r'a:\Thesis_Data\L3_RF_gridded\ESA_CCI_processing\Merged_ESA_CCI_years'
folder_rf = r'a:\Thesis_Data\L3_RF_gridded\RF_predicted_global_nc'
folder_out = r'Global_Figure'
if not os.path.exists(folder_out):
    os.mkdir(folder_out)


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

# Calculate the 18 yrs average of ESA-CCI soil moisture.

for idx, year in enumerate(np.arange(2000, 2019)):
    print(f'Calculating the global mean value  of ESA-CCI soil moisture data: {idx}: {year}')
    file_esa = os.path.join(folder_cci, 'ESA_CCI_MERGED_' + str(year) + '.tif')
    gdal_esa = gdal.Open(file_esa)
    ds = gdal_esa.ReadAsArray()[:, 80: 720 - 128, :]
    if year == 2000:
        esa_18yrs = ds
    else:
        esa_18yrs = np.concatenate((esa_18yrs, ds), axis=0)


    geo_trans = gdal_esa.GetGeoTransform()
    proj = gdal_esa.GetProjection()

geo_trans = (-180.0, 0.25, 0, 90, 0, -0.25)
# geo_trans = gdal_basemap.GetGeoTransform()
proj = gdal_basemap.GetProjection()
esa_18yrs_mean[esa_18yrs_mean[::] == 0]=np.nan
esa_18yrs_mean = esa_18yrs_mean / 19.0
writeTiff(esa_18yrs_mean, 720, 1440, 1, geo_trans, proj, 'ESACCI_18yrs_mean.tif')


# Calculate the 18 yrs average of RF soil moisture.
rf_18yrs_mean = np.arange(512 *1440).reshape(512, 1440).astype(float)
rf_18yrs_mean[:] = np