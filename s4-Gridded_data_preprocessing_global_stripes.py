"""
Thesis_MSc_Lijie Gridded_data_preprocessing_global_stripes
date: 05-Aug-2020
author: leojay
Contact: l.zhang-8@student.utwente.nl
-------------------------------------
Description: 
"""
# libraries
import os
from datetime import datetime
import pandas as pd
import numpy as np
import gdal
from scipy.signal import savgol_filter

# Set the parameters of the API equation.
k = 0.91
n = 34


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


def read_tif_simple(file):
    gdal_file = gdal.Open(file)
    ds = gdal_file.ReadAsArray()
    columns = gdal_file.RasterXSize
    lines = gdal_file.RasterYSize
    bands = gdal_file.RasterCount
    geo_trans = gdal_file.GetGeoTransform()
    proj = gdal_file.GetProjection()
    return ds, columns, lines, bands, geo_trans, proj


def cal_gridded_api(ds_p, n, k):
    """
    :param ds_p: the gridded data set of precipitation
    :param n: Number of days.
    :param k: the coefficient
    :return: Calculated API
    """
    ds_result = ds_p.copy()
    ds_result[::] = np.nan
    for idx in np.arange(ds_p.shape[0]):
        if idx < n:
            ds_result[idx, ::] = np.nan
        else:
            api_t = ds_result[0, ::]
            api_t[::] = 0
            for i in np.arange(n):
                api_t = api_t + (ds_p[[idx - i], ::]) * pow(k, i)
            ds_result[idx, ::] = api_t

    return ds_result


def main_transfer_api(file_input, file_ouptut):
    """
    :param file_input: Gridded precipitation file from Google Earth Engine.
    :param file_ouptut: Calculated Antecedent Precipitation Index.
    :return: Save the API.
    """
    # Read the precipitation file.
    ds_p, columns_p, lines_p, bands_p, geo_trans_p, proj_p = read_tif_simple(file_input)
    # Calculate the API.
    ds_result = cal_gridded_api(ds_p, n, k)
    # Save the out file.
    writeTiff(ds_result, columns_p, lines_p, bands_p, geo_trans_p, proj_p, file_ouptut)


def main_interpolate_VI(file_input, file_output, VI_dates):
    """
    :param file_input: Input gridded file from Google Earth Engine.
    :param file_output: path for the Interpolated Vegetation Index file.
    :param VI_dates: The range of the input file.
    :param n_dates: The number of dates for the output file.
    :return: Interpolated Vegetation Index file.
    """
    # Date Range for the Interpolated VI.
    date_rng_df = pd.date_range('2000-02-24', '2019-12-30')
    n_dates = len(date_rng_df)
    # Open the file through GDAL.
    ds, columns, lines, bands, geo_trans, proj = read_tif_simple(file_input)
    # Initialize the output.
    df_interpolated_grid = np.arange(n_dates * columns * lines, dtype=float).reshape(n_dates, lines, columns)
    df_interpolated_grid[::] = np.nan

    # Loop of lines
    for count_j in np.arange(lines):
        # Mark the Processing status.
        print(f"Total lines: {lines}. In the Processing {count_j}")
        # Loop of columns
        for count_k in np.arange(columns):
            # Read the time-series of NDVI on the pixel location of [count_j, count_k].
            ts_vi = ds[:, count_j, count_k]

            # S-G filter
            df_pixel_sg = pd.DataFrame(data=savgol_filter(ts_vi, 7, 3), index=VI_dates)
            df_pixel_sg.index = pd.to_datetime(df_pixel_sg.index)

            # Initialize the interpolated time-series.
            df_interplate = pd.DataFrame(columns=['date'], index=date_rng_df)
            df_interplate.index = pd.to_datetime(df_interplate.index)
            df_interplate['date'] = date_rng_df

            # Merge the crated data frame with the data
            df_merged = pd.merge(df_interplate, df_pixel_sg, left_index=True, right_index=True, how='left')
            df_merged.columns = ['Date', 'VI_ori']
            df_merged['VI_SG_linear'] = df_merged['VI_ori'].interpolate(method='linear')

            # Save the interpolated NDVI time-series into the DataFrame.
            vi_interpolated = df_merged['VI_SG_linear'].astype(float)
            vi_interpolated[vi_interpolated <= 0] = np.nan
            df_interpolated_grid[:, count_j, count_k] = vi_interpolated
    print(f"The {file_input} gridded data has been successfully interpolated !")
    writeTiff(df_interpolated_grid, columns, lines, n_dates, geo_trans, proj, file_output)


def Copy_file_to(file_input, file_output):
    """
    :param file_input: Path of the file input.
    :param file_output: Path of the file output
    :return: copied a TIFF file into a new path.
    """
    ds, columns, lines, bands, geo_trans, proj = read_tif_simple(file_input)
    writeTiff(ds, columns, lines, bands, geo_trans, proj, file_output)


def Check_file_path(list_files):
    for file in list_files:
        if os.path.exists(file):
            # print(f'Valid path {file}')
            pass
        else:
            print(f'Invalid path {file}')

# time (start running)
time_0 = datetime.now()
start_time = time_0.strftime("%Y-%m-%d %H:%M:%S")
print('Program start running at: %s. ' % start_time)

# Set the working path
work_path = r'a:\Thesis_Data\L3_RF_gridded\Global_gridded_stripes'
os.chdir(work_path)
folder_input = 'L0_downloaded'
folder_output = 'L1_formated'
if not os.path.exists(folder_output):
    os.mkdir(folder_output)

file_ndvi_point = os.path.join(folder_input, 'AACES_Farm01.csv')
ndvi_dates = pd.read_csv(file_ndvi_point, index_col=0, header=1).index

for count_i in np.arange(0, 36):
    print('\n\n\n ', '-'*30, count_i, '-'*30)

    # path of the precipitation file.
    file_p = os.path.join(folder_input, 'p_Global' + str(count_i) + '.tif')
    # path of the api
    file_api = os.path.join(folder_output, 'Gridded_API' + str(count_i) + '.tif')

    # Path for the vegetation index.
    file_ndvi = os.path.join(folder_input, 'NDVI_Global' + str(count_i) + '.tif')
    file_evi = os.path.join(folder_input, 'EVI_Global' + str(count_i) + '.tif')
    file_ndvi_interpolate = os.path.join(folder_output, 'GEE_interpolated_ndvi' + str(count_i) + '.tif')
    file_evi_interpolate = os.path.join(folder_output, 'GEE_interpolated_evi' + str(count_i) + '.tif')

    # Check the file path.
    Check_file_path([file_p, file_ndvi, file_ndvi])

    # Process the API calculation.
    main_transfer_api(file_p, file_api)

    # NDVI
    main_interpolate_VI(file_ndvi, file_ndvi_interpolate, ndvi_dates)

    # EVI
    main_interpolate_VI(file_evi, file_evi_interpolate, ndvi_dates)
