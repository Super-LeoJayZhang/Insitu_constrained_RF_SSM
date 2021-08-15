"""
Thesis_MSc_Lijie NDVI_SG_interpolate
date: 19-Jun-2020
author: leojay
Contact: leojayak@gmail.com
-------------------------------------
Description: 
"""
# libraries
import os
import pandas as pd
from scipy.signal import savgol_filter

# Set the working path
work_path = r'a:\Thesis_Data\L3_Analysis_spatial_resolution\L1_time_series_on_stations'
os.chdir(work_path)
folder_input = r'MOD13A1'
folder_output = r'MOD13A1_interpolate'
if not os.path.exists(folder_output):
    os.mkdir(folder_output)


def S_G_Interplate(file, out_name):
    # Open the file
    df = pd.read_csv(file, index_col=0)
    # S-G filter
    df['NDVI_SG'] = savgol_filter(df['NDVI'], 7, 3)
    df['EVI_SG'] = savgol_filter(df['EVI'], 7, 3)

    ''' Interpolate per year, since the first image is always year-01-01.'''
    # Create the data frame (from 2000-02-18 to 2019-12-31).
    date_rng_df = pd.date_range(df.index[0], '2019-12-31')
    df_interplate = pd.DataFrame(columns=['date'], index=date_rng_df)
    df_interplate.index = pd.to_datetime(df_interplate.index)
    df_interplate['date'] = date_rng_df

    # Merge the crated data frame with the data
    df_interplate = pd.merge(df_interplate, df, left_index=True, right_index=True, how='left')

    # Interpolate NDVI_SG
    df_interplate['NDVI_SG_linear'] = df_interplate['NDVI_SG'].interpolate(method='linear')
    df_interplate['NDVI_SG_nearest'] = df_interplate['NDVI_SG'].interpolate(method='nearest')
    df_interplate['NDVI_SG_polynomial'] = df_interplate['NDVI_SG'].interpolate(method='polynomial', order=5)

    # Interpolate EVI_SG
    df_interplate['EVI_SG_linear'] = df_interplate['EVI_SG'].interpolate(method='linear')
    df_interplate['EVI_SG_nearest'] = df_interplate['EVI_SG'].interpolate(method='nearest')
    df_interplate['EVI_SG_polynomial'] = df_interplate['EVI_SG'].interpolate(method='polynomial', order=5)

    # Save the result into CSV
    df_interplate.to_csv(out_name)


# Loop of the folders
for folder_name in os.listdir(folder_input):
    folder_in = os.path.join(folder_input, folder_name)
    folder_out = os.path.join(folder_output, folder_name)
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    files_name = os.listdir(folder_in)
    # Loop of the files
    for idx, file_name in enumerate(files_name):
        print('In the processing of %s. %d of %d' % (folder_name, len(files_name), idx))
        file = os.path.join(folder_in, file_name)
        if not os.path.exists(file):
            print('Invalid path: ', file)
        file_out = os.path.join(folder_out, file_name)
        S_G_Interplate(file, file_out)

