"""
Thesis_MSc_Lijie Synchronize_the_GEE_downloaded_pts
date: 17-Jun-2020
author: leojay
Contact: leojayak@gmail.com
-------------------------------------
Description: This script is used to synchronize the time-series from GEE.
    1. Extract the needed information.
    2. precipitation: into API.
    3. NDVI: S-G filter -> interpolate.
"""
# libraries
import os
from datetime import datetime
import pandas as pd

# time (start running)
time_0 = datetime.now()
start_time = time_0.strftime("%Y-%m-%d %H:%M:%S")
print('Program start running at: %s. ' % start_time)

# Set the working path.
work_path = r'a:\Thesis_Data\L3_Analysis_spatial_resolution'
os.chdir(work_path)
# Input path.
path_input = r'L0_pts_fr_GEE'
folder_p_27830_in = os.path.join(path_input, 'ERA5_27830')
folders_lst_in = os.listdir(os.path.join(path_input, 'MOD11A1'))
folders_ndvi_in = os.listdir(os.path.join(path_input, 'MOD13A1'))
# Output Path.
path_output = r'L1_time_series_on_stations'
folder_p_27830_out = os.path.join(path_output, 'ERA5_27830')
if not os.path.exists(folder_p_27830_out):
    os.mkdir(folder_p_27830_out)


def Convert_gee_pts_precipitation(file_gee_pts_p, path_out):
    """
    :param file_gee_pts_p:
    :param path_out:
    :return:
    This function is used to convert the data of GEE_CSV into the record based on the station.
    """
    # Read the CSV file.
    df = pd.read_csv(file_gee_pts_p)
    # remove the system:index, which is not used for further process (Reduce the memory size).
    df = df.drop(df.columns[0], axis=1)
    # Convert the 'id' from string into 'Integer' for the convenience of the comparison.
    df['id'] = df['id'].astype(int)
    # Filter the data based on the id (Remove the Duplicate value)
    ids = df['id'].drop_duplicates().sort_values(inplace=False)

    # Loop of the 'ID'.
    for id in ids:
        df_station = pd.DataFrame(columns=['total_precipitation'])
        df_temp = df[df['id'] == id]
        # convert the total_precipitation unit from [m] into [mm]
        df_station['total_precipitation'] = df_temp['first'] * 1000
        df_station.index = df_temp['date']

        # Network Information.
        network = df_temp.iloc[0]['network']  # network
        station = df_temp.iloc[0]['station']  # station

        # save the df_station
        out_name = os.path.join(path_out, (network + '_' + station + '.csv'))

        if not os.path.exists(out_name):
            df_station.to_csv(out_name)
        else:
            df_previous = pd.read_csv(out_name, index_col=0)
            df_after = pd.concat([df_previous, df_station])
            df_after.to_csv(out_name)


def Convert_gee_pts_lst(file_gee_pts_lst, path_out):
    """
    :param file_gee_pts_lst:
    :param path_out:
    :return:
    """
    # Read the CSV file.
    df = pd.read_csv(file_gee_pts_lst)
    # remove the system:index, which is not used for further process (Reduce the memory size).
    df = df.drop([df.columns[0], '.geo'], axis=1)
    # Remove the first line (format of string)
    df = df.drop(df.index[0])
    # Convert the 'id' from string into 'Integer' for the convenience of the comparison.
    df['id'] = df['id'].astype(int)
    # Filter the data based on the id (Remove the Duplicate value)
    ids = df['id'].drop_duplicates().sort_values(inplace=False)

    # Loop of the 'ID'.
    for id in ids:
        columns = ['LST_Day_1km', 'LST_Night_1km', 'LST_DAILY', 'LST_Diff']
        df_station = pd.DataFrame(columns=columns)
        df_temp = df[df['id'] == id]
        # If both the QC_Day and QC_Night is 0.
        df_temp_day = df_temp['LST_Day_1km'].astype(float)
        df_temp_night = df_temp['LST_Night_1km'].astype(float)
        # Save the info.
        df_station['LST_Day_1km'] = df_temp_day
        df_station['LST_Night_1km'] = df_temp_night
        df_station['LST_DAILY'] = (df_temp_day + df_temp_night) / 2.0 * 0.02 - 273.15
        df_station['LST_Diff'] = (df_temp_day - df_temp_night) * 0.02
        # Set the index
        df_station.index = df_temp['date']

        # Network Information.
        network = df_temp.iloc[0]['network']  # network
        station = df_temp.iloc[0]['station']  # station

        # save the df_station
        out_name = os.path.join(path_out, (network + '_' + station + '.csv'))

        if not os.path.exists(out_name):
            df_station.to_csv(out_name)
        else:
            df_previous = pd.read_csv(out_name, index_col=0)
            df_after = pd.concat([df_previous, df_station])
            df_after.to_csv(out_name)


def Convert_gee_pts_ndvi(file_gee_pts_ndvi, path_out):
    """
    :param file_gee_pts_ndvi:
    :param path_out:
    :return:
    """
    # Read the CSV file.
    df = pd.read_csv(file_gee_pts_ndvi)
    # remove the system:index, which is not used for further process (Reduce the memory size).
    df = df.drop([df.columns[0], '.geo'], axis=1)
    df = df.drop(df.index[0])
    # Convert the 'id' from string into 'Integer' for the convenience of the comparison.
    df['id'] = df['id'].astype(int)
    # Filter the data based on the id (Remove the Duplicate value)
    ids = df['id'].drop_duplicates().sort_values(inplace=False)

    # Loop of the 'ID'.
    for id in ids:
        df_station = pd.DataFrame(columns=['NDVI', 'EVI'])
        df_temp = df[df['id'] == id]
        # Convert the NDVI and EVI based on the scale: 0.0001.
        df_station['NDVI'] = df_temp['NDVI'].astype(float) * 0.0001
        df_station['EVI'] = df_temp['EVI'].astype(float) * 0.0001
        df_station.index = df_temp['date']

        # Network Information.
        network = df_temp.iloc[0]['network']  # network
        station = df_temp.iloc[0]['station']  # station

        # save the df_station
        out_name = os.path.join(path_out, (network + '_' + station + '.csv'))

        if not os.path.exists(out_name):
            df_station.to_csv(out_name)
        else:
            df_previous = pd.read_csv(out_name, index_col=0)
            df_after = pd.concat([df_previous, df_station])
            df_after.to_csv(out_name)



""" Precipitation """
# list the files and sort the order (from 2000 to 2019)
files_g_p = os.listdir(folder_p_27830_in)
files_g_p.sort()    # Sort.
# Loop of the files.
for file_name in files_g_p:
    print('Converting the Precipitation data: ', file_name)
    file = os.path.join(folder_p_27830_in, file_name)
    if not os.path.exists(file):
        print('The path is not correct')
    Convert_gee_pts_precipitation(file, folder_p_27830_out)

""" LST """
# List the folders of the different spatial resolution.
print('\n Start Converting the LST data.')
folder_lst_out = os.path.join(path_output, 'MOD11A1')
if not os.path.exists(folder_lst_out):
    os.mkdir(folder_lst_out)
for folder_lst_in in folders_lst_in:
    # out_name
    folder_lst_1000_out = os.path.join(folder_lst_out, folder_lst_in)
    if not os.path.exists(folder_lst_1000_out):
        os.mkdir(folder_lst_1000_out)
    # Obtain the list of files (from 2000 to 2019) and sort.
    files_g_lst = os.listdir(os.path.join(path_input, 'MOD11A1', folder_lst_in))
    files_g_lst.sort()
    for file_name in files_g_lst:
        file = os.path.join(path_input, 'MOD11A1', folder_lst_in, file_name)
        if not os.path.exists(file):
            print('The path is not correct')
        # Call the function
        Convert_gee_pts_lst(file, folder_lst_1000_out)

""" NDVI """
# List the folders of NDVI with the different spatial resolution.
print('\n Start Converting the NDVI data.')
folder_ndvi_out = os.path.join(path_output, 'MOD13A1')
if not os.path.exists(folder_ndvi_out):
    os.mkdir(folder_ndvi_out)
# Loop of the folders.
for folder_ndvi_in in folders_ndvi_in:
    # out_name for the NDVI
    folder_ndvi_1000_out = os.path.join(folder_ndvi_out, folder_ndvi_in)
    if not os.path.exists(folder_ndvi_1000_out):
        os.mkdir(folder_ndvi_1000_out)
    # Obtain the list of files (from 2000 to 2019) and sort.
    files_g_ndvi = os.listdir(os.path.join(path_input, 'MOD13A1', folder_ndvi_in))
    files_g_ndvi.sort()
    for file_name in files_g_ndvi:
        file = os.path.join(path_input, 'MOD13A1', folder_ndvi_in, file_name)
        print('In the processing:', file)
        if not os.path.exists(file):
            print('The path is not correct')
        # Call the function
        Convert_gee_pts_ndvi(file, folder_ndvi_1000_out)

# time (end running)
time_1 = datetime.now()
end_time = time_1.strftime("%Y-%m-%d %H:%M:%S")
print('program end running at: %s.' % end_time)
time_consuming = (time_1 - time_0).seconds
print('It takes %d seconds to run this scripts' % int(time_consuming))
