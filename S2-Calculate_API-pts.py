"""
Thesis_MSc_Lijie Calculate_API
date: 19-Jun-2020
author: leojay
Contact: leojayak@gmail.com
-------------------------------------
Description: This script is used to calculate the Antecedent Precipitation Index from the daily precipitation data.
"""
# libraries
import os
import pandas as pd
import numpy as np

# Set the parameters of the API equation.
k = 0.91
n = 34


def cal_API_li(p, k, n):
    """
    p: precipitation (Series from pandas)
    k: The recession coefficient
    n: the days that would affect the API
    """
    # Initialize the output
    api = pd.Series(index=p.index, dtype='float64')
    # loop for each day
    for idx_p in np.arange(len(p)):
        if idx_p < n:
            api[api.index[idx_p]] = np.nan

        else:
            api_t = 0
            for i in np.arange(n):
                api_t = api_t + p[p.index[idx_p - i]] * pow(k, i)
            api[api.index[idx_p]] = api_t

    return api


# Set the working path
work_path = r'a:\Thesis_Data\L3_Analysis_spatial_resolution'
os.chdir(work_path)
folder_p = r'L1_time_series_on_stations\ERA5_27830'
output_folder_p = r'L1_time_series_on_stations\ERA5_API_27830'
if not os.path.exists(output_folder_p):
    os.mkdir(output_folder_p)
# Get the list of file_names in the folder_p.
files_p = os.listdir(folder_p)

# Loop of every single file (precipitation of the station extracted from GEE)
for idx, file_name in enumerate(files_p):
    print('In the processing. %d of %d' % (len(files_p), idx+1))
    file = os.path.join(folder_p, file_name)
    if not os.path.exists(file):
        print('The path of the file is not corrected: ', file)
    # Open the file.
    df = pd.read_csv(file, index_col=0)
    # Calculate API.
    precipitation = df['total_precipitation']
    api = cal_API_li(precipitation, k, n)
    df['api'] = api

    # Save the NEW data frame into a CSV file
    out_file = os.path.join(output_folder_p, file_name)
    df.to_csv(out_file)