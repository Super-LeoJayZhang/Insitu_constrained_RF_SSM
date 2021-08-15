"""
thesis_proj2 Functions_general_li
date: 24-Feb-2020
author: leojay
Contact: leojayak@gmail.com
"""

from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy.stats import pearsonr
import os
import gdal

def file_path_check(files):
    """
    This Function is used to check the input file path.
    """
    for file in files:
        if not os.path.exists(file):
            print('The file path is not correct: ', file)
        else:
            print('Valid file path: ', file)


class Date_Range:
    """ This class is used to calculate the date range
    Input: date_begins (the first index of the data frame), date_ends (the last index of the date frame)
    OUtput: The date range.
    """

    def __init__(self, date_begins, date_ends):
        # Information of the start date
        self.year_start = date_begins.year
        self.month_start = date_begins.month
        self.day_start = date_begins.day
        # Information of the end date
        self.year_end = date_ends.year
        self.month_end = date_ends.month
        self.day_end = date_ends.day

    def cal_range(self):
        """ This function is to calculate the date range."""
        # calculate the Day of Year
        doy_start = (datetime(self.year_start, self.month_start, self.day_start) - datetime(self.year_start, 1, 1)).days
        date_start = datetime(self.year_start, 1, 1) + timedelta(doy_start)
        # Calculate the start and end year.
        doy_end = (datetime(self.year_end, self.month_end, self.day_end) - datetime(self.year_end, 1, 1)).days
        date_end = datetime(self.year_end, 1, 1) + timedelta(doy_end)
        # Return the value
        return date_start, date_end


class Run_RF:
    """
    This class is used to Run the Random Forest Regression Model
    """

    def __init__(self, train_features, test_features, train_labels, test_labels,
                 feature_list, path):
        self.train_features = train_features
        self.test_features = test_features
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.feature_list = feature_list
        self.path = path
        self.n_estimator = 1000

    def train_model(self):
        t0 = datetime.now()
        # Initialize the model
        rf = RandomForestRegressor(n_estimators=self.n_estimator, n_jobs=6)
        print('Random Forest regression model has been Initialized')
        # Train the model
        rf.fit(self.train_features, self.train_labels)
        # know the training time
        t1 = datetime.now()
        time_consuming = (t1 - t0).seconds
        print('Training time: %d seconds' % time_consuming)
        print('Trained Completed!')

        # calculate the estimated values
        predictions = rf.predict(self.test_features)
        # calculate the metrics
        importances = rf.feature_importances_
        # calculate the standard deviation of the importance
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        # sort the indices with decreasing ratios.
        indices = np.argsort(importances)[::-1]  # [::-1] inverse
        # Calculate the errors of the prediction result
        rmse = np.sqrt(mean_squared_error(self.test_labels, predictions))  # Root Mean Square Error
        r2 = r2_score(self.test_labels, predictions)  # Determination coefficient
        r = pearsonr(self.test_labels, predictions)[0]  # Pearson Correlation coefficient

        return rf, predictions, importances, std, indices, rmse, r2, r


# Function of API
def cal_API_li(p, k, n):
    """
    p: precipitation (Series from pandas)
    k: The recession coefficient
    n: the days that would affect the API
    """
    # Initialize the output
    api = pd.Series(index=p.index)
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


def read_tif(file):
    """
    This function is used to read the tiff file, and return the data in format [bands, lines, columns]
    And also Geo-transformation and Projection.
    """

    ds = gdal.Open(file)
    if ds is None:
        print('Error! Could not open file: %s.' % file)
        return

    data = []
    columns = ds.RasterXSize
    lines = ds.RasterYSize
    bands = ds.RasterCount
    geo_transformation = ds.GetGeoTransform()
    projection = ds.GetProjection
    for count_idx in range(bands):
        #  obtain the band of number count_i+1
        band_i = ds.GetRasterBand(count_idx + 1)
        band_data = band_i.ReadAsArray(0, 0, band_i.XSize, band_i.YSize)
        if count_idx == 0:
            data = band_data[np.newaxis, :]
        else:
            data = np.concatenate((data, band_data[np.newaxis, :]), axis=0)
    print('Successfully read data: %s.' % file)
    return data, columns, lines, bands, geo_transformation, projection


def cal_ubrmse(a, b):
    ''' This function is used to calculate the unbiased Root Mean Square Error'''
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    part_sq = ((a - a_mean) - (b - b_mean))**2
    ubrmse = np.sqrt(np.mean(part_sq))
    return ubrmse, a_mean, b_mean

