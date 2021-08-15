"""
Thesis_MSc_Lijie RF_Train_Test.py
date: 25-Jun-2021
author: leojay
Contact: leojayak@gmail.com
-------------------------------------
Description: 
"""
# libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.externals import joblib
import netCDF4 as nc

# Parameters
split_scale = 0.7  # split of the data
n_estimators = 100  # The number of trees that used in the RF model.
n_jobs = 6  # Number of cores for parallel processing.


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
        # Initialize the model
        rf = RandomForestRegressor(n_estimators=self.n_estimator, n_jobs=6)
        print('Random Forest regression model has been Initialized')
        # Train the model
        rf.fit(self.train_features, self.train_labels)
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


def write_nc(file_path, features_list_, importances_, std_, indices_, rmse_, r2_, r_, predictions_, test_labels_):
    nc_out = nc.Dataset(file_path, 'w')
    # Create the dimension.
    d_features = nc_out.createDimension('d_features', len(features_list_))
    d_testdata = nc_out.createDimension('d_data', len(predictions_))
    d_1 = nc_out.createDimension('d_1', 1)

    # Feature list
    features_list = nc_out.createVariable('features_list', str, 'd_features')
    importances = nc_out.createVariable('importances', np.float32, 'd_features')
    std = nc_out.createVariable('std', np.float32, 'd_features')
    indices = nc_out.createVariable('indices', np.int, 'd_features')
    # Statistics Metrics.
    rmse = nc_out.createVariable('rmse', np.float32, 'd_1')
    r2 = nc_out.createVariable('r2', np.float32, 'd_1')
    r = nc_out.createVariable('r', np.float32, 'd_1')
    # Data.
    predictions = nc_out.createVariable('predications', np.float32, 'd_data')
    in_situ = nc_out.createVariable('in-situ', np.float32, 'd_data')

    # Define the data.
    features_list[:] = np.array(features_list_)
    importances[:] = importances_
    std[:] = std_
    indices[:] = indices_
    rmse[:] = rmse_
    r2[:] = r2_
    r[:] = r_
    predictions[:] = predictions_
    in_situ[:] = test_labels_

    nc_out.close()


# Set the working path
work_path = r'a:\Thesis_Data\L2_RF\RF_Train_Test'
os.chdir(work_path)
file_train_test = 'training_testing_data.csv'

# Open the file.
df_all = pd.read_csv(file_train_test, index_col=0)
# Select the features for the model running.
df_all_1 = df_all[['soil moisture', 'api', 'LST_DAILY', 'LST_Diff', 'NDVI_SG_linear',
                   'EVI_SG_linear', 'silt', 'sand', 'clay', 'lon', 'lat']]
# Drop out the invalid value.
print("Before dropna(), the records of data for trainning: %d" % len(df_all_1))
df_all_1['soil moisture'].loc[df_all['soil moisture'] < 0.01] = np.nan
# df_all_1[df_all_1['soil moisture'] < 0.01] = np.nan
df_all_1 = df_all_1.dropna()
print("After dropna(), the records of data for trainning: %d" % len(df_all_1))

# labels (soil moisture)
labels = df_all_1['soil moisture']
df_all_1 = df_all_1.drop('soil moisture', axis=1)
feature_list = list(df_all_1.columns)  # Features that will be used to train the model.
feature_list[0], feature_list[3], feature_list[4] = 'API', 'NDVI', 'EVI'
feature_list[5], feature_list[6], feature_list[7] = 'Silt', 'Sand', 'Clay'
# convert the feature from pandas into numpy
features = np.array(df_all_1)
# Randomly split the features and labels .
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                 random_state=42)
file_rf_model = 'train_model_geo.m'
rf_model = Run_RF(train_features, test_features, train_labels, test_labels, feature_list, path=os.curdir)
print('train: %d' % len(train_features))
print('test: %d' % len(test_features))
rf_model.n_estimator = n_estimators
rf, predictions, importances, std, indices, rmse, r2, r = rf_model.train_model()
print('rmse: %.3f' % rmse)
print('r2: %.3f' % r2)
print('r: %.3f' % r)
# Save the model on the local drive for the usage of next time.
joblib.dump(rf, file_rf_model)

file_nc = 'Feature_importance_geo.nc'
if os.path.exists(file_nc):
    os.remove(file_nc)
write_nc(file_nc, feature_list, importances, std, indices, rmse, r2, r,
         predictions_=predictions, test_labels_=test_labels)


""" """
file_rf_model = 'train_model.m'
df_all_2 = df_all_1.drop(['lon', 'lat'], axis=1)
# convert the feature from pandas into numpy
features = np.array(df_all_2)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                 random_state=42)
feature_list = list(df_all_2.columns)  # Features that will be used to train the model.
rf_model = Run_RF(train_features, test_features, train_labels, test_labels, feature_list, path=os.curdir)
print('train: %d' % len(train_features))
print('test: %d' % len(test_features))
rf_model.n_estimator = n_estimators
rf, predictions, importances, std, indices, rmse, r2, r = rf_model.train_model()
print('rmse: %.3f' % rmse)
print('r2: %.3f' % r2)
print('r: %.3f' % r)
# Save the model on the local drive for the usage of next time.
joblib.dump(rf, os.path.join('train_model.m'))
file_nc1= 'Feature_importance.nc'
if os.path.exists(file_nc1):
    os.remove(file_nc1)
write_nc(file_nc1, feature_list, importances, std, indices, rmse, r2, r,
         predictions_=predictions, test_labels_=test_labels)