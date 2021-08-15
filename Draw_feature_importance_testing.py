"""
 Draw_feature_importance_testing.py
date: 25-Jun-2021
author: leojay
Contact: leojayak@gmail.com
-------------------------------------
Description: 
"""
# libraries
import os
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from pylab import hist2d

# Set the working path
work_path = r'a:\Thesis_Data\L2_RF\RF_Train_Test'
os.chdir(work_path)
file_feature0 = 'Feature_importance.nc'
file_feature1 = 'Feature_importance_geo.nc'


def read_nc(file):
    nc_feature = nc.Dataset(file, 'r')
    features_list = nc_feature['features_list'][:]
    importances = nc_feature['importances']
    std = nc_feature['std'][:]
    indices = nc_feature['indices'][:]
    rmse = nc_feature['rmse'][:]
    r2 = nc_feature['r2'][:]
    r = nc_feature['r'][:]
    predictions = nc_feature['predications']
    in_situ = nc_feature['in-situ']

    # Sorted feature list.
    features_list_new = features_list.copy()
    for idx in np.arange(len(features_list)):
        features_list_new[idx] = features_list[list(indices)[idx]]

    return features_list, importances, std, indices, rmse, r2, r, predictions, in_situ, features_list_new


features_list0, importances0, std0, indices0, rmse0, r20, r0, predictions0, in_situ0, features_list0_new = read_nc(
    file_feature0)
features_list1, importances1, std1, indices1, rmse1, r21, r1, predictions1, in_situ1, features_list1_new = read_nc(
    file_feature1)


def draw_testing_result(in_situ_SSM, predicted_ssm, rmse, r, save_path, show_figure=0):
    plt.figure()
    # label of the axis.
    plt.xlabel(r'In-situ soil moisture [$cm^3/cm^3$]', fontsize=12)
    plt.ylabel(r'Estimated soil moisture [$cm^3/cm^3$]', fontsize=12)
    # plot the data with the density (number of pixels)
    h = hist2d(list(in_situ_SSM), list(predicted_ssm), bins=40, cmap='PuBu',
               range=[[0, 0.4], [0, 0.4]])
    cb = plt.colorbar()
    cb.set_label('Numbers of points')
    # Add 1:1 line
    x = np.arange(0, 0.4, 0.1, dtype=float)
    y = x
    # Add the information of RMSE and r on the figure.
    plt.text(0.1, 0.35, 'RMSE: %.2f' % rmse, fontdict={'size': 12})
    plt.text(0.1, 0.3, 'r: %.2f' % r, fontdict={'size': 12})
    plt.text(0.1, 0.25, 'Num: %d' % len(predicted_ssm), fontdict={'size': 12})

    plt.plot(x, y, color='black')
    plt.savefig(save_path, dpi=300)
    if show_figure == 0:
        plt.close()


def draw_feature_importance(feature_list_new, importances, indices, std, save_path, show_figure=0):
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(feature_list_new)), importances[indices],
             xerr=std[indices], align='center')
    plt.yticks(range(len(feature_list_new)), feature_list_new, fontsize=20)
    plt.ylim(-1, len(feature_list_new))
    # Add the labels
    # plt.ylabel('Name of Features', fontsize=10)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(save_path, dpi=300)
    if show_figure == 0:
        plt.close()


draw_testing_result(in_situ0, predictions0, rmse0, r0, 'Test_withoutgeo.jpg')
draw_testing_result(in_situ1, predictions1, rmse1, r1, 'Test_geo.jpg')


draw_feature_importance(features_list0_new, importances0, indices0, std0, 'Importance_withoutgeo.jpg')
draw_feature_importance(features_list1_new, importances1, indices1, std1, 'Importance_geo.jpg')


