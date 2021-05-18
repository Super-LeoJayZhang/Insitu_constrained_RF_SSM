"""
Thesis_MSc_Lijie RF_run_vol1
date: 10-May-2020  // update 02-07-2020. Add the ESA-CCI data.
author: leojay
Contact: l.zhang-8@student.utwente.nl
-------------------------------------
Description: Run the RF model with the predictors from GEE.
    API, NDVI, EVI, LST
    SILT, SAND CLAY
"""
# libraries
import os
from datetime import datetime
from Functions_general_li import Run_RF, cal_ubrmse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pylab import hist2d
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.externals import joblib
import seaborn as sns


# Function to Read and split the files into Training and testing set and evaluation and validation set.
def Read_Split_frFiles(folder_input, folder_output, folder_evaluate_data, network_name='all'):
    """
    :param folder_input: The folder that contains all the synchronized files.
    :param folder_output: The root output folder of the model.
    :param folder_test_data: A sub_folder that contains the test files (testing set).
    :param network_name: The network that shall be selected.
    :return: Save the files into the defined path.
    """

    # Obtain the files list of the synchronized data.
    files = os.listdir(folder_input)
    # Initialize the data frame.
    df_all = pd.DataFrame()
    # Read data in a loop.
    for file_name in files:
        if network_name == 'all':
            # file_path
            file = os.path.join(folder_input, file_name)
            # Read the data from the CSV file into DATA FRAME.
            df = pd.read_csv(file, index_col=0)
            # Split the data with train and result.
            train_num = int(len(df) * split_scale)
            df_train = df.head(train_num)
            df_test = df.tail(len(df) - train_num)
            df_all = df_all.append(df_train)
            file_out_test = os.path.join(folder_evaluate_data, file_name)
            df_test.to_csv(file_out_test)
            del df
        else:
            # only select the data from Network
            if network_name in file_name:
                # file_path
                file = os.path.join(folder_input, file_name)
                # Read the data from the CSV file into DATA FRAME.
                df = pd.read_csv(file, index_col=0)
                # Split the data with train and result.
                train_num = int(len(df) * split_scale)
                df_train = df.head(train_num)
                df_validate = df.tail(len(df) - train_num)
                df_all = df_all.append(df_train)
                file_out_test = os.path.join(folder_evaluate_data, file_name)
                df_validate.to_csv(file_out_test)
                del df
        # Save the train_data
    file_train_data = os.path.join(folder_output, 'training_testing_data.csv')
    df_all.to_csv(file_train_data)


# Function to draw the feature_importance.
def draw_feature_importance(feature_list, importances, save_path, show_figure=0):
    #  sort the feature_list
    feature_list_new = feature_list.copy()
    for idx in np.arange(len(feature_list)):
        feature_list_new[idx] = feature_list[list(indices)[idx]]

    plt.figure(figsize=(12, 6))
    plt.barh(range(train_features.shape[1]), importances[indices],
             xerr=std[indices], align='center')
    plt.yticks(range(train_features.shape[1]), feature_list_new, fontsize=20)
    plt.ylim(-1, train_features.shape[1])
    # Add the labels
    # plt.ylabel('Name of Features', fontsize=10)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(save_path)
    if show_figure == 0:
        plt.close()


def draw_testing_result(in_situ_SSM, predicted_ssm, save_path, show_figure=0):
    plt.figure()
    # label of the axis.
    plt.xlabel('In-situ soil moisture [cm3/cm3]', fontsize=12)
    plt.ylabel('Estimated soil moisture [cm3/cm3]', fontsize=12)
    # plot the data with the density (number of pixels)
    h = hist2d(list(in_situ_SSM), list(predicted_ssm), bins=40, cmap='PuBu',
               range=[[0, 0.6], [0, 0.6]])
    cb = plt.colorbar()
    cb.set_label('Numbers of points')
    # Add 1:1 line
    x = np.arange(0, 0.6, 0.1, dtype=float)
    y = x
    # Add the information of RMSE and r on the figure.
    plt.text(0.1, 0.5, 'RMSE: %.2f' % rmse, fontdict={'size': 12})
    plt.text(0.1, 0.46, 'r: %.2f' % r, fontdict={'size': 12})
    plt.text(0.1, 0.42, 'Num: %d' % len(test_features), fontdict={'size': 12})

    plt.plot(x, y, color='black')
    plt.savefig(save_path)
    if show_figure == 0:
        plt.close()


# Parameters
split_scale = 0.7  # split of the data
model_trees = 100  # The number of trees that used in the RF model.

# Keys to switch on the process
key_read_frFiles = False

# time (start running)
time_0 = datetime.now()
start_time = time_0.strftime("%Y-%m-%d %H:%M:%S")
print('Program start running at: %s. ' % start_time)

""" Working path, folders and files"""
# Set the working path
work_path = r'a:\Thesis_Data\L2_RF'
os.chdir(work_path)

# Define the input and output folder.
folder_input = 'sychronized\RF_GEE_Dataset_v02'
# File that contains the information of climate zone and land cover
file_lc = 'sychronized\station_climate_lc.csv'

folder_output_path = 'RF_regional_with_geographic_info_v2'
if not os.path.exists(folder_output_path):
    os.mkdir(folder_output_path)

networks_list = ['USCRN', 'SCAN', 'SNOTEL', 'SOILSCAPE', 'all']
for network_name in networks_list:
    folder_output = os.path.join(folder_output_path, network_name)

    if not os.path.exists(folder_output):
        os.mkdir(folder_output)

    # Define the folder that save the 30% of the data, which would be used for the evaluation of the model.
    folder_evaluate_data = os.path.join(folder_output, 'Evaluate_validation_folder_30')
    if not os.path.exists(folder_evaluate_data):
        os.mkdir(folder_evaluate_data)

    # Define the folder that used to store the estimated soil moisture of every single station.
    folder_estimated_sm = os.path.join(folder_output, 'Predicted_SM_time_series')
    if not os.path.exists(folder_estimated_sm):
        os.mkdir(folder_estimated_sm)

    # Define the folder: save the time-series.
    folder_time_series = os.path.join(folder_output, 'time_series_figure')
    if not os.path.exists(folder_time_series):
        os.mkdir(folder_time_series)

    """ Read the data """
    if key_read_frFiles:
        Read_Split_frFiles(folder_input, folder_output, folder_evaluate_data, network_name)

    # Read the training and testing data set.
    df_all = pd.read_csv(os.path.join(folder_output, 'training_testing_data.csv'), index_col=0)

    ''' Train the RF Model'''
    df_all_1 = df_all[['soil moisture', 'api', 'LST_DAILY', 'LST_Diff', 'NDVI_SG_linear',
                       'EVI_SG_linear', 'silt', 'sand', 'clay', 'lon', 'lat']]

    # Drop out the invalid value.
    print("Before dropna(), the records of data for trainning: %d" % len(df_all_1))
    df_all_1[df_all_1['soil moisture'] < 0.01] = np.nan
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
    # Run the random forest regression model
    rf_model = Run_RF(train_features, test_features, train_labels, test_labels, feature_list, folder_output)
    print('train: %d' % len(train_features))
    print('test: %d' % len(test_features))
    rf_model.n_estimator = model_trees
    rf, predictions, importances, std, indices, rmse, r2, r = rf_model.train_model()
    print('rmse: %.3f' % rmse)
    print('r: %.3f' % r)

    # Save the model on the local drive for the usage of next time.
    joblib.dump(rf, os.path.join(folder_output, 'train_model_1.m'))

    ''' Feature Importance Figure.'''
    img_importance = os.path.join(folder_output, 'Feature_Importance.jpg')
    draw_feature_importance(feature_list, importances, img_importance)

    ''' Testing result'''
    img_testing = os.path.join(folder_output, 'Testing_Result.jpg')
    draw_testing_result(test_labels, predictions, img_testing)

    # Read the file of land cover.
    df_lc = pd.read_csv(file_lc, index_col=0)
    # # Reload the trained RF model.
    # rf = joblib.load(os.path.join(folder_output, 'train_model_1.m'))
    ''' Evaluate the model on the rest data'''
    files_evaluate = os.listdir(folder_evaluate_data)

    list_metrics = ['network', 'station', 'NOP', 'RMSE_RF', 'ubRMSE_RF', 'r_RF',
                    'LC', 'Climate', 'RMSE_cci', 'ubRMSE_cci', 'r_cci']
    df_metrics = pd.DataFrame(columns=list_metrics)
    # Loop of all the evaluating files.
    for idx, file_name in enumerate(files_evaluate):
        # Mark the processing status.
        print('In the Processing. %d of %d' % (idx, len(files_evaluate)))

        file = os.path.join(folder_evaluate_data, file_name)
        df = pd.read_csv(file, index_col=0)

        df_1 = df[['soil moisture', 'api', 'LST_DAILY', 'LST_Diff', 'NDVI_SG_linear',
                   'EVI_SG_linear', 'silt', 'sand', 'clay', 'esa_cci', 'lon', 'lat']]
        df_1 = df_1.dropna()
        df_1[df_1['soil moisture'] < 0.01] = np.nan
        df_1 = df_1.dropna()

        if len(df_1) > 10:
            in_situ = df_1['soil moisture']
            esa_cci = df_1['esa_cci']
            df_1 = df_1.drop(['soil moisture', 'esa_cci'], axis=1)
            features = np.array(df_1)
            estimated = rf.predict(features)

            '''Save the In-situ soil mositure as well as the estimated one.'''
            df_insitu = pd.DataFrame(columns=['in_situ', 'estimated'])
            df_insitu['in_situ'] = in_situ
            df_insitu['estimated'] = estimated
            df_insitu['esa_cci'] = esa_cci

            # Define the output file
            file_comparison_station = os.path.join(folder_estimated_sm, file_name)
            # Save the estimated soil moisture in the format of CSV.
            df_insitu.to_csv(file_comparison_station)

            # calculate RMSE and ubRMSE (RF_trained, and ESA_CCI)
            rmse_RF = np.sqrt(mean_squared_error(in_situ, estimated))
            ubrmse_RF = cal_ubrmse(in_situ, estimated)[0]
            p_r_RF = pearsonr(in_situ, estimated)[0]
            rmse_cci = np.sqrt(mean_squared_error(in_situ, esa_cci))
            ubrmse_cci = cal_ubrmse(in_situ, esa_cci)[0]
            p_r_cci = pearsonr(in_situ, esa_cci)[0]
            number_n = len(in_situ)

            # # Save the time-series figure.
            # ax = df_insitu.plot(yticks=[0, 0.2, 0.4, 0.6], figsize=[10, 6])
            # plt.text(x=0.1, y=0.5, s=str('RMSE_RF  %.2f' % rmse_RF))
            # plt.text(x=0.1, y=0.47, s=str('RMSE_CCI  %.2f' % rmse_cci))
            # plt.text(x=0.1, y=0.44, s=str('r_RF  %.2f' % p_r_RF))
            # plt.text(x=0.1, y=0.41, s=str('r_CCI  %.2f' % p_r_cci))
            #
            # plt.text('RMSE_RF: %.2f' % rmse_RF)
            # plt.savefig(os.path.join(folder_time_series, file_name.strip('.csv') + '.jpg'))
            # plt.close()

            # Initialize the metrics
            series_metrics = pd.Series(index=list_metrics, dtype='object')

            # Network and station
            network = file_name.split('_')[0]
            station = file_name.strip(network).strip('_').strip('.csv')
            series_metrics['network', 'station'] = network, station
            # Land cover and climate zones.
            series_lc = df_lc[df_lc.index == file_name.strip('.csv')]
            if len(series_lc) == 1:
                lc_2010 = series_lc['LC_2010'].values[0]
                climate = series_lc['climate_insitu'].values[0]
                series_metrics['LC', 'Climate'] = lc_2010, climate
            # Metrics of the validation.
            series_metrics['number of data records'] = number_n
            series_metrics['RMSE_RF', 'ubRMSE_RF', 'r_RF'] = rmse_RF, ubrmse_RF, p_r_RF
            series_metrics['RMSE_cci', 'ubRMSE_cci', 'r_cci'] = rmse_cci, ubrmse_cci, p_r_cci
            df_metrics.loc[file_name] = series_metrics
            del series_metrics, lc_2010, climate, network, station

    # Save the metrics data frame
    file_metrics = os.path.join(folder_output, 'Evaluation_metrics.csv')
    df_metrics.to_csv(file_metrics)

    # sns.set(style='whitegrid')
    ax = sns.boxplot(data=df_metrics, order=['RMSE_RF', 'RMSE_cci', 'ubRMSE_RF', 'ubRMSE_cci'],
                     palette=[sns.color_palette("hls", 8)[5], sns.color_palette("hls", 8)[1],
                              sns.color_palette("hls", 8)[5], sns.color_palette("hls", 8)[1]],
                     width=0.3)
    ax.set_ylabel(ylabel='Surface Soil Moisture [$\mathregular{cm^3}$/$\mathregular{cm^3}$]')
    plt.savefig(os.path.join(folder_output, 'BoxPlot_rmse_ubrmse.jpg'))
    plt.close()

    ax = sns.boxplot(data=df_metrics, order=['r_RF', 'r_cci'],
                     palette=[sns.color_palette("hls", 8)[5], sns.color_palette("hls", 8)[1]],
                     width=0.3)
    plt.savefig(os.path.join(folder_output, 'BoxPlot_r.jpg'))
    plt.close()

# time (end running)
time_1 = datetime.now()
end_time = time_1.strftime("%Y-%m-%d %H:%M:%S")
print('program end running at: %s.' % end_time)
time_consuming = (time_1 - time_0).seconds
print('It takes %d seconds to run this scripts' % int(time_consuming))


def Draw_time_series_station(file_in, figure_out):
    df = pd.read_csv(file_in, index_col=0)
    # calculate RMSE and ubRMSE (RF_trained, and ESA_CCI)
    rmse_RF = np.sqrt(mean_squared_error(df['in_situ'], df['estimated']))
    ubrmse_RF = cal_ubrmse(df['in_situ'], df['estimated'])[0]
    p_r_RF = pearsonr(df['in_situ'], df['estimated'])[0]
    rmse_cci = np.sqrt(mean_squared_error(df['in_situ'], df['esa_cci']))
    ubrmse_cci = cal_ubrmse(df['in_situ'], df['esa_cci'])[0]
    p_r_cci = pearsonr(df['in_situ'], df['esa_cci'])[0]
    number_n = len(df['in_situ'])

    # Plot the figure.
    df.columns = ['In_situ', 'RF_predicted', 'ESA_CCI']
    ax = df.plot(yticks=[0, 0.2, 0.4, 0.6], figsize=[10, 6])
    # plt.text(x=0.1, y=0.53, s=str('Number of records:  %d' % number_n))
    network_name = file_in.split('\\')[-1].split('.csv')[0].split('_')[0]
    station_name = file_in.split('\\')[-1].split('.csv')[0].split('_')[-1]

    # Add the statistic information
    plt.text(x=0.1, y=0.57, s=str("Network:  %s.   Station:  %s." % (network_name, station_name)))
    plt.text(x=0.1, y=0.54, s=str('RMSE_RF:  %.2f [$\mathregular{cm^3}$/$\mathregular{cm^3}$].  '
                                  'ubRMSE:  %.2f  [$\mathregular{cm^3}$/$\mathregular{cm^3}$].' % (rmse_RF, ubrmse_RF)))
    plt.text(x=0.1, y=0.51, s=str('RMSE_CCI  %.2f [$\mathregular{cm^3}$/$\mathregular{cm^3}$].  '
                                  'ubRMSE:  %.2f  [$\mathregular{cm^3}$/$\mathregular{cm^3}$].' % (rmse_cci, ubrmse_cci)))
    plt.text(x=0.1, y=0.48, s=str('r_RF:  %.2f ' % p_r_RF))
    plt.text(x=0.1, y=0.45, s=str('r_CCI:  %.2f ' % p_r_cci))

    ax.set_ylabel(ylabel='Surface Soil Moisture [$\mathregular{cm^3}$/$\mathregular{cm^3}$]')

    plt.savefig(figure_out)
    plt.close()
