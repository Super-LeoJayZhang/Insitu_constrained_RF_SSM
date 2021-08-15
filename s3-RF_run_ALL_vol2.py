"""
 RF_run_ALL_vol1
date: 19-May-2020. All the networks (stations).
 author: L.Zhang
Contact: leojayak@gmail.com
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
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from pylab import hist2d
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Parameters
split_scale = 0.7  # split of the data
model_trees = 100  # The number of trees that used in the RF model.

# steps switch
step_training = 1  # Train the RF model, 1: training the model and save the trained *.m 0: read the model directly.

step_spliting = 0  # 1: Read the data, split and save the time-series of predictor variables into separated CSV file.


# time (start running)
time_0 = datetime.now()
start_time = time_0.strftime("%Y-%m-%d %H:%M:%S")
print('Program start running at: %s. ' % start_time)

""" Working path, folders and files"""
# Set the working path
work_path = r'a:\Thesis_Data\L2_RF'
os.chdir(work_path)

# Define the input and output folder.
folder_input = 'sychronized\RF_GEE_Dataset_v01'
folder_output = 'RF_ALL_vol2'
if not os.path.exists(folder_output):
    os.mkdir(folder_output)

# Define the folder that save the 30% of the data, which would be used for the evaluation of the model.
folder_test_data = os.path.join(folder_output, 'Test_folder_30')
if not os.path.exists(folder_test_data):
    os.mkdir(folder_test_data)

# Define the folder that used to store the estimated soil moisture of every single station.
folder_estimated_sm = os.path.join(folder_output, 'Estimated_SM')
if not os.path.exists(folder_estimated_sm):
    os.mkdir(folder_estimated_sm)

# Define the folder: save the time-series.
folder_time_series = os.path.join(folder_output, 'time_series_figure')
if not os.path.exists(folder_time_series):
    os.mkdir(folder_time_series)

# File that contains the information of climate zone and land cover
file_lc = 'sychronized\station_climate_lc.csv'

# Obtain the files list of the synchronized data.
files = os.listdir(folder_input)


""" Read the data """
if step_training == 1:
    # Read the data, split and save.  Append all the predictor variables and labels.
    if step_spliting == 1:
        # Initialize the data frame.
        df_all = pd.DataFrame()

        # Read data in a loop.
        for idx, file_name in enumerate(files):
            # Mark
            print('In the Processing: Count: %d, Station Name: %s' % (idx, file_name))

            # file_path
            file = os.path.join(folder_input, file_name)
            # Read the data from the CSV file into DATA FRAME.
            df = pd.read_csv(file, index_col=0)

            # Split the data with train and result.
            train_num = int(len(df) * split_scale)
            df_train = df.head(train_num)
            df_test = df.tail(len(df) - train_num)
            df_all = df_all.append(df)

            file_out_test = os.path.join(folder_test_data, file_name)
            df_test.to_csv(file_out_test)
            del df
        # Save the train_data
        file_train_data = os.path.join(folder_output, 'train_data_all.csv')
        df_all.to_csv(file_train_data)

        ''' Save the test data'''
        for idx, file_name in enumerate(files):
            # Mark
            print('In the Processing: Count: %d, Station Name: %s' % (idx, file_name))
            # file_path
            file = os.path.join(folder_input, file_name)
            # Read the data from the CSV file into DATA FRAME.
            df = pd.read_csv(file, index_col=0)

            # Split the data with train and result.
            train_num = int(len(df) * split_scale)
            df_train = df.head(train_num)
            df_test = df.tail(len(df) - train_num)

            file_out_test = os.path.join(folder_test_data, file_name)
            df_test.to_csv(file_out_test)
            del df
    # The data has already processed, just read it.
    else:
        file_train_data = os.path.join(folder_output, 'train_data_all.csv')
        df_all = pd.read_csv(file_train_data, index_col=0)

    ''' Train the RF Model'''
    # Linear
    df_all_1 = df_all[['soil moisture', 'api', 'LST_DAILY', 'LST_Diff', 'NDVI_SG_linear',
                       'EVI_SG_linear', 'silt', 'sand', 'clay', 'lon', 'lat']]

    # drop the invalid data
    print("Before dropna(), the records of data for trainning: %d" % len(df_all_1))
    df_all_1 = df_all_1.dropna()
    df_all_1[df_all_1['soil moisture'] < 0.01] = np.nan
    df_all_1 = df_all_1.dropna()
    print("After dropna(), the records of data for trainning: %d" % len(df_all_1))

    # labels (soil moisture)
    labels = df_all_1['soil moisture']

    df_all_1 = df_all_1.drop('soil moisture', axis=1)
    feature_list = list(df_all_1.columns)  # Features that will be used to train the model.
    feature_list[0], feature_list[3], feature_list[4] = 'API', 'NDVI', 'EVI'
    # convert the feature from pandas into numpy
    features = np.array(df_all_1)

    # Randomly split the features and labels .
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25)
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

    print('\n Feature Importance: ')
    for f in range(features.shape[1]):
        print('%s: %.3f' % (feature_list[indices[f]], importances[indices[f]]))

    # Save the feature importance in a *.csv file.
    df_importances = pd.DataFrame(columns=['feature_list', 'importances', 'indices'])
    df_importances['feature_list'] = feature_list
    df_importances['importances'] = importances
    df_importances['indices'] = indices
    df_importances.to_csv(os.path.join(folder_output, 'Feature_importance.csv'))

    # Figures.
    plt.figure(figsize=(20, 10))
    #  sort the feature_list
    feature_list_new = feature_list.copy()
    for idx in np.arange(len(feature_list)):
        feature_list_new[idx] = feature_list[list(indices)[idx]]
    # plt.title("Importance of features")
    plt.bar(range(train_features.shape[1]), importances[indices],
            color='b', yerr=std[indices], align='center')
    # set the x labels
    plt.xticks(range(train_features.shape[1]), feature_list_new, fontsize=16)
    plt.xlim(-1, train_features.shape[1])
    # Add the xlabel and ylabel
    plt.xlabel('Name of features', fontsize=16)
    plt.ylabel('Relative importance [%]', fontsize=16)

    # Save the figure on the local drive.
    plt.savefig(os.path.join(folder_output, 'Feature_Importances.jpg'))
    plt.close()

    # show the validation of the test set
    plt.figure()
    # label of the axis.
    plt.xlabel('In-situ soil moisture [$\mathregular{cm^3}$/$\mathregular{cm^3}$]', fontsize=12)
    plt.ylabel('Estimated soil moisture [$\mathregular{cm^3}$/$\mathregular{cm^3}$]', fontsize=12)
    # plot the data with the density (number of pixels)
    h = hist2d(list(test_labels), list(predictions), bins=40, cmap='PuBu',
               range=[[0, 0.6], [0, 0.6]])
    cb = plt.colorbar()
    cb.set_label('Numbers of points')
    # Add 1:1 line
    x = np.arange(0, 0.6, 0.1, dtype=float)
    y = x
    # Add the information of RMSE and r on the figure.
    plt.text(0.1, 0.54, 'RMSE: %.2f' % rmse, fontdict={'size': 12})
    plt.text(0.1, 0.50, 'ubRMSE: %.2f' %cal_ubrmse(test_labels, predictions)[0], fontdict={'size': 12})
    plt.text(0.1, 0.46, 'r: %.2f' % r, fontdict={'size': 12})
    plt.text(0.1, 0.42, 'Num: %d' % len(test_features), fontdict={'size': 12})

    plt.plot(x, y, color='black')
    plt.savefig(os.path.join(folder_output, 'ComparisonOfTestDataset.png'))
    plt.close()


    """ Evaluate the RF model"""
    # Read the file of land cover.
    df_lc = pd.read_csv(file_lc, index_col=0)
    # Reload the trained RF model.
    rf = joblib.load(os.path.join(folder_output, 'train_model_1.m'))
    ''' Evaluate the model on the rest data'''
    files_evaluate = os.listdir(folder_test_data)
    df_metrics = pd.DataFrame(columns=['network', 'station', 'number of data records', 'rmse',
                                       'ubrmse', 'r', 'lc', 'climate', 'lon', 'lat'])
    for idx, file_name in enumerate(files_evaluate):
        # Mark the processing status.
        print('In the Processing. %d of %d' % (idx, len(files_evaluate)))

        file = os.path.join(folder_test_data, file_name)
        df = pd.read_csv(file, index_col=0)

        df_1 = df[['soil moisture', 'api', 'LST_DAILY', 'LST_Diff', 'NDVI_SG_linear',
                   'EVI_SG_linear', 'silt', 'sand', 'clay', 'lon', 'lat']]
        df_1 = df_1.dropna()
        df_1[df_1['soil moisture'] < 0.01] = np.nan
        df_1 = df_1.dropna()

        if len(df_1) > 10:
            in_situ = df_1['soil moisture']
            df_1 = df_1.drop('soil moisture', axis=1)
            features = np.array(df_1)
            estimated = rf.predict(features)

            # calculate RMSE and ubRMSE
            insitu_rmse = np.sqrt(mean_squared_error(in_situ, estimated))
            insitu_ubrmse = cal_ubrmse(in_situ, estimated)[0]
            p_r = pearsonr(in_situ, estimated)[0]
            number_n = len(in_situ)

            # Initialize the metrics
            series_metrics = pd.Series(index=['network', 'station', 'number of data records', 'rmse',
                                              'ubrmse', 'r', 'lc', 'climate', 'lon', 'lat'])

            # Network and station
            network = file_name.split('_')[0]
            station = file_name.strip(network).strip('_').strip('.csv')
            series_metrics['network', 'station'] = network, station
            # Land cover and climate zones.
            series_lc = df_lc[df_lc.index == file_name.strip('.csv')]
            if len(series_lc) == 1:
                lc_2010 = series_lc['LC_2010'].values[0]
                climate = series_lc['climate_insitu'].values[0]
                series_metrics['lc', 'climate'] = lc_2010, climate
            # Metrics of the validation.
            series_metrics['number of data records'] = number_n
            series_metrics['rmse', 'ubrmse', 'r'] = insitu_rmse, insitu_ubrmse, p_r
            df_metrics.loc[file_name] = series_metrics
            del series_metrics, lc_2010, climate, network, station

            '''Save the In-situ soil mositure as well as the estimated one.'''
            df_insitu = pd.DataFrame(columns=['in_situ', 'estimated'])
            df_insitu['in_situ'] = in_situ
            df_insitu['estimated'] = estimated

            # Save the time-series figure.
            df_insitu.plot()
            plt.savefig(os.path.join(folder_time_series, file_name.strip('.csv') + '.jpg'))
            plt.close()

            # Define the output file
            file_comparison_station = os.path.join(folder_estimated_sm, file_name)
            # Save the estimated soil moisture in the format of CSV.
            df_insitu.to_csv(file_comparison_station)

        # Save the metrics data frame
        file_metrics = os.path.join(folder_output, 'Evaluation_metrics.csv')
        df_metrics.to_csv(file_metrics)

# time (end running)
time_1 = datetime.now()
end_time = time_1.strftime("%Y-%m-%d %H:%M:%S")
print('program end running at: %s.' % end_time)
time_consuming = (time_1 - time_0).seconds
print('It takes %d seconds to run this scripts' % int(time_consuming))
