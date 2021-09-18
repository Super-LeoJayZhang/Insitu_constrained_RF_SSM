"""
Insitu_constrained_RF_SSM Boxplot_evaluation_periods
date: 18-Sep-2021
author: L.Zhang
Contact: leojayak@gmail.com
-------------------------------------
Description: 
"""
# libraries
import os
import pandas as pd
import seaborn as sns

# Set the working path
file_metrics = r'C:\Users\leojay\Desktop\ESSD-Long Term SSM\Insitu_constrained_RF_SSM\Evaluation_metrics.csv'
df_metrics = pd.read_csv(file_metrics, index_col=0)
# Rename the columns name.
df_metrics.columns = ['network', 'station', 'NOP',
                      'RMSE: RF SSM', 'ubRMSE: RF SSM', 'Person Correlation: RF SSM',
                      'LandCover', 'Climate',
                      'RMSE: ESA-CCI SSM', 'ubRMSE: ESA-CCI SSM', 'Person Correlation: ESA-CCI SSM']


ax = sns.boxplot(data=df_metrics, order=['RMSE: RF SSM', 'RMSE: ESA-CCI SSM', 'ubRMSE: RF SSM', 'ubRMSE: ESA-CCI SSM'],
                     palette=[sns.color_palette("hls", 8)[5], sns.color_palette("hls", 8)[1],
                              sns.color_palette("hls", 8)[5], sns.color_palette("hls", 8)[1]],
                     width=0.3)

