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
import matplotlib.pyplot as plt

# Set the working path
file_metrics = r'C:\Users\leojay\Desktop\ESSD-Long Term SSM\Insitu_constrained_RF_SSM\Evaluation_metrics.csv'
df_metrics = pd.read_csv(file_metrics, index_col=0)
# Rename the columns name.
df_metrics.columns = ['network', 'station', 'NOP',
                      'RMSE: RF SSM', 'ubRMSE: RF SSM', 'Person Correlation: RF SSM',
                      'LandCover', 'Climate',
                      'RMSE: ESA-CCI SSM', 'ubRMSE: ESA-CCI SSM', 'Person Correlation: ESA-CCI SSM']

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
axs[0] = sns.boxplot(ax=axs[0], data=df_metrics,
                     order=['RMSE: RF SSM', 'RMSE: ESA-CCI SSM', 'ubRMSE: RF SSM', 'ubRMSE: ESA-CCI SSM'],
                     palette=[sns.color_palette("hls", 8)[5], sns.color_palette("hls", 8)[1],
                              sns.color_palette("hls", 8)[5], sns.color_palette("hls", 8)[1]],
                     width=0.4)

axs[0].set_ylabel(ylabel='Surface Soil Moisture [$\mathregular{cm^3}$/$\mathregular{cm^3}$]', fontsize=12)
axs[0].set_title(r'(a) Boxplot of RMSE and ubRMSE', fontsize=12)
axs[1] = sns.boxplot(ax=axs[1], data=df_metrics, order=['Person Correlation: RF SSM', 'Person Correlation: ESA-CCI SSM'],
                     palette=[sns.color_palette("hls", 8)[5], sns.color_palette("hls", 8)[1]],
                     width=0.3)
axs[1].set_ylabel(ylabel='Person Correlation (r)', fontsize=12)
axs[1].set_title(r'(b) Boxplot of Person Correlation (r)')

plt.savefig(r'C:\Users\leojay\Desktop\ESSD-Long Term SSM\ESSD_manuscript\Scientific_papers\Figures\Boxplot_evaluation.jpg'
            , dpi=300)