"""
Thesis_MSc_Lijie s3_boxplot_verfication_by_climate
date: 05-Jul-2021
author: leojay
Contact: l.zhang-8@student.utwente.nl
-------------------------------------
Description: 
"""
# libraries
import os
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# File Directory.
file = r'a:\Thesis_Data\L2_RF\RF_regional_without_geographic_info_v2\all\Evaluation_metrics.csv'
df = pd.read_csv(file, index_col=0)
print(df.head(5))

df_rf = df[df.columns[[0, 1, 3, 4, 5, 6, -1]]]
df_rf.columns = ['network', 'station', 'RMSE', 'ubRMSE', 'r', 'Land cover type', 'Climate Zones']
df_rf['source'] = 'RF'

df_cci = df[df.columns[[0, 1, 8, 9, 10, 6, -1]]]
df_cci.columns = ['network', 'station', 'RMSE', 'ubRMSE', 'r', 'Land cover type', 'Climate Zones']
df_cci['source'] = 'CCI'

df_merged = pd.concat([df_rf, df_cci])
df_merged_sort = df_merged.sort_values(by='Climate Zones')

# Draw the figures based on the climate.
figs, axs = plt.subplots(nrows=3, figsize=(12, 10), constrained_layout=True)

ax0 = sns.boxplot(data=df_merged_sort, x='Climate Zones', y='RMSE', hue="source", width=0.3, ax=axs[0])
ax1 = sns.boxplot(data=df_merged_sort, x='Climate Zones', y='ubRMSE', hue="source", width=0.3, ax=axs[1])
ax2 = sns.boxplot(data=df_merged_sort, x='Climate Zones', y='r', hue="source", width=0.3, ax=axs[2])

plt.savefig(r'C:\Users\leojay\Desktop\ESSD-Long Term SSM\Figures\Boxplot_ClimateZones.jpg', dpi=300)

# figs, axs = plt.subplots(nrows=3, figsize=(12, 10), constrained_layout=True)
# ax0 = sns.boxplot(data=df_merged_sort,  x='Land cover type', y='RMSE', hue="source", width=0.3, ax=axs[0])
# ax1 = sns.boxplot(data=df_merged_sort,  x='Land cover type', y='ubRMSE', hue="source", width=0.3, ax=axs[1])
# ax2 = sns.boxplot(data=df_merged_sort,  x='Land cover type', y='r', hue="source", width=0.3, ax=axs[2])
