"""
 s3_Draw_insitu_SSM_frequency_distribution.py
date: 05-Jul-2021
 author: L.Zhang
Contact: leojayak@gmail.com
-------------------------------------
Description: 
"""
# libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# sns.kdeplot(df_all['soil moisture'])
df_all = pd.read_csv(r'a:\Thesis_Data\In_situ_all_climate.csv')
climate_list = df_all['climate'].drop_duplicates().sort_values()

#  Draw the figure of in-situ distribution.
figs, axs = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
for climate_name in climate_list:
    if 'A' in climate_name or 'E' in climate_name:
        print(climate_name)
        ax0 = sns.distplot(df_all[df_all['climate'] == climate_name]['soil moisture'],
                           kde_kws={"label": climate_name}, ax=axs[0, 0])
    if 'B' in climate_name:
        ax1 = sns.distplot(df_all[df_all['climate'] == climate_name]['soil moisture'],
                           kde_kws={"label": climate_name}, ax=axs[0, 1])
    if 'C' in climate_name:
        ax2 = sns.distplot(df_all[df_all['climate'] == climate_name]['soil moisture'],
                           kde_kws={"label": climate_name}, ax=axs[1, 0])
    if 'D'in climate_name:
        ax3 = sns.distplot(df_all[df_all['climate'] == climate_name]['soil moisture'],
                           kde_kws={"label": climate_name}, ax=axs[1, 1])
ax0.set(xlabel=r"Soil moisture ($cm^3/cm^3$)", ylabel="Percent (%)", title="a. Tropical and polar climate group")
ax0.set_xlim(0, 0.6)

ax1.set(xlabel=r"Soil moisture ($cm^3/cm^3$)", ylabel="Percent (%)", title="b. Dry climate group")
ax1.set_xlim(0, 0.6)

ax2.set(xlabel=r"Soil moisture ($cm^3/cm^3$)", ylabel="Percent (%)", title="c. Temperate climate group")
ax2.set_xlim(0, 0.6)

ax3.set(xlabel=r"Soil moisture ($cm^3/cm^3$)", ylabel="Percent (%)", title="d. Continental climate group")
ax3.set_xlim(0, 0.6)