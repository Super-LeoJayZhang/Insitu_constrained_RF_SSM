"""
Thesis_MSc_Lijie Gridded_data_download_global_stripes
date: 03-Aug-2020
author: leojay
Contact: leojayak@gmail.com
-------------------------------------
Description: 
"""
# libraries
import ee
import numpy as np

ee.Initialize()

# Date range
date_start = '2000-02-24'
# date_end = '2000-03-12'
date_end = '2019-12-31'


def Download_land_surface_features(polygon, idx):
    mod11a1 = ee.ImageCollection("MODIS/006/MOD11A1")
    mod13a1 = ee.ImageCollection("MODIS/006/MOD13A1")
    era5 = ee.ImageCollection("ECMWF/ERA5/DAILY")

    # Land Surface Temperature.
    data_lst_day = mod11a1.filterDate(date_start, date_end).select(['LST_Day_1km']).toBands().reproject('EPSG:4326',
                                                                                                        None, 27830)
    data_lst_night = mod11a1.filterDate(date_start, date_end).select(['LST_Night_1km']).toBands().reproject('EPSG:4326',
                                                                                                            None, 27830)
    # NDVI & EVI.
    data_ndvi = mod13a1.filterDate(date_start, date_end).select(['NDVI']).toBands().reproject('EPSG:4326', None, 27830)
    data_evi = mod13a1.filterDate(date_start, date_end).select(['EVI']).toBands().reproject('EPSG:4326', None, 27830)
    # Precipitation.
    data_p = era5.filterDate(date_start, date_end).select(['total_precipitation']).toBands().reproject('EPSG:4326',
                                                                                                       None, 27830)
    # Soil Texture.
    data_sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").reproject('EPSG:4326', None, 27830)
    data_clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").reproject('EPSG:4326', None, 27830)
    list = [data_p, data_lst_day, data_lst_night, data_ndvi, data_evi, data_sand, data_clay]
    list_name = ['p_Global'+str(idx), 'LST_day_Global'+str(idx), 'LST_night_Global'+str(idx), 'NDVI_Global'+str(idx),
                 'EVI_Global'+str(idx), 'sand_Global'+str(idx), 'clay_Global'+str(idx)]

    for count_j in np.arange(len(list)):
        task = ee.batch.Export.image.toDrive(image=list[count_j], description=list_name[count_j], region=polygon,
                                             folder=r'GEE_gridded_python_global_stripes', scale=27830)
        task.start()


# A text boundary box.
for idx, lon in enumerate(np.arange(-180, 180, 10)):
    print(idx, lon)
    polygon = ee.Geometry.Rectangle([int(lon), -58, int(lon+10), 70], 'EPSG:4326', False)
    Download_land_surface_features(polygon, idx)


# #
# polygon = ee.Geometry.Rectangle([int(lon), -58, int(lon+10), 70], 'EPSG:4326', False)
# task = ee.batch.Export.image.toDrive(image=data_p, description='test_p', region=polygon,
#                                      folder=r'GEE_gridded_python_global_stripes', scale=27830)
# task.start()

