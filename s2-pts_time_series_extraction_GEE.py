"""
 pts_time_series_extraction_v01
date: 27-Apr-2020; // updated: 12-Jun-2020
 author: L.Zhang
Contact: leojayak@gmail.com
-------------------------------------
Description: This script is used to download the time-series of land surface features from GEE at the point scale.
"""
# packages
import ee
import datetime
import numpy as np


# Define the function to extract the time-series.
def fill_27830(img, ini):
    # type cast
    ini_ft = ee.FeatureCollection(ini)
    # Gets the values for the points in the current img.
    ft2 = img.reduceRegions(pts, ee.Reducer.first(), 27830)
    # Gets the date of the img.
    date = img.date().format('YYYY-MM-dd')
    # Writes the date in each feature
    ft3 = ft2.map(lambda f: f.set("date", date))
    # Merges the FeatureCollections
    return ini_ft.merge(ft3)


def fill_1000(img, ini):
    # type cast
    ini_ft = ee.FeatureCollection(ini)
    # Gets the values for the points in the current img.
    ft2 = img.reduceRegions(pts, ee.Reducer.first(), 1000)
    # Gets the date of the img.
    date = img.date().format('YYYY-MM-dd')
    # Writes the date in each feature
    ft3 = ft2.map(lambda f: f.set("date", date))
    # Merges the FeatureCollections
    return ini_ft.merge(ft3)


def fill_2000(img, ini):
    # type cast
    ini_ft = ee.FeatureCollection(ini)
    # Gets the values for the points in the current img.
    ft2 = img.reduceRegions(pts, ee.Reducer.first(), 2000)
    # Gets the date of the img.
    date = img.date().format('YYYY-MM-dd')
    # Writes the date in each feature
    ft3 = ft2.map(lambda f: f.set("date", date))
    # Merges the FeatureCollections
    return ini_ft.merge(ft3)


def fill_4000(img, ini):
    # type cast
    ini_ft = ee.FeatureCollection(ini)
    # Gets the values for the points in the current img.
    ft2 = img.reduceRegions(pts, ee.Reducer.first(), 4000)
    # Gets the date of the img.
    date = img.date().format('YYYY-MM-dd')
    # Writes the date in each feature
    ft3 = ft2.map(lambda f: f.set("date", date))
    # Merges the FeatureCollections
    return ini_ft.merge(ft3)


def fill_8000(img, ini):
    # type cast
    ini_ft = ee.FeatureCollection(ini)
    # Gets the values for the points in the current img.
    ft2 = img.reduceRegions(pts, ee.Reducer.first(), 8000)
    # Gets the date of the img.
    date = img.date().format('YYYY-MM-dd')
    # Writes the date in each feature
    ft3 = ft2.map(lambda f: f.set("date", date))
    # Merges the FeatureCollections
    return ini_ft.merge(ft3)


def fill_16000(img, ini):
    # type cast
    ini_ft = ee.FeatureCollection(ini)
    # Gets the values for the points in the current img.
    ft2 = img.reduceRegions(pts, ee.Reducer.first(), 16000)
    # Gets the date of the img.
    date = img.date().format('YYYY-MM-dd')
    # Writes the date in each feature
    ft3 = ft2.map(lambda f: f.set("date", date))
    # Merges the FeatureCollections
    return ini_ft.merge(ft3)


# Initialize the Google Earth Engine in Python API.
ee.Initialize()
# Import the table (lon, lat)
pts = ee.FeatureCollection("users/leojayak/ISMN_stations")

# Import the dataset [of ERA5, MOD11A1, MOD13A1]
collections_era5_p = ee.ImageCollection("ECMWF/ERA5/DAILY").select('total_precipitation')  # precipitation
collections_mod11a1 = ee.ImageCollection('MODIS/006/MOD11A1')  # LST
collections_mod13a1 = ee.ImageCollection("MODIS/006/MOD13A1")  # NDVI

for year_0 in np.arange(2000, 2020):
    year_1 = year_0 + 1
    date_0 = datetime.datetime.strftime(datetime.datetime(year_0, 1, 1), '%Y-%m-%d')
    date_1 = datetime.datetime.strftime(datetime.datetime(year_1, 1, 1), '%Y-%m-%d')

    year_p = collections_era5_p.filterDate(date_0, date_1)
    year_lst = collections_mod11a1.filterDate(date_0, date_1)
    year_ndvi = collections_mod13a1.filterDate(date_0, date_1)

    ''' Precipitation '''
    # empty fill
    ft_p = ee.FeatureCollection(ee.List([]))
    # Obtain the data
    newft_p = ee.FeatureCollection(year_p.iterate(fill_27830, ft_p))
    file_name_p = 'ERA5_P_27830_' + str(year_0)
    task_p = ee.batch.Export.table.toDrive(collection=newft_p,
                                           description=file_name_p,
                                           folder='GEE_python',
                                           fileFormat='CSV')
    task_p.start()

    ''' LST '''
    # Select the features needed.
    features_ini_lst = ee.Feature(ee.Geometry.Point([0, 0])) \
        .set('id', 'id') \
        .set('date', 'date') \
        .set('LST_Day_1km', 'LST_Day_1km') \
        .set('QC_Day', 'QC_Day') \
        .set('LST_Night_1km', 'LST_Night_1km') \
        .set('QC_Night', 'QC_Night') \
        .set('network', 'network') \
        .set('station', 'station')

    # 1000m
    ft_lst_1000 = ee.FeatureCollection(ee.List([features_ini_lst]))
    new_ft_lst_1000 = ee.FeatureCollection(year_lst.iterate(fill_1000, ft_lst_1000))
    file_name_lst_1000 = 'MOD11A1_1000_' + str(year_0)
    task_lst_1000 = ee.batch.Export.table.toDrive(collection=new_ft_lst_1000,
                                                  description=file_name_lst_1000,
                                                  folder='GEE_python',
                                                  fileFormat='CSV')
    task_lst_1000.start()

    # 2000m
    ft_lst_2000 = ee.FeatureCollection(ee.List([features_ini_lst]))
    new_ft_lst_2000 = ee.FeatureCollection(year_lst.iterate(fill_2000, ft_lst_2000))
    file_name_lst_2000 = 'MOD11A1_2000_' + str(year_0)
    task_lst_2000 = ee.batch.Export.table.toDrive(collection=new_ft_lst_2000,
                                                  description=file_name_lst_2000,
                                                  folder='GEE_python',
                                                  fileFormat='CSV')
    task_lst_2000.start()

    # 4000m
    ft_lst_4000 = ee.FeatureCollection(ee.List([features_ini_lst]))
    new_ft_lst_4000 = ee.FeatureCollection(year_lst.iterate(fill_4000, ft_lst_4000))
    file_name_lst_4000 = 'MOD11A1_4000_' + str(year_0)
    task_lst_4000 = ee.batch.Export.table.toDrive(collection=new_ft_lst_4000,
                                                  description=file_name_lst_4000,
                                                  folder='GEE_python',
                                                  fileFormat='CSV')
    task_lst_4000.start()


    # 8000m
    ft_lst_8000 = ee.FeatureCollection(ee.List([features_ini_lst]))
    new_ft_lst_8000 = ee.FeatureCollection(year_lst.iterate(fill_8000, ft_lst_8000))
    file_name_lst_8000 = 'MOD11A1_8000_' + str(year_0)
    task_lst_8000 = ee.batch.Export.table.toDrive(collection=new_ft_lst_8000,
                                                  description=file_name_lst_8000,
                                                  folder='GEE_python',
                                                  fileFormat='CSV')
    task_lst_8000.start()

    # 16000m
    ft_lst_16000 = ee.FeatureCollection(ee.List([features_ini_lst]))
    new_ft_lst_16000 = ee.FeatureCollection(year_lst.iterate(fill_16000, ft_lst_16000))
    file_name_lst_16000 = 'MOD11A1_16000_' + str(year_0)
    task_lst_16000 = ee.batch.Export.table.toDrive(collection=new_ft_lst_16000,
                                                   description=file_name_lst_16000,
                                                   folder='GEE_python',
                                                   fileFormat='CSV')
    task_lst_16000.start()

    # 27830m
    ft_lst_27830 = ee.FeatureCollection(ee.List([features_ini_lst]))
    new_ft_lst_27830 = ee.FeatureCollection(year_lst.iterate(fill_27830, ft_lst_27830))
    file_name_lst_27830 = 'MOD11A1_27830_' + str(year_0)
    task_lst_27830 = ee.batch.Export.table.toDrive(collection=new_ft_lst_27830,
                                                   description=file_name_lst_27830,
                                                   folder='GEE_python',
                                                   fileFormat='CSV')
    task_lst_27830.start()

    '''  NDVI '''
    # Select the features needed.
    features_ini_ndvi = ee.Feature(ee.Geometry.Point([0, 0])) \
        .set('id', 'id') \
        .set('date', 'date') \
        .set('NDVI', 'NDVI') \
        .set('EVI', 'EVI') \
        .set('network', 'network') \
        .set('station', 'station')

    # 1000m
    ft_ndvi_1000 = ee.FeatureCollection(ee.List([features_ini_ndvi]))
    new_ft_ndvi_1000 = ee.FeatureCollection(year_ndvi.iterate(fill_1000, ft_ndvi_1000))
    file_name_ndvi_1000 = 'MOD13A1_1000_' + str(year_0)
    task_ndvi_1000 = ee.batch.Export.table.toDrive(collection=new_ft_ndvi_1000,
                                                   description=file_name_ndvi_1000,
                                                   folder='GEE_python',
                                                   fileFormat='CSV')
    task_ndvi_1000.start()

    # 2000m
    ft_ndvi_2000 = ee.FeatureCollection(ee.List([features_ini_ndvi]))
    new_ft_ndvi_2000 = ee.FeatureCollection(year_ndvi.iterate(fill_2000, ft_ndvi_2000))
    file_name_ndvi_2000 = 'MOD13A1_2000_' + str(year_0)
    task_ndvi_2000 = ee.batch.Export.table.toDrive(collection=new_ft_ndvi_2000,
                                                   description=file_name_ndvi_2000,
                                                   folder='GEE_python',
                                                   fileFormat='CSV')
    task_ndvi_2000.start()

    # 4000m
    ft_ndvi_4000 = ee.FeatureCollection(ee.List([features_ini_ndvi]))
    new_ft_ndvi_4000 = ee.FeatureCollection(year_ndvi.iterate(fill_4000, ft_ndvi_4000))
    file_name_ndvi_4000 = 'MOD13A1_4000_' + str(year_0)
    task_ndvi_4000 = ee.batch.Export.table.toDrive(collection=new_ft_ndvi_4000,
                                                   description=file_name_ndvi_4000,
                                                   folder='GEE_python',
                                                   fileFormat='CSV')
    task_ndvi_4000.start()

    # 8000m
    ft_ndvi_8000 = ee.FeatureCollection(ee.List([features_ini_ndvi]))
    new_ft_ndvi_8000 = ee.FeatureCollection(year_ndvi.iterate(fill_8000, ft_ndvi_8000))
    file_name_ndvi_8000 = 'MOD13A1_8000_' + str(year_0)
    task_ndvi_8000 = ee.batch.Export.table.toDrive(collection=new_ft_ndvi_8000,
                                                   description=file_name_ndvi_8000,
                                                   folder='GEE_python',
                                                   fileFormat='CSV')
    task_ndvi_8000.start()

    # 16000m
    ft_ndvi_16000 = ee.FeatureCollection(ee.List([features_ini_ndvi]))
    new_ft_ndvi_16000 = ee.FeatureCollection(year_ndvi.iterate(fill_16000, ft_ndvi_16000))
    file_name_ndvi_16000 = 'MOD13A1_16000_' + str(year_0)
    task_ndvi_16000 = ee.batch.Export.table.toDrive(collection=new_ft_ndvi_16000,
                                                    description=file_name_ndvi_16000,
                                                    folder='GEE_python',
                                                    fileFormat='CSV')
    task_ndvi_16000.start()

    # 27830m
    ft_ndvi_27830 = ee.FeatureCollection(ee.List([features_ini_ndvi]))
    new_ft_ndvi_27830 = ee.FeatureCollection(year_ndvi.iterate(fill_27830, ft_ndvi_27830))
    file_name_ndvi_27830 = 'MOD13A1_27830_' + str(year_0)
    task_ndvi_27830 = ee.batch.Export.table.toDrive(collection=new_ft_ndvi_27830,
                                                    description=file_name_ndvi_27830,
                                                    folder='GEE_python',
                                                    fileFormat='CSV')
    task_ndvi_27830.start()
