"""
Thesis_MSc_Lijie pts_static_extraction_v01
date: 19-Jun-2020
author: leojay
Contact: leojayak@gmail.com
-------------------------------------
Description: Download the static features
"""
# libraries
import ee

# Initialize the Google Earth Engine in Python API.
ee.Initialize()
# Import the table (lon, lat)
pts = ee.FeatureCollection("users/leojayak/ISMN_stations")

gee_sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b0')
gee_clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0')
gee_bd = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select('b0')


def Obtain_pts(img, pts, spatial_resolution):
    return img.reduceRegions(pts, ee.Reducer.first(), spatial_resolution)


def Export_pts(feature, name):
    task = ee.batch.Export.table.toDrive(collection=feature,
                                         description=name,
                                         folder='GEE_python',
                                         fileFormat='CSV')
    task.start()


list_spatial_resolution = [1000, 2000, 4000, 8000, 16000, 27830]

for spatial_i in list_spatial_resolution:
    pts_sand = Obtain_pts(gee_sand, pts, spatial_i)
    pts_clay = Obtain_pts(gee_clay, pts, spatial_i)
    pts_bd = Obtain_pts(gee_bd, pts, spatial_i)
    # Export the pts into the google drive.
    Export_pts(pts_sand, 'sand_p_' + str(spatial_i))
    Export_pts(pts_clay, 'clay_p_' + str(spatial_i))
    Export_pts(pts_bd, 'bd_p_' + str(spatial_i))





