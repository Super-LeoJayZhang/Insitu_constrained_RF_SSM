"""
date: 05-Jul-2021
Contact: leojayak@gmail.com
-------------------------------------
Description: This script shows the method to read the surface soil moisture data from the netCDF4 file.
"""
# libraries
import netCDF4 as nc
# !!!please define the NC_file_position before running the script.
Data = nc.Dataset(NC_file_position, 'r')
ssm = Data['ssm']
time = Data['time']
lon = Data['lon']
lat = Data['lat']