# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:31:02 2020

@author: Menglin Wang
"""

import numpy as np
import netCDF4 as nc

file_path = r'C:\Users\96349\Documents\MERRA2\G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
file_obj = nc.Dataset(file_path)
file_obj.variables.keys()
file_obj.variables["BCEXTTAU"]

# nc file has three basic parts: metadata, dimensions and variables
# 1. Metadata
print(file_obj)

# 2. Dimensions
for dim in file_obj.dimensions.values():
    print(dim)
    
# 3. Variable Metadata
for var in file_obj.variables.values():
    print(var)
file_obj.variables.keys()

# 4. Access Data Values
bc_AOT = file_obj['BCEXTTAU'][:]
bc_AOT.shape
one_day_bc_AOT = bc_AOT[0,:,:]
one_day_bc_AOT.shape
type(one_day_bc_AOT)
# every variable is a 365*499*788 masked array 

time_array = file_obj['time'][:]
time_array.shape
time_array
# time is just index, no real time stap was used

lat_array = file_obj['lat'][:]
lon_array = file_obj['lon'][:]
lat_array.max()
lat_array.min()
lon_array.max()
lon_array.min()
