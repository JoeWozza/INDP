# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 19:30:55 2021

@author: Joe.WozniczkaWells
"""

# GEOCODING: approximating UTLAs of the Midlands using circles

from os import chdir, getcwd

# Set file path
filepath = ("C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\"
            "SPFINDP21T4\\")
chdir(filepath)

from INDP.Code import CircleApprox
class_ca = CircleApprox.CircleApprox()

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
import pandas as pd
import folium
import geopy
import geopy.distance
from random import randrange
import random
import requests

# Download polygons from https://geoportal.statistics.gov.uk/
geojson_url = ("https://opendata.arcgis.com/datasets/"
               "244b257482da4778995cf11ff99e9997_0.geojson")
res = requests.get(geojson_url)
utla_polygons = gpd.GeoDataFrame.from_features(res.json()).set_crs("epsg:4326")

# Set seed for random numbers
random.seed(123)

# List of Midlands UTLAs
mids_utlas = ['Derby','Leicester','Rutland','Nottingham',
              'Herefordshire, County of','Telford and Wrekin','Stoke-on-Trent',
              'Shropshire','North Northamptonshire','West Northamptonshire',
              'Birmingham','Coventry','Dudley','Sandwell','Solihull','Walsall',
              'Wolverhampton','Derbyshire','Leicestershire','Lincolnshire',
              'Nottinghamshire','Staffordshire','Warwickshire','Worcestershire'
              ]
mids_utlas = ['Derby','Nottingham']
#mids_utlas = ['Derby']
#mids_utlas = ['Derbyshire','Derby']

# Min area of target UTLA that must be covered by circle-based approximation
#min_utla_perc_tot = 90
min_utla_perc_tot = 80
# Min amount of circle-based approximation that must be in target UTLA
min_circle_perc_tot = 95

#%%
# Loop through Midlands UTLAs
df_utlas = class_ca.areas_circles(mids_utlas,1,utla_polygons,"CTYUA21NM","geometry",
                         min_utla_perc_tot,min_circle_perc_tot)

# Save df_utlas as csv (temporary solution while still putting together the
# other code, so I don't have to run this every time I want to work on
# subsequent code)
df_utlas.to_csv("df_utlas_{0}.csv".format(min_utla_perc_tot))