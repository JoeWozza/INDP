# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 19:30:55 2021

@author: Joe.WozniczkaWells
"""

# GEOCODING: approximating UTLAs of the Midlands using circles

from os import chdir, getcwd

# Set file path
filepath = ("C:\\Users\\Joe.WozniczkaWells\\Documents\\Apprenticeship\\UoB\\"
            "SPFINDP21T4")
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
import os

# Create folders in which to save maps and circle details
geo_folder = '{0}/INDP/Geo'.format(filepath)
map_folder = '{0}/INDP/Geo/Maps'.format(filepath)

if not os.path.exists(geo_folder):
    os.makedirs('{0}/INDP/Geo'.format(filepath))
if not os.path.exists(map_folder):
    os.makedirs('{0}/INDP/Geo/Maps'.format(filepath))
    
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
#mids_utlas = ['Derby','Nottingham']
#mids_utlas = ['Derby']
#mids_utlas = ['Derbyshire','Derby']

# Min area of target UTLA that must be covered by circle-based approximation
min_utla_perc_tot = 90
# Min amount of circle-based approximation that must be in target UTLA
min_circle_perc_tot = 95

#%%
# Loop through Midlands UTLAs
df_utlas = class_ca.areas_circles(mids_utlas,1,utla_polygons,"CTYUA21NM","geometry",
                         min_utla_perc_tot,min_circle_perc_tot,map_folder)

# Save circle details in csv
df_utlas.to_csv("{0}/df_utlas_{1}_{2}.csv".format(geo_folder,min_utla_perc_tot,
                min_circle_perc_tot))

#%% Circle-based approximation of England

# Use England polygon
geojson_url = ("https://opendata.arcgis.com/datasets/"
               "cf156624007344f2a4a067fe7711c0ee_0.geojson")
res = requests.get(geojson_url)
ctry_polygons = gpd.GeoDataFrame.from_features(res.json()).set_crs("epsg:4326")

# Combine all UTLAs to get UK polygon
df = ctry_polygons
var = 'CTRY20NM'

gdfd = ctry_polygons.loc[ctry_polygons['CTRY20NM']=='England'].copy()
    
eng_poly = class_ca.extract_area_poly(gdfd['geometry'],gdfd)

# Approximate UK with circles
lat = gdfd.LAT
long = gdfd.LONG
area = 'England'
df_area = pd.DataFrame(columns = ['area','lat','long','radius'])
radius_increment = 10
min_area_perc_tot = 90
n_bad = 0

class_ca.map_poly(eng_poly,lat,long,area)

initial_poly, df_area, area_perc_tot, circle_perc_tot = (class_ca.
                                                         initial_circle(
        lat,long,eng_poly,area,df_area,radius_increment,min_circle_perc_tot)
                                                         )

df_area, all_poly = class_ca.fill_area_with_circles(area_perc_tot,
                                                    circle_perc_tot,
                                                    min_area_perc_tot,
                                                    radius_increment,n_bad,50,
                                                    initial_poly,eng_poly,
                                                    area,df_area,
                                                    min_circle_perc_tot)

class_ca.map_poly(all_poly,lat,long,area)
df_area.to_csv("{0}/df_eng_{1}_{2}.csv".format(geo_folder,min_area_perc_tot,
               min_circle_perc_tot))
