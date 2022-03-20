# -*- coding: utf-8 -*-
"""
This script produces visualisations of the circle-based approximation
algorithm, to demonstrate how the algorithm works in the project report.

@author: Joe Wozniczka-Wells
"""

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
map_folder = '{0}/INDP/Geo/Demo'.format(filepath)

if not os.path.exists(geo_folder):
    os.makedirs(geo_folder)
if not os.path.exists(map_folder):
    os.makedirs(map_folder)
    
# Download polygons from https://geoportal.statistics.gov.uk/
geojson_url = ("https://opendata.arcgis.com/datasets/"
               "244b257482da4778995cf11ff99e9997_0.geojson")
res = requests.get(geojson_url)
utla_polygons = gpd.GeoDataFrame.from_features(res.json()).set_crs("epsg:4326")

# Set seed for random numbers
random.seed(123)

# Min area of target UTLA that must be covered by circle-based approximation
min_utla_perc_tot = 90
# Min amount of circle-based approximation that must be in target UTLA
min_circle_perc_tot = 95

#%%

area_name = 'Nottinghamshire'

# Get lat/long of centre of Nottinghamshire
lat = utla_polygons[utla_polygons['CTYUA21NM']==area_name].LAT
long = utla_polygons[utla_polygons['CTYUA21NM']==area_name].LONG

# Get shapely polygon of Nottinghamshire
notts_poly = class_ca.extract_area_poly(
        utla_polygons[utla_polygons['CTYUA21NM']==area_name]
        ['geometry'])

# Define other variables
radius_increment = 1
df_area = pd.DataFrame(columns = ['area','lat','long','radius','utla_perc_tot',
                                  'circle_perc_tot'])
n_bad = 0
n = 50
i=1

# Nottinghamshire shape for map
notts_geoj = folium.GeoJson(data=notts_poly,
                            style_function=lambda x: {'fillColor': 'green',
                                                      'opacity': 0}
                            )

## Shape 1
# Draw initial circle
shape, df_area, area_perc_tot, circle_perc_tot = (class_ca.initial_circle(
        lat,long,notts_poly,area_name,df_area,radius_increment,
        min_circle_perc_tot))
df_area.iat[len(df_area)-1,4] = area_perc_tot
df_area.iat[len(df_area)-1,5] = circle_perc_tot

# Map initial circle and save
all_geoj = folium.GeoJson(data=shape,
                          style_function=lambda x: {'fillColor': 'blue',
                                                    'fillOpacity': 0.02})

c = folium.Map(location=[lat,long],tiles = 'CartoDB positron',zoom_start=9)
all_geoj.add_to(c)
notts_geoj.add_to(c)
c.save("{0}/{1}_circles_demo_{2}.html".format(map_folder,area_name,i))

i=2

while area_perc_tot < min_utla_perc_tot:
    
    # Every 50 consecutive unsuccessful attempts to draw a circle, half the
    # radius_increment
    radius_increment = class_ca.dec_radius_increment(radius_increment,n_bad,n)
    
    # Select a random point on the polygon
    rand_point = class_ca.poly_random_point(shape,notts_poly,area_name)
    
    #	Keep drawing circles
    circle_area_area_tot = class_ca.circle_intersection_area(shape,notts_poly)
    
    circle_perc_tot_work, area_perc_tot_work = class_ca.calc_area_perc(
            shape,notts_poly)
    
    radius = 0
    
    shape, area_perc_tot, df_area, n_bad = class_ca.draw_save_circle(
            min_circle_perc_tot,circle_perc_tot,radius,radius_increment,
            rand_point,notts_poly,shape,n_bad,area_name,df_area)
    
    if n_bad == 0:
        df_area.iat[len(df_area)-1,4] = area_perc_tot
        df_area.iat[len(df_area)-1,5] = (
                class_ca.circle_intersection_area(shape,notts_poly)/
                shape.area*100)
        
        # Map circle-based approximation and save
        all_geoj = folium.GeoJson(data=shape,
                                  style_function=lambda x: {
                                          'fillColor': 'blue',
                                          'fillOpacity': 0.02})
        
        c = folium.Map(location=[lat,long],tiles = 'CartoDB positron',
                       zoom_start=9)
        all_geoj.add_to(c)
        notts_geoj.add_to(c)
        c.save("{0}/{1}_circles_demo_{2}.html".format(map_folder,area_name,i))
        
        i+=1
