# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 19:30:55 2021

@author: Joe.WozniczkaWells
"""

# GEOCODING
## Get UTLA polygons from https://geoportal.statistics.gov.uk/

import geopandas as gpd

utla_polygons = gpd.read_file('https://opendata.arcgis.com/datasets/69109c4fbbc54f1f9d6e18000031a5fd_0.geojson')
print(utla_polygons.dtypes)

# utla_polygons contains the following:
# OBJECTID
# CTYUA21CD: area code
# CTYUA21NM: area name
# CTYUA21NMW: area name - Welsh (only applicable to Welsh areas)
# BNG_E: X coordinate of centroid
# BNG_N: Y coordinate of centroid
# LONG: longitude of centroid
# LAT: latitude of centroid
# Shape__Area: area of polygon (not sure what the unit is)
# Shape__Length: length of polygon (not sure exactly which length this is referring to or what the unit is)

# Try drawing a circle of radius 10km around a point
# Use Derbyshire centroid to do this

import numpy as np
import json
import geog
from shapely.geometry import Point, mapping, Polygon, asPolygon
import shapely
import pandas as pd
import folium
import geopy
import geopy.distance

point = Point(utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LONG,
              utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LAT)

# Get polygon as list of coords
def coord_lister(geom):
    coords = np.array(geom.exterior.coords)
    return (coords)

derbs = (utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].geometry.
         apply(coord_lister).iloc[0])

derbs_poly = Polygon([(d[0],d[1]) for d in derbs])

derbs_geoj = folium.GeoJson(data=derbs_poly,
                       style_function=lambda x: {'fillColor': 'orange'})

# Draw approximation of a circle using geopy.destination
d_obj_v = geopy.distance.distance(kilometers=15)

circle_array = []
for b in range(0,360):
    coords = ((d_obj_v.destination(point = geopy.Point(point.y, point.x), bearing = b).latitude),
              (d_obj_v.destination(point = geopy.Point(point.y, point.x), bearing = b).longitude))
    circle_array.append(coords)
    
circle_poly = Polygon([(c[1],c[0]) for c in circle_array])

circle_geoj = folium.GeoJson(data=circle_poly,
                       style_function=lambda x: {'fillColor': 'blue'})

# Try adding Derbyshire polygon to map
m = folium.Map(location=[utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LAT,
               utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LONG],
               tiles = 'CartoDB positron')
derbs_geoj.add_to(m)
circle_geoj.add_to(m)
m.save("mymap.html")

# Calculate intersection %
# Percent of Derbyshire covered by circle
derbys_perc = circle_poly.intersection(derbs_poly).area/derbs_poly.area*100

# Percent of circle that is in Derbyshire
circle_perc = circle_poly.intersection(derbs_poly).area/circle_poly.area*100














