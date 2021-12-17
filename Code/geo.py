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

point = Point(utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LAT,
              utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LONG)

n_points = 20
d = 1000
angles = np.linspace(0,360,n_points)
polygon = geog.propagate(point,angles,d)

#import matplotlib.pyplot as plt
import folium

# Get polygon as list of coords
def coord_lister(geom):
    coords = np.array(geom.exterior.coords)
    return (coords)

derbs = (utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].geometry.
         apply(coord_lister).iloc[0])

derbs_poly = Polygon([(d[0],d[1]) for d in derbs])

derbs_geoj = folium.GeoJson(data=derbs_poly,
                       style_function=lambda x: {'fillColor': 'orange'})

# Adding circle_poly like this works but is elliptical
#circle_poly = Polygon([(p[1],p[0]) for p in polygon])

# This works!
circle_array = []
for b in range(0,360):
    coords = ((d_obj_v.destination(point = geopy.Point(point.x, point.y), bearing = b).latitude),
              (d_obj_v.destination(point = geopy.Point(point.x, point.y), bearing = b).longitude))
    circle_array.append(coords)
    
circle_poly = Polygon([(c[1],c[0]) for c in circle_array])

circle_geoj = folium.GeoJson(data=circle_poly,
                       style_function=lambda x: {'fillColor': 'blue'})
# Adding circle like this looks better but not sure if I can use it to calculate overlap area
#circle = folium.Circle(location=[utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LAT, 
#                         utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LONG], 
#    popup='Point 1A', fill_color='#000', radius=2000, weight=2, color="#000")
#circle.add_to(m)

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

# Circles appear elliptical on map and a manual check using https://www.gps-coordinates.net/distance confirms this.

# Try solution from https://gis.stackexchange.com/questions/268250/generating-polygon-representing-rough-100km-circle-around-latitude-longitude-poi/268277 to get proper circles
import pyproj
from functools import partial
from shapely.ops import transform

local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={point.y} +lon_0={point.x}"

wgs84_to_aeqd = partial(
    pyproj.transform,
    pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
    pyproj.Proj(local_azimuthal_projection),
)

aeqd_to_wgs84 = partial(
    pyproj.transform,
    pyproj.Proj(local_azimuthal_projection),
    pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
)

point_transformed = transform(wgs84_to_aeqd, point)

buffer = point_transformed.buffer(1000)

def flip(x,y):
    return y,x

buffer_wgs84 = transform(flip,transform(aeqd_to_wgs84, buffer))

circle_geoj = folium.GeoJson(data=buffer_wgs84,
                       style_function=lambda x: {'fillColor': 'blue'})
m = folium.Map(location=[utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LAT,
               utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LONG],
               tiles = 'CartoDB positron')
derbs_geoj.add_to(m)
circle_geoj.add_to(m)
m.save("mymap.html")

print(json.dumps(mapping(buffer_wgs84)))
# Still elliptical

# https://stackoverflow.com/questions/24427828/calculate-point-based-on-distance-and-direction
import geopy
import geopy.distance

d_obj_v = geopy.distance.distance(kilometers=1)
d_obj_c = geopy.distance.great_circle(kilometers=1)

d_obj_v.destination(point = geopy.Point(point.x, point.y), bearing = 0)
d_obj_v.destination(point = geopy.Point(point.x, point.y), bearing = 45)
d_obj_v.destination(point = geopy.Point(point.x, point.y), bearing = 90)
# looks like this works

circle_array = []
for b in [0,45,90,135,180,225,270,315]:
    coords = ((d_obj_v.destination(point = geopy.Point(point.x, point.y), bearing = b).latitude),
              (d_obj_v.destination(point = geopy.Point(point.x, point.y), bearing = b).longitude))
    print(coords)
    circle_array.append(coords)
    print(circle_array)
    
circle_poly = Polygon([(c[1],c[0]) for c in circle_array])

(d_obj_v.destination(point = geopy.Point(point.x, point.y), bearing = 0).latitude),(d_obj_v.destination(point = geopy.Point(point.x, point.y), bearing = 0).longitude)

d_obj_v.destination(point = geopy.Point(0, 0), bearing = 0)
d_obj_v.destination(point = geopy.Point(0, 0), bearing = 45)
d_obj_v.destination(point = geopy.Point(0, 0), bearing = 90)

# Try solution from https://gis.stackexchange.com/questions/289044/creating-buffer-circle-x-kilometers-from-point-using-python
proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')

def geodesic_point_buffer(lat, lon, km):
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    return transform(project, buf).exterior.coords[:]

# Example
b = geodesic_point_buffer(point.y, point.x, 10)

Polygon([(b[0],b[1]) for b in b])

# Investigate outside of function
lat = point.y
lon = point.x
# Azimuthal equidistant projection
aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
project = partial(
    pyproj.transform,
    pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
    proj_wgs84)
buf = Point(0, 0).buffer(1000)
transform(project,buf).exterior.coords[:]

# Try example from stack
def geodesic_point_buffer(lat, lon, km):
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    return transform(project, buf).exterior.coords[:]

# Example
b = geodesic_point_buffer(point.y, point.x, 100.0)
print(b)

# What about this: https://stochasticcoder.com/2016/04/06/python-custom-distance-radius-with-basemap/

from osgeo import ogr, osr
import math

def createCircleAroundWithRadius(lat, lon, radiusMiles):
 ring = ogr.Geometry(ogr.wkbLinearRing)
 latArray = []
 lonArray = []
 
 #for brng in range(0,360):
 for brng in [0,90,180,270,360]:
   lat2, lon2 = getLocation(lat,lon,brng,radiusMiles)
   latArray.append(lat2)
   lonArray.append(lon2)

 return lonArray,latArray

def getLocation(lat1, lon1, brng, distanceMiles):
 lat1 = lat1 * math.pi/ 180.0
 lon1 = lon1 * math.pi / 180.0
 #earth radius
 #R = 6378.1Km
 #R = ~ 3959 MilesR = 3959
 R = 3959

 distanceMiles = distanceMiles/R

 brng = (brng / 90)* math.pi / 2

 lat2 = math.asin(math.sin(lat1) * math.cos(distanceMiles) 
   + math.cos(lat1) * math.sin(distanceMiles) * math.cos(brng))

 lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(distanceMiles)
   * math.cos(lat1),math.cos(distanceMiles)-math.sin(lat1)*math.sin(lat2))

 lon2 = 180.0 * lon2/ math.pi
 lat2 = 180.0 * lat2/ math.pi

 return lat2, lon2

X,Y = createCircleAroundWithRadius(point.y,point.x,10)


