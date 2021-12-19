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
from shapely.ops import cascaded_union
import shapely
import pandas as pd
import folium
import geopy
import geopy.distance
from random import randrange

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

#%% Try to cover Derbyshire in circles
### S1

# Create empty dataframe, to contain centroids and radiuses of circles
df_derbs = pd.DataFrame(columns = ['lat','long','radius'])

# Get Derbyshire polygon as list of coords
def coord_lister(geom):
    coords = np.array(geom.exterior.coords)
    return (coords)

derbs = (utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].geometry.
         apply(coord_lister).iloc[0])

derbs_poly = Polygon([(d[0],d[1]) for d in derbs])

derbs_geoj = folium.GeoJson(data=derbs_poly,
                       style_function=lambda x: {'fillColor': 'orange'})

m = folium.Map(location=[utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LAT,
               utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LONG],
               tiles = 'CartoDB positron')
derbs_geoj.add_to(m)

def draw_circle(point,radius):
    # Draw approximation of a circle using geopy.destination
    d_obj_v = geopy.distance.distance(kilometers=radius)
    
    circle_array = []
    for b in range(0,360):
        coords = ((d_obj_v.destination(point = geopy.Point(point.y, point.x), bearing = b).latitude),
                  (d_obj_v.destination(point = geopy.Point(point.y, point.x), bearing = b).longitude))
        circle_array.append(coords)
        
    circle_poly = Polygon([(c[1],c[0]) for c in circle_array])
    
    # Area of circle that is in Derbyshire
    circle_derbs_area = circle_poly.intersection(derbs_poly).area
    
    return circle_derbs_area, circle_poly

# 1. Check centroid is in target UTLA (may be that in some cases it is in 
# another UTLA that is entirely within the target UTLA or that an irregular 
# (e.g. concave) shape means it is in a different UTLA)
derbs_cent = Point(utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LONG,
                   utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LAT)

if derbs_cent.within(derbs_poly):
    #	2. Try to get the biggest circle possible, where the centre is the UTLA 
    # centroid, that contains no more than X(start with 5)% of other UTLAs: 
    # start with 1km radius. If circle_perc >= 95, add 1km to radius until 
    # circle_perc < 95, then use circle with radius = radius - 1.
    radius = 1
    circle_poly = draw_circle(derbs_cent,radius)[1]
    derbs_perc = draw_circle(derbs_cent,radius)[0]/derbs_poly.area*100
    circle_perc = draw_circle(derbs_cent,radius)[0]/circle_poly.area*100
    
    while (circle_perc >= 95) & (derbs_perc < 98):
        radius = radius + 1
        circle_poly = draw_circle(derbs_cent,radius)[1]
        derbs_perc = draw_circle(derbs_cent,radius)[0]/derbs_poly.area*100
        circle_perc = draw_circle(derbs_cent,radius)[0]/circle_poly.area*100
    
    circle_poly_1 = draw_circle(derbs_cent,radius-1)[1]
    circle_derbs_area_tot = draw_circle(derbs_cent,radius-1)[0]
        
    circle_geoj_1 = folium.GeoJson(data=circle_poly_1,
                       style_function=lambda x: {'fillColor': 'blue'})
    
    dict_derbs = {'lat': derbs_cent.y, 'long': derbs_cent.x, 'radius': radius-1}
    df_derbs = df_derbs.append(dict_derbs, ignore_index = True)
    
    #circle_geoj_1.add_to(m)
#m.save("mymap.html")
#	3. Take a random point from the circumference of the first circle and check 
# it is in the target UTLA.
rand_no = randrange(361)    

rand_point = Point(list(circle_poly_1.exterior.coords)[rand_no][0],
                   list(circle_poly_1.exterior.coords)[rand_no][1])

#		a. If it is not, choose another random point from the circ. of the first 
#   circle and check it is in the target UTLA. Proceed to step 4 when a point 
#   on the circ. of the first circle that is in the target UTLA has been 
#   identified.
while rand_point.within(derbs_poly) == False:
    print("Find a new random point that is in Derbyshire")
    rand_no = randrange(361)
    rand_point = Point(list(circle_poly_1.exterior.coords)[rand_no][0],
                       list(circle_poly_1.exterior.coords)[rand_no][1])

#	4. Repeat step 2, using the point identified in step 3. Circle_perc should 
# be updated to be the area covered by any of the circles.
radius = 1

circle_perc_tot = circle_derbs_area_tot/circle_poly_1.area*100
derbs_perc_tot = circle_derbs_area_tot/derbs_poly.area*100

while (circle_perc_tot >= 95) & (derbs_perc_tot < 98):
    radius = radius + 1
    circle_derbs_area, circle_poly = draw_circle(rand_point,radius)
    derbs_perc_tot = ((circle_derbs_area_tot + circle_derbs_area - 
                       circle_poly_1.intersection(circle_poly)
                       .intersection(derbs_poly).area)/
        derbs_poly.area*100)
    circle_perc_tot = ((circle_derbs_area_tot + circle_derbs_area - 
                        circle_poly_1.intersection(circle_poly)
                       .intersection(derbs_poly).area)/
        (circle_poly_1.area + circle_poly.area - 
         circle_poly_1.intersection(circle_poly).area)*100)
    print(radius)
    print(derbs_perc_tot)
    print(circle_perc_tot)
    dict_derbs = {'lat': rand_point.y, 'long': rand_point.x, 'radius': radius-1}
    df_derbs = df_derbs.append(dict_derbs, ignore_index = True)

circle_derbs_area = draw_circle(rand_point,radius-1)[0]
circle_poly_2 = draw_circle(rand_point,radius-1)[1]

circles_comb_poly = cascaded_union([circle_poly_1,circle_poly_2])
    
circles_comb_geoj = folium.GeoJson(data=circles_comb_poly,
                   style_function=lambda x: {'fillColor': 'blue'})

circles_comb_geoj.add_to(m)
m.save("mymap.html")

circle_derbs_area_tot = (circles_comb_poly.intersection(derbs_poly).area)

#	5. Repeat step 3, using points from the circle drawn in step 4 rather than the original circle.
#	6. Repeat steps 4 and 5 until a circle with radius >= 1km cannot be drawn so circle_perc >= 95
#	7. Go back to step 3 and continue until utla_perc >= 98




#%% Try to put it all into one step
# Create empty dataframe, to contain centroids and radiuses of circles
df_derbs = pd.DataFrame(columns = ['lat','long','radius'])

# Get Derbyshire polygon as list of coords
def coord_lister(geom):
    coords = np.array(geom.exterior.coords)
    return (coords)

derbs = (utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].geometry.
         apply(coord_lister).iloc[0])

derbs_poly = Polygon([(d[0],d[1]) for d in derbs])

derbs_geoj = folium.GeoJson(data=derbs_poly,
                       style_function=lambda x: {'fillColor': 'orange'})

m = folium.Map(location=[utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LAT,
               utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LONG],
               tiles = 'CartoDB positron')
derbs_geoj.add_to(m)

def draw_circle(point,radius):
    # Draw approximation of a circle using geopy.destination
    d_obj_v = geopy.distance.distance(kilometers=radius)
    
    circle_array = []
    for b in range(0,360):
        coords = ((d_obj_v.destination(point = geopy.Point(point.y, point.x), bearing = b).latitude),
                  (d_obj_v.destination(point = geopy.Point(point.y, point.x), bearing = b).longitude))
        circle_array.append(coords)
        
    circle_poly = Polygon([(c[1],c[0]) for c in circle_array])
    
    # Area of circle that is in Derbyshire
    circle_derbs_area = circle_poly.intersection(derbs_poly).area
    
    return circle_derbs_area, circle_poly

# 1. Check centroid is in target UTLA (may be that in some cases it is in 
# another UTLA that is entirely within the target UTLA or that an irregular 
# (e.g. concave) shape means it is in a different UTLA)
derbs_cent = Point(utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LONG,
                   utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire'].LAT)

if derbs_cent.within(derbs_poly):
    #	2. Try to get the biggest circle possible, where the centre is the UTLA 
    # centroid, that contains no more than X(start with 5)% of other UTLAs: 
    # start with 1km radius. If circle_perc >= 95, add 1km to radius until 
    # circle_perc < 95, then use circle with radius = radius - 1.
    radius = 1
    circle_poly = draw_circle(derbs_cent,radius)[1]
    derbs_perc = draw_circle(derbs_cent,radius)[0]/derbs_poly.area*100
    circle_perc = draw_circle(derbs_cent,radius)[0]/circle_poly.area*100
    
    while circle_perc >= 95:
        radius = radius + 1
        circle_poly = draw_circle(derbs_cent,radius)[1]
        derbs_perc = draw_circle(derbs_cent,radius)[0]/derbs_poly.area*100
        circle_perc = draw_circle(derbs_cent,radius)[0]/circle_poly.area*100
    
    all_poly = draw_circle(derbs_cent,radius-1)[1]
    
    dict_derbs = {'lat': derbs_cent.y, 'long': derbs_cent.x, 'radius': radius-1}
    df_derbs = df_derbs.append(dict_derbs, ignore_index = True)
    
derbs_perc_tot = all_poly.intersection(derbs_poly).area/derbs_poly.area*100

while derbs_perc_tot < 98:
    #	3. Take a random point from the circumference of the first circle and check 
    # it is in the target UTLA.
    rand_no = randrange(len(all_poly.exterior.coords))
    
    rand_point = Point(list(all_poly.exterior.coords)[rand_no][0],
                       list(all_poly.exterior.coords)[rand_no][1])
    
    #		a. If it is not, choose another random point from the circ. of the first 
    #   circle and check it is in the target UTLA. Proceed to step 4 when a point 
    #   on the circ. of the first circle that is in the target UTLA has been 
    #   identified.
    while rand_point.within(derbs_poly) == False:
        print("Find a new random point that is in Derbyshire")
        rand_no = randrange(len(all_poly.exterior.coords))
        rand_point = Point(list(all_poly.exterior.coords)[rand_no][0],
                           list(all_poly.exterior.coords)[rand_no][1])
    
    #	4. Repeat step 2, using the point identified in step 3. Circle_perc should 
    # be updated to be the area covered by any of the circles.
    circle_derbs_area_tot = all_poly.intersection(derbs_poly).area
    circle_perc_tot_work = circle_derbs_area_tot/all_poly.area*100
    derbs_perc_tot_work = circle_derbs_area_tot/derbs_poly.area*100
    radius = 0
    while circle_perc_tot_work >= 95:
        radius = radius + 1
        circle_derbs_area, circle_poly = draw_circle(rand_point,radius)
        derbs_perc_tot_work = ((circle_derbs_area_tot + circle_derbs_area - 
                           all_poly.intersection(circle_poly)
                           .intersection(derbs_poly).area)/
            derbs_poly.area*100)
        circle_perc_tot_work = ((circle_derbs_area_tot + circle_derbs_area - 
                            all_poly.intersection(circle_poly)
                           .intersection(derbs_poly).area)/
            (all_poly.area + circle_poly.area - 
             all_poly.intersection(circle_poly).area)*100)
        print(radius)
        print(derbs_perc_tot_work)
        print(circle_perc_tot_work)
    
    if radius > 1:
        dict_derbs = {'lat': rand_point.y, 'long': rand_point.x, 'radius': radius-1}
        df_derbs = df_derbs.append(dict_derbs, ignore_index = True)    
        circle_poly = draw_circle(rand_point,radius - 1)[1]      
        all_poly = cascaded_union([all_poly,circle_poly])
        derbs_perc_tot = all_poly.intersection(derbs_poly).area/derbs_poly.area*100

all_geoj = folium.GeoJson(data=all_poly,
                          style_function=lambda x: {'fillColor': 'blue'})
all_geoj.add_to(m)
m.save("mymap.html")

