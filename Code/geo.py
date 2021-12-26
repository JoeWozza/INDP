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
from shapely.geometry import Point, mapping, Polygon, MultiPolygon
from shapely.geometry import shape
from shapely.ops import cascaded_union
import shapely
import pandas as pd
import folium
import geopy
import geopy.distance
from random import randrange
import random
import requests

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
# Works for Derbyshire, now need to check for other areas
res = requests.get(
    "https://opendata.arcgis.com/datasets/244b257482da4778995cf11ff99e9997_0.geojson"
)
utla_polygons = gpd.GeoDataFrame.from_features(res.json()).set_crs("epsg:4326")

random.seed(123)

mids_utlas = ['Derby','Leicester','Rutland','Nottingham',
              'Herefordshire, County of','Telford and Wrekin','Stoke-on-Trent',
              'Shropshire','North Northamptonshire','West Northamptonshire',
              'Birmingham','Coventry','Dudley','Sandwell','Solihull','Walsall',
              'Wolverhampton','Derbyshire','Leicestershire','Lincolnshire',
              'Nottinghamshire','Staffordshire','Warwickshire','Worcestershire'
              ]

#mids_utlas = ['Derbyshire','Nottingham','Derby']

min_utla_perc_tot = 98
max_circle_perc_tot = 95

# Create empty dataframe, to contain centroids and radiuses of circles
df_utlas = pd.DataFrame(columns = ['utla','lat','long','radius'])

for utla_name in mids_utlas:
    print(utla_name)
    
    radius_increment = 1
    n_bad = 0
    
    df_utla = pd.DataFrame(columns = ['utla','lat','long','radius'])
    gdfd = utla_polygons.loc[utla_polygons["CTYUA21NM"]==utla_name].copy()         
    
    int_coords = gdfd["geometry"].apply(
            lambda g: [g3.coords for g2 in g.geoms for g3 in g2.interiors]
            ).explode().explode()
    
    ext = gdfd.explode().geometry.exterior
    
    def coord_lister(geom):
        coords = np.array(geom.coords)
        return (coords)
    
    #ext_coords = (ext.apply(coord_lister).iloc[0])
    ext_coords = ext.apply(coord_lister)
    
    # A bit of a fudge but this seems to work for UTLAs with multiple external 
    # rings.
    # Need something like: if int_coords = NaN then only use ext_coords
    for l in range(0,len(ext_coords)):
        if l == 0:
            if len(int_coords) > 1:
                utla_poly = Polygon(tuple([(e[0],e[1]) for e in ext_coords.iloc[l]]),
                             ((tuple([(i[0],i[1]) for i in int_coords]),)))
            else:
                utla_poly = Polygon(tuple([(e[0],e[1]) for e in ext_coords.iloc[l]]))
        else:
            if len(int_coords) > 1:
                utla_poly = cascaded_union([utla_poly,
                                            Polygon(tuple([(e[0],e[1]) for e in 
                                                           ext_coords.iloc[l]]),
                ((tuple([(i[0],i[1]) for i in int_coords]),)))])
            else:
                utla_poly = cascaded_union([utla_poly,
                                            Polygon(tuple([(e[0],e[1]) for e in 
                                                           ext_coords.iloc[l]]))])
    
    utla_geoj = folium.GeoJson(data=utla_poly,
                           style_function=lambda x: {'fillColor': 'orange'})
    
    m = folium.Map(location=[utla_polygons[utla_polygons['CTYUA21NM']==utla_name].LAT,
                   utla_polygons[utla_polygons['CTYUA21NM']==utla_name].LONG],
                   tiles = 'CartoDB positron')
    utla_geoj.add_to(m)
    m.save("{0}.html".format(utla_name))
    del int_coords, ext_coords
    
    def draw_circle(point,radius):
        # Draw approximation of a circle using geopy.destination
        d_obj_v = geopy.distance.distance(kilometers=radius)
        
        circle_array = []
        for b in range(0,360):
            coords = ((d_obj_v.destination(point = geopy.Point(point.y, point.x), bearing = b).latitude),
                      (d_obj_v.destination(point = geopy.Point(point.y, point.x), bearing = b).longitude))
            circle_array.append(coords)
            
        circle_poly = Polygon([(c[1],c[0]) for c in circle_array])
        
        # Area of circle that is in target utla
        circle_utla_area = circle_poly.intersection(utla_poly).area
        
        return circle_utla_area, circle_poly
    
    # 1. Check centroid is in target UTLA (may be that in some cases it is in 
    # another UTLA that is entirely within the target UTLA or that an irregular 
    # (e.g. concave) shape means it is in a different UTLA)
    utla_cent = Point(utla_polygons[utla_polygons['CTYUA21NM']==utla_name].LONG,
                       utla_polygons[utla_polygons['CTYUA21NM']==utla_name].LAT)
    
    if utla_cent.within(utla_poly):
        #	2. Try to get the biggest circle possible, where the centre is the UTLA 
        # centroid, that contains no more than X(start with 5)% of other UTLAs: 
        # start with 1km radius. If circle_perc >= 95, add 1km to radius until 
        # circle_perc < 95, then use circle with radius = radius - 1.
        radius = radius_increment
        circle_poly = draw_circle(utla_cent,radius)[1]
        utla_perc = draw_circle(utla_cent,radius)[0]/utla_poly.area*100
        circle_perc = draw_circle(utla_cent,radius)[0]/circle_poly.area*100
        
        while circle_perc >= max_circle_perc_tot:
            radius = radius + radius_increment
            circle_poly = draw_circle(utla_cent,radius)[1]
            utla_perc = draw_circle(utla_cent,radius)[0]/utla_poly.area*100
            circle_perc = draw_circle(utla_cent,radius)[0]/circle_poly.area*100
        
        all_poly = draw_circle(utla_cent,radius-radius_increment)[1]
        
        dict_utla = {'utla': utla_name, 'lat': utla_cent.y, 
                      'long': utla_cent.x, 'radius': radius-radius_increment}
        df_utla = df_utla.append(dict_utla, ignore_index = True)
        
    utla_perc_tot = all_poly.intersection(utla_poly).area/utla_poly.area*100
    
    while utla_perc_tot < min_utla_perc_tot:
        if (n_bad > 0) & (round(n_bad/50) == n_bad/50):
            radius_increment = radius_increment/2
        
        #	3. Take a random point from the circumference of the first circle and check 
        # it is in the target UTLA.
        rand_no = randrange(len(all_poly.exterior.coords))
        
        rand_point = Point(list(all_poly.exterior.coords)[rand_no][0],
                           list(all_poly.exterior.coords)[rand_no][1])
        
        #		a. If it is not, choose another random point from the circ. of the first 
        #   circle and check it is in the target UTLA. Proceed to step 4 when a point 
        #   on the circ. of the first circle that is in the target UTLA has been 
        #   identified.
        while rand_point.within(utla_poly) == False:
            print("Find a new random point that is in",utla_name)
            rand_no = randrange(len(all_poly.exterior.coords))
            rand_point = Point(list(all_poly.exterior.coords)[rand_no][0],
                               list(all_poly.exterior.coords)[rand_no][1])
        
        #	4. Repeat step 2, using the point identified in step 3. Circle_perc should 
        # be updated to be the area covered by any of the circles.
        circle_utla_area_tot = all_poly.intersection(utla_poly).area
        circle_perc_tot_work = circle_utla_area_tot/all_poly.area*100
        utla_perc_tot_work = circle_utla_area_tot/utla_poly.area*100
        radius = 0
        while circle_perc_tot_work >= 95:
            radius = radius + radius_increment
            circle_utla_area, circle_poly = draw_circle(rand_point,radius)
            utla_perc_tot_work = ((circle_utla_area_tot + circle_utla_area - 
                               all_poly.intersection(circle_poly)
                               .intersection(utla_poly).area)/
                utla_poly.area*100)
            circle_perc_tot_work = ((circle_utla_area_tot + circle_utla_area - 
                                all_poly.intersection(circle_poly)
                               .intersection(utla_poly).area)/
                (all_poly.area + circle_poly.area - 
                 all_poly.intersection(circle_poly).area)*100)
            print('radius_increment:')
            print(radius_increment)
            print('Radius:')
            print(radius)
            print('utla_perc:')
            print(utla_perc_tot_work)
            print('circle_perc:')
            print(circle_perc_tot_work)
            print('n_bad:')
            print(n_bad)
        
        if radius > radius_increment:
            dict_utla = {'utla': utla_name, 'lat': rand_point.y, 
                         'long': rand_point.x, 'radius': radius-radius_increment}
            df_utla = df_utla.append(dict_utla, ignore_index = True)    
            circle_poly = draw_circle(rand_point,radius - radius_increment)[1]      
            all_poly = cascaded_union([all_poly,circle_poly])
            utla_perc_tot = all_poly.intersection(utla_poly).area/utla_poly.area*100
            n_bad = 0
        else:
            # Add one to unsuccessful coordinates
            n_bad = n_bad + 1
    
    print('Saving')
    df_utlas = df_utlas.append(df_utla)
    
    all_geoj = folium.GeoJson(data=all_poly,
                              style_function=lambda x: {'fillColor': 'blue'})
    
    c = folium.Map(location=[utla_polygons[utla_polygons['CTYUA21NM']==utla_name].LAT,
                   utla_polygons[utla_polygons['CTYUA21NM']==utla_name].LONG],
                   tiles = 'CartoDB positron')
    all_geoj.add_to(c)
    c.save("{0}_circles.html".format(utla_name))

#%% Try to sort out hole in Derbyshire

# BFC - single bracket around second polygon (gap for Derby) where in Hartlepool there is a double bracket around second polygon (albeit this isn't a gap)
utla_polygons = gpd.read_file('https://opendata.arcgis.com/datasets/244b257482da4778995cf11ff99e9997_0.geojson')

derbs = utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire']#.geometry

derbs_ext = derbs.explode().geometry.exterior

def coord_lister(geom):
    coords = np.array(geom.coords)
    return (coords)

derbs_ext_coords = (derbs_ext.apply(coord_lister).iloc[0])

derbs_ext_poly = Polygon([(d[0],d[1]) for d in derbs_ext_coords])
# Works for exterior, not interior


derbs_int = derbs.explode().geometry.interiors
derbs_int_coords = []
for interior in derbs_int:
    print(list(interior))
    derbs_int_coords += interior.coords[:]
    
[a.coords for a in derbs.explode().geometry.interiors]

derbs_int_coords = np.array(derbs_int)

derbs_poly = Polygon([(d[0],d[1]) for d in derbs_int])



derbs_explode = derbs.explode().apply(coord_lister).iloc[0]
derbs_explode_poly = Polygon([(d[0],d[1]) for d in derbs_explode])

print([p for p in list(derbs)])


# From https://stackoverflow.com/questions/21824157/how-to-extract-interior-polygon-coordinates-using-shapely

def extract_poly_coords(geom):
    print('-1')
    if geom.type.values[0] == 'Polygon':
        print('0')
        exterior_coords = geom.exterior.coords[:]
        interior_coords = []
        for interior in geom.interiors:
            interior_coords += interior.coords[:]
    elif geom.type.values[0] == 'MultiPolygon':
        print('1')
        exterior_coords = []
        interior_coords = []
        print('2')
        for part in geom:
            print('3')
            epc = extract_poly_coords(part)  # Recursive call
            print('4')
            exterior_coords += epc['exterior_coords']
            print('5')
            interior_coords += epc['interior_coords']
    else:
        raise ValueError('Unhandled geometry type: ' + repr(geom.type))
    print('6')
    return {'exterior_coords': exterior_coords,
            'interior_coords': interior_coords}
    
extract_poly_coords(derbs.geometry)

for part in derbs_explode.geometry:
    print('1')
    
    
# Reproducible example for StackOverflow
import geopandas as gpd

utla_polygons = gpd.read_file('https://opendata.arcgis.com/datasets/244b257482da4778995cf11ff99e9997_0.geojson')

derbs = utla_polygons[utla_polygons['CTYUA21NM']=='Derbyshire']
derbs_int = derbs.explode().geometry.interiors
print(derbs_int)

derbs_int.coords

# StackOverflow solution 1
import geopandas as gpd
import requests

res = requests.get(
    "https://opendata.arcgis.com/datasets/244b257482da4778995cf11ff99e9997_0.geojson"
)
gdf = gpd.GeoDataFrame.from_features(res.json()).set_crs("epsg:4326")

gdfd = gdf.loc[gdf["CTYUA21NM"].str.contains("Derbyshire")].copy()

gdfd_coords = gdfd["geometry"].apply(
        lambda g: [g3.coords for g2 in g.geoms for g3 in g2.interiors]
        ).explode().explode()

derbs_poly = Polygon(tuple([(e[0],e[1]) for e in derbs_ext_coords]),
                     ((tuple([(i[0],i[1]) for i in gdfd_coords]),)))

# StackOverflow solution 2
from shapely.geometry import shape
import requests

url = 'https://opendata.arcgis.com/datasets/244b257482da4778995cf11ff99e9997_0.geojson'
r = requests.get(url)
data = r.json()
for f in data['features']:
    print(f)
    if f['properties']['CTYUA21NM'] == 'Derby':
        geom = f['geometry']
        if geom.get('type') == 'MultiPolygon':
            print("BEFORE:")
            for p in shape(geom).geoms:
                print(len(p.interiors))
            geom = [g[0] for g in geom['coordinates']]
            geom = shape({
                'type': 'MultiPolygon',
                'coordinates': [geom]
            })
            # geom is a MultiPolygon instance with polygons only having an exterior ring
            print("AFTER:")
            for p in geom.geoms:
                print(len(p.interiors))
        break
    
# Identify whether there is an internal polygon
data_json = res.json()
geom = data_json['geometry']
for p in shape(geom).geoms:
    print(len(p.interiors))
    
    
url = 'https://opendata.arcgis.com/datasets/244b257482da4778995cf11ff99e9997_0.geojson'
r = requests.get(url)
data = r.json()
for f in data['features']:
    if f['properties']['CTYUA21NM'] == 'Nottinghamshire':
        geom = f['geometry']
        if geom.get('type') == 'MultiPolygon':
            print("BEFORE:")
            for p in shape(geom).geoms:
                print(len(p.interiors))
                geom = [g[0] for g in geom['coordinates']]
                geom = shape({
                'type': 'MultiPolygon',
                'coordinates': [geom]
            })
        # geom is a MultiPolygon instance with polygons only having an exterior ring
        print("AFTER:")
        for p in geom.geoms:
            print(len(p.interiors))
###   