# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 19:30:55 2021

@author: Joe.WozniczkaWells
"""

# GEOCODING

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

df_utlas.to_csv("df_utlas.csv")
