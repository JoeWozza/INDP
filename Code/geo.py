# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 19:30:55 2021

@author: Joe.WozniczkaWells
"""

# GEOCODING: approximating UTLAs of the Midlands using circles

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
res = requests.get(
    "https://opendata.arcgis.com/datasets/244b257482da4778995cf11ff99e9997_0.geojson"
)
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

# Min area of target UTLA that must be covered by circle-based approximation
min_utla_perc_tot = 98
# Min amount of circle-based approximation that must be in target UTLA
min_circle_perc_tot = 95

# Create empty dataframe, to contain centroids and radiuses of circles
df_utlas = pd.DataFrame(columns = ['utla','lat','long','radius'])

# Loop through Midlands UTLAs
for utla_name in mids_utlas:
    print(utla_name)
    
    # Set initial radius increment to 1km
    radius_increment = 1
    # Set number of consecutive unsuccessful attempts to draw a circle to 0
    n_bad = 0
    
    # Create empty dataframe, to contain centroids and radiuses of target UTLA
    # circles
    df_utla = pd.DataFrame(columns = ['utla','lat','long','radius'])
    
    # Keep target UTLA data only
    gdfd = utla_polygons.loc[utla_polygons["CTYUA21NM"]==utla_name].copy()         
    
    # Look for coordinates of internal polygon(s)
    int_coords = gdfd["geometry"].apply(
            lambda g: [g3.coords for g2 in g.geoms for g3 in g2.interiors]
            ).explode().explode()
    
    # Extract coordinates of external polygon(s)
    ext = gdfd.explode().geometry.exterior
    
    def coord_lister(geom):
        coords = np.array(geom.coords)
        return (coords)
    
    ext_coords = ext.apply(coord_lister)
    
    # A bit of a fudge but this seems to work for UTLAs with multiple external 
    # rings.
    for l in range(0,len(ext_coords)):
        if l == 0:
            if len(int_coords) > 1:
                utla_poly = Polygon(tuple([(e[0],e[1]) for e in 
                                           ext_coords.iloc[l]]),
                             ((tuple([(i[0],i[1]) for i in int_coords]),)))
            else:
                utla_poly = Polygon(tuple([(e[0],e[1]) for e in 
                                           ext_coords.iloc[l]]))
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
    
    # Map actual target UTLA and save
    utla_geoj = folium.GeoJson(data=utla_poly,
                           style_function=lambda x: {'fillColor': 'orange'})
    
    m = folium.Map(location=[utla_polygons[utla_polygons['CTYUA21NM']==
                                           utla_name].LAT,
                   utla_polygons[utla_polygons['CTYUA21NM']==utla_name].LONG],
                   tiles = 'CartoDB positron')
    utla_geoj.add_to(m)
    m.save("{0}.html".format(utla_name))
    # Delete polygon coordinates
    del int_coords, ext_coords
    
    # Define function that draws circle with given radius around a certain 
    # point
    def draw_circle(point,radius):
        # Draw approximation of a circle using geopy.destination
        d_obj_v = geopy.distance.distance(kilometers=radius)
        
        # Create array of coordinates
        circle_array = []
        for b in range(0,360):
            coords = ((d_obj_v.destination(point = geopy.Point(point.y, 
                                                               point.x),
                bearing = b).latitude),
                      (d_obj_v.destination(point = geopy.Point(point.y, 
                                                               point.x),
                bearing = b).longitude))
            circle_array.append(coords)
        
        # Convert to shapely polygon
        circle_poly = Polygon([(c[1],c[0]) for c in circle_array])
        
        # Area of circle that is in target utla
        circle_utla_area = circle_poly.intersection(utla_poly).area
        
        return circle_utla_area, circle_poly
    
    # 1. Check centroid is in target UTLA (may be that in some cases it is in 
    # another UTLA that is entirely within the target UTLA or that an irregular 
    # (e.g. concave) shape means it is in a different UTLA)
    utla_cent = Point(utla_polygons[utla_polygons['CTYUA21NM']==utla_name].LONG,
                       utla_polygons[utla_polygons['CTYUA21NM']==utla_name].LAT
                       )
    
    if utla_cent.within(utla_poly):
        #	2. Try to get the biggest circle possible, where the centre is the 
        # UTLA centroid, that contains no more than X(start with 5)% of other 
        # UTLAs: start with 1km radius. If circle_perc >= 95, add 1km to radius
        # until circle_perc < 95, then use circle with radius = radius - 1.
        radius = radius_increment
        circle_poly = draw_circle(utla_cent,radius)[1]
        utla_perc = draw_circle(utla_cent,radius)[0]/utla_poly.area*100
        circle_perc = draw_circle(utla_cent,radius)[0]/circle_poly.area*100
        
        # While the circle can be made bigger without the target UTLA content
        # dropping below 95%, do this
        while circle_perc >= min_circle_perc_tot:
            radius = radius + radius_increment
            circle_poly = draw_circle(utla_cent,radius)[1]
            utla_perc = draw_circle(utla_cent,radius)[0]/utla_poly.area*100
            circle_perc = draw_circle(utla_cent,radius)[0]/circle_poly.area*100
        
        # Draw max size circle that satisfies the requirements
        all_poly = draw_circle(utla_cent,radius-radius_increment)[1]
        
        # Add details of circle to df_utla
        dict_utla = {'utla': utla_name, 'lat': utla_cent.y, 
                      'long': utla_cent.x, 'radius': radius-radius_increment}
        df_utla = df_utla.append(dict_utla, ignore_index = True)
    
    # Calculate the area % of the target UTLA that the circle covers
    utla_perc_tot = all_poly.intersection(utla_poly).area/utla_poly.area*100
    
    while utla_perc_tot < min_utla_perc_tot:
        # Every 50 consecutive unsuccessful attempts to draw a circle, half the
        # radius_increment
        if (n_bad > 0) & (round(n_bad/50) == n_bad/50):
            radius_increment = radius_increment/2
        
        #	3. Take a random point from the circumference of the first circle and
        # check it is in the target UTLA.
        rand_no = randrange(len(all_poly.exterior.coords))
        
        rand_point = Point(list(all_poly.exterior.coords)[rand_no][0],
                           list(all_poly.exterior.coords)[rand_no][1])
        
        #	  a. If it is not, choose another random point from the circ. of the
        #   first circle and check it is in the target UTLA. Proceed to step 4 
        #   when a point on the circ. of the first circle that is in the target
        #   UTLA has been identified.
        while rand_point.within(utla_poly) == False:
            print("Find a new random point that is in",utla_name)
            rand_no = randrange(len(all_poly.exterior.coords))
            rand_point = Point(list(all_poly.exterior.coords)[rand_no][0],
                               list(all_poly.exterior.coords)[rand_no][1])
        
        #	4. Repeat step 2, using the point identified in step 3. Circle_perc 
        # should be updated to be the area covered by any of the circles.
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
            
            print('radius_increment:',radius_increment)
            print('Radius:',radius)
            print('utla_perc:',utla_perc_tot_work)
            print('circle_perc:',circle_perc_tot_work)
            print('n_bad:',n_bad)
        
        # If a circle has successfully been drawn, save it
        if radius > radius_increment:
            dict_utla = {'utla': utla_name, 'lat': rand_point.y, 
                         'long': rand_point.x, 
                         'radius': radius-radius_increment}
            df_utla = df_utla.append(dict_utla, ignore_index = True)    
            circle_poly = draw_circle(rand_point,radius - radius_increment)[1]      
            all_poly = cascaded_union([all_poly,circle_poly])
            utla_perc_tot = (all_poly.intersection(utla_poly).area/
                             utla_poly.area*100)
            n_bad = 0
        # Otherwise add one to the consecutive unsuccessful attempts counter
        else:
            n_bad = n_bad + 1
    
    # Append the circles for the target UTLA to those for all UTLAs
    df_utlas = df_utlas.append(df_utla)
    
    # Map circle-based approximation and save
    all_geoj = folium.GeoJson(data=all_poly,
                              style_function=lambda x: {'fillColor': 'blue'})
    
    c = folium.Map(location=[utla_polygons[utla_polygons['CTYUA21NM']==
                                           utla_name].LAT,
                   utla_polygons[utla_polygons['CTYUA21NM']==utla_name].LONG],
                   tiles = 'CartoDB positron')
    all_geoj.add_to(c)
    c.save("{0}_circles.html".format(utla_name))

# Save df_utlas as csv (temporary solution while still putting together the
# other code, so I don't have to run this every time I want to work on
# subsequent code)
df_utlas.to_csv("df_utlas.csv")
