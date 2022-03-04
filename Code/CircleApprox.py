# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:57:20 2022

@author: Joe.WozniczkaWells
"""

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

class CircleApprox():
        
    # Extracts the polygon of an area from the coordinates in a geopandas 
    # geoseries
    # geom_col: geopandas geoseries containing coordinates of area
    def extract_area_poly(self,geom_col):         
        
        # Look for coordinates of internal polygon(s)
        int_coords = geom_col.apply(
                lambda g: [g3.coords for g2 in g.geoms for g3 in g2.interiors]
                ).explode().explode()
        
        # Extract coordinates of external polygon(s)
        ext = geom_col.explode().exterior
        
        ext_coords = ext.apply(self.coord_lister)
        
        # A bit of a fudge but this seems to work for areas with multiple external 
        # rings.
        for l in range(0,len(ext_coords)):
            if l == 0:
                if len(int_coords) > 1:
                    area_poly = Polygon(tuple([(e[0],e[1]) for e in 
                                               ext_coords.iloc[l]]),
                                 ((tuple([(i[0],i[1]) for i in int_coords]),)))
                else:
                    area_poly = Polygon(tuple([(e[0],e[1]) for e in 
                                               ext_coords.iloc[l]]))
            else:
                if len(int_coords) > 1:
                    area_poly = cascaded_union([area_poly,
                                                Polygon(tuple([(e[0],e[1]) for e in
                                                              ext_coords.iloc[l]]),
                    ((tuple([(i[0],i[1]) for i in int_coords]),)))])
                else:
                    area_poly = cascaded_union([area_poly,
                                                Polygon(tuple([(e[0],e[1]) for e in 
                                                            ext_coords.iloc[l]]))])
        
        return area_poly
    
    # Creates and saves map of polygon
    # area_poly: shapely polygon to be mapped
    # lat: latitude of centre of mapped area
    # long: longitude of centre of mapped area
    # area_name: name of area to be mapped
    # map_folder: folder in which to save map
    def map_poly(self,area_poly,lat,long,area_name,map_folder):
        
        # Map actual target area and save
        area_geoj = folium.GeoJson(data=area_poly,
                               style_function=lambda x: {'fillColor': 'orange'})
        
        m = folium.Map(location=[lat,long],tiles = 'CartoDB positron')
        area_geoj.add_to(m)
        m.save("{0}/{1}.html".format(map_folder,area_name))
    
    # Draws circle with given radius around a certain point
    # point: shapely point to draw a circle around
    # radius: radius of the desired circle
    def draw_circle(self,point,radius):
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
        
        return circle_poly
    
    # Calculates intersection area between two shapely polygons, specifically
    # a circle and a target area polygon
    # circle_poly: shapely polygon of circle
    # area_poly: shapely polygon of target area
    def circle_intersection_area(self,circle_poly,area_poly):
        
        # Area of circle that is in target area
        circle_area_area = circle_poly.intersection(area_poly).area
        
        return circle_area_area
    
    # Lists coordinates from a geopandas geoseries
    # geom: geopandas geoseries
    def coord_lister(self,geom):
        coords = np.array(geom.coords)
        return (coords)
    
    # Draws an initial circle within a target area, increasing in size within
    # the bounds of min_circle_perc_tot
    # lat: latitude of centre of target area
    # long: longitude of centre of target area
    # area_poly: shapely polygon of target area
    # area_name: name of target area
    # df_area: pandas dataframe in which to save details of circle
    # radius_increment: increment by which radius should increase
    # min_circle_perc_tot: minimum percentage of circle-covered area that must
    #   be within target area
    def initial_circle(self,lat,long,area_poly,area_name,df_area,
                       radius_increment,min_circle_perc_tot):
        # 1. Check centroid is in target area (may be that in some cases it is 
        # in another area that is entirely within the target area or that an 
        # irregular (e.g. concave) shape means it is in a different area)
        area_cent = Point(long,lat)
        
        if area_cent.within(area_poly):
            #	2. Try to get the biggest circle possible, where the centre is 
            # the area centroid, that contains no more than 
            # (100-min_circle_perc_tot) % of other areas: start with 
            # radius_increment radius. If circle_perc >= min_circle_perc_tot, 
            # add radius_increment km to radius until circle_perc < 
            # min_circle_perc_tot, then use circle with 
            # radius = radius - radius_increment.
            radius = radius_increment
            circle_poly = self.draw_circle(area_cent,radius)
            circle_perc = self.circle_intersection_area(circle_poly,area_poly)/circle_poly.area*100
            
            # While the circle can be made bigger without the target area 
            # content dropping below min_circle_perc_tot %, do so
            while circle_perc >= min_circle_perc_tot:
                radius = radius + radius_increment
                circle_poly = self.draw_circle(area_cent,radius)
                circle_perc = self.circle_intersection_area(circle_poly,area_poly)/circle_poly.area*100
            
            # Draw max size circle that satisfies the requirements
            all_poly = self.draw_circle(area_cent,radius-radius_increment)
            
            # Add details of circle to df_area
            dict_area = {'area': area_name, 'lat': area_cent.y, 
                          'long': area_cent.x, 'radius': radius-radius_increment}
            df_area = df_area.append(dict_area, ignore_index = True)
        #ADD ELSE: CHOOSE RANDOM POINT IN POLY
        
        # Calculate the area % of the target area that the circle covers
        area_perc_tot = (all_poly.intersection(area_poly).area/area_poly.area
                         *100)
        circle_perc_tot = (all_poly.intersection(area_poly).area/all_poly.area
                           *100)
        
        return all_poly,df_area,area_perc_tot,circle_perc_tot
    
    # Halves the radius increment after every n consecutive unsuccessful
    # attempts to draw a new circle
    # radius_increment: current increment by which radius should increase
    # n_bad: number of consecutive unsuccessful attempts to draw a circle
    # n: number of consecutive unsuccessfull attempts to draw a circle after
    #   which radius should be halved
    def dec_radius_increment(self,radius_increment,n_bad,n):
        if (n_bad > 0) & (round(n_bad/n) == n_bad/n):
            radius_increment = radius_increment/2
        else:
            radius_increment = radius_increment
        
        return radius_increment
    
    # Selects a random point that is in the target area from the perimeter of a 
    # shapely polygon
    # all_poly: shapely polygon from which to select point
    # area_poly: shapely polygon of target area
    # area_name: name of target area
    def poly_random_point(self,all_poly,area_poly,area_name):
        # Take a random point from the circumference of the first circle and
        # check it is in the target area.
        rand_no = randrange(len(all_poly.exterior.coords))
        
        rand_point = Point(list(all_poly.exterior.coords)[rand_no][0],
                           list(all_poly.exterior.coords)[rand_no][1])
        
        # If it is not, choose another random point from the perimeter of the 
        # first circle and check it is in the target area. Repeat until a valid
        # point is identified
        while rand_point.within(area_poly) == False:
            print("Find a new random point that is in",area_name)
            rand_no = randrange(len(all_poly.exterior.coords))
            rand_point = Point(list(all_poly.exterior.coords)[rand_no][0],
                               list(all_poly.exterior.coords)[rand_no][1])
        
        return rand_point
    
    # Calculates the percentage of a target area that is covered by an 
    # approximate area and the percentage of the approximate area that is 
    # within the target area
    # all_poly: shapely polygon of approximate area
    # area_poly: shapely polygon of target area
    def calc_area_perc(self,all_poly,area_poly):
        circle_area_area_tot = self.circle_intersection_area(all_poly,
                                                             area_poly)
        # % of poly that's in the area
        circle_perc_tot_work = circle_area_area_tot/all_poly.area*100
        # % of area that's in the poly
        area_perc_tot_work = circle_area_area_tot/area_poly.area*100
        
        return circle_perc_tot_work, area_perc_tot_work
    
    # Draws a circle from a specified point and increases the radius
    # incrementally until it cannot be increased further within the parameters
    # set by min_circle_perc_tot
    # min_circle_perc_tot: minimum percentage of approximate area that must be
    #   within target area
    # circle_perc_tot_work: current percentage of approximate area within 
    #   target area
    # radius: initial radius of new circle
    # radius_increment: increment by which radius should increase
    # rand_point: shapely point of centre of new circle
    # area_poly: shapely polygon of target area
    # all_poly: shapely polygon of target area
    # circle_area_area_tot: REMOVE
    # n_bad: number of consecutive unsuccessful attempts to draw a circle
    # area_name: name of target area
    # df_area: pandas dataframe in which to save details of circle
    def draw_save_circle(self,min_circle_perc_tot,circle_perc_tot_work,radius,
                         radius_increment,rand_point,area_poly,all_poly,
                         circle_area_area_tot,n_bad,area_name,df_area):
        
        while circle_perc_tot_work >= min_circle_perc_tot:
                
            radius = radius + radius_increment
            circle_poly = self.draw_circle(rand_point,radius)
            all_poly_test = cascaded_union([all_poly,circle_poly])
            circle_perc_tot_work, area_perc_tot_work = self.calc_area_perc(
                    all_poly_test,area_poly)
            
            old_circle_perc_tot, old_area_perc_tot = self.calc_area_perc(
                    all_poly,area_poly)
            
            print('radius_increment:',radius_increment,' radius:',radius)
            print('old area_perc:',old_area_perc_tot,' area_perc:',
                  area_perc_tot_work)
            print('old circle_perc:',old_circle_perc_tot,' circle_perc:',
                  circle_perc_tot_work)
            print('n_bad:',n_bad)
            
        # If a circle has successfully been drawn, save it
        if radius > radius_increment:
            dict_area = {'area': area_name, 'lat': rand_point.y, 
                         'long': rand_point.x, 
                         'radius': radius-radius_increment}
            df_area = df_area.append(dict_area, ignore_index = True)    
            circle_poly = self.draw_circle(rand_point,radius - radius_increment)
            all_poly = cascaded_union([all_poly,circle_poly])
            circle_perc_tot_work, area_perc_tot = self.calc_area_perc(all_poly,area_poly)
            n_bad = 0
        # Otherwise add one to the consecutive unsuccessful attempts counter
        else:
            n_bad = n_bad + 1
            area_perc_tot = self.calc_area_perc(all_poly,area_poly)[1]
            
        return all_poly, area_perc_tot, df_area, n_bad
    
    # Creates a circle-based approximation of a target area
    # area_perc_tot: percentage of target area covered by approximate area
    # circle_perc_tot: percentage of approximate area within target area
    # min_area_perc_tot: minimum percentage of target area to be covered by
    #   approximate area
    # radius_increment: increment by which radius should increase
    # n_bad: number of consecutive unsuccessful attempts to draw a circle
    # n: number of consecutive unsuccessfull attempts to draw a circle after
    #   which radius should be halved
    # all_poly: shapely polygon of target area
    # area_poly: shapely polygon of target area
    # area_name: name of target area
    # df_area: pandas dataframe in which to save details of circle
    # min_circle_perc_tot: minimum percentage of approximate area that must be
    #   within target area
    def fill_area_with_circles(self,area_perc_tot,circle_perc_tot,
                               min_area_perc_tot,radius_increment,n_bad,n,
                               all_poly,area_poly,area_name,df_area,
                               min_circle_perc_tot):
        while area_perc_tot < min_area_perc_tot:
            # Every 50 consecutive unsuccessful attempts to draw a circle, half the
            # radius_increment
            radius_increment = self.dec_radius_increment(radius_increment,
                                                         n_bad,n)
            
            # Select a random point on the polygon
            rand_point = self.poly_random_point(all_poly,area_poly,area_name)
            
            #	4. Repeat step 2, using the point identified in step 3. Circle_perc 
            # should be updated to be the area covered by any of the circles.
            circle_area_area_tot = self.circle_intersection_area(all_poly,
                                                                 area_poly)
            circle_perc_tot_work, area_perc_tot_work = self.calc_area_perc(
                    all_poly,area_poly)
            
            radius = 0
            
            all_poly, area_perc_tot, df_area, n_bad = self.draw_save_circle(
                    min_circle_perc_tot,circle_perc_tot,radius,
                    radius_increment,rand_point,area_poly,all_poly,
                    circle_area_area_tot,n_bad,area_name,df_area)
            
        return df_area, all_poly
    
    # Creates and saves map of circle
    # all_poly: shapely polygon of target area
    # area_name: name of target area
    # min_area_perc_tot: minimum percentage of target area to be covered by
    #   approximate area# lat: latitude of centre of mapped area
    # lat: latitude of centre of mapped area
    # long: longitude of centre of mapped area
    # map_folder: folder in which to save map
    def map_circle(self,all_poly,area_name,min_area_perc_tot,lat,long,
                   map_folder):
        # Map circle-based approximation and save
        all_geoj = folium.GeoJson(data=all_poly,
                                  style_function=lambda x: {'fillColor': 
                                      'blue'})
        
        c = folium.Map(location=[lat,long],tiles = 'CartoDB positron')
        all_geoj.add_to(c)
        c.save("{0}/{1}_circles_{2}_{3}.html".format(map_folder,area_name,
               min_area_perc_tot,min_circle_perc_tot))
    
    # Loops over areas and creates and saves circle-based approximations of
    # them, as maps and coordinates/radiuses
    # areas: list of target areas to loop over
    # radius_increment: increment by which radius should increase
    # df_polygons: pandas dataframe containing target area polygons
    # area_var: variable in df_polygons containing name of target area
    # poly_var: variable in df_polygons containing polygon coordinates as
    #   geopandas geoseries
    # min_area_perc_tot: minimum percentage of target area to be covered by
    #   approximate area
    # min_circle_perc_tot: minimum percentage of circle-covered area that must
    #   be within target area
    # map_folder: folder in which to save map
    def areas_circles(self,areas,radius_increment,df_polygons,area_var,
                      poly_var,min_area_perc_tot,min_circle_perc_tot,
                      map_folder):
        df_areas = pd.DataFrame()
        for area in areas:
            print(area)
            
            # Set initial radius increment to 1km
            radius_increment = radius_increment
            # Set number of consecutive unsuccessful attempts to draw a circle 
            # to 0
            n_bad = 0
            
            # Create empty dataframe, to contain centroids and radiuses of 
            # target area circles
            df_area = pd.DataFrame(columns = ['area','lat','long','radius'])
            
            # Keep target area data only
            gdfd = df_polygons.loc[df_polygons[area_var]==area].copy()
            
            area_poly = self.extract_area_poly(gdfd[poly_var])
            
            lat = df_polygons[df_polygons[area_var]==area].LAT
            long = df_polygons[df_polygons[area_var]==area].LONG
            
            self.map_poly(area_poly,lat,long,area,map_folder)
            
            initial_poly, df_area, area_perc_tot, circle_perc_tot = (
                    self.initial_circle(lat,long,area_poly,area,df_area,
                                        radius_increment,min_circle_perc_tot)
            
            df_area, all_poly = self.fill_area_with_circles(area_perc_tot,
                                                       circle_perc_tot,
                                                       min_area_perc_tot,
                                                       radius_increment,n_bad,
                                                       50,initial_poly,
                                                       area_poly,area,df_area,
                                                       min_circle_perc_tot)
            
            # Append the circles for the target area to those for all areas
            df_areas = df_areas.append(df_area)
            
            # Map circle-based approximation and save
            self.map_circle(all_poly,area,min_area_perc_tot,lat,long,
                            map_folder)
            
        return df_areas