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
    
    def extract_area_poly(self,geom_col,polygon):         
        
        # Look for coordinates of internal polygon(s)
        int_coords = geom_col.apply(
                lambda g: [g3.coords for g2 in g.geoms for g3 in g2.interiors]
                ).explode().explode()
        
        # Extract coordinates of external polygon(s)
        ext = polygon.explode().geometry.exterior
        
        ext_coords = ext.apply(self.coord_lister)
        
        # A bit of a fudge but this seems to work for UTLAs with multiple external 
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
    
    def map_poly(self,area_poly,lat,long,area_name,map_folder):
        
        # Map actual target UTLA and save
        area_geoj = folium.GeoJson(data=area_poly,
                               style_function=lambda x: {'fillColor': 'orange'})
        
        m = folium.Map(location=[lat,long],tiles = 'CartoDB positron')
        area_geoj.add_to(m)
        m.save("{0}/{1}.html".format(map_folder,area_name))
    
    # Define function that draws circle with given radius around a certain 
    # point
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
    
    def circle_intersection_area(self,circle_poly,area_poly):
        
        # Area of circle that is in target utla
        circle_area_area = circle_poly.intersection(area_poly).area
        
        return circle_area_area
    
    def coord_lister(self,geom):
        coords = np.array(geom.coords)
        return (coords)
    
    def initial_circle(self,lat,long,area_poly,area_name,df_area,radius_increment,min_circle_perc_tot):
        # 1. Check centroid is in target UTLA (may be that in some cases it is in 
        # another UTLA that is entirely within the target UTLA or that an irregular 
        # (e.g. concave) shape means it is in a different UTLA)
        area_cent = Point(long,lat)
        
        if area_cent.within(area_poly):
            #	2. Try to get the biggest circle possible, where the centre is the 
            # UTLA centroid, that contains no more than X(start with 5)% of other 
            # UTLAs: start with 1km radius. If circle_perc >= 95, add 1km to radius
            # until circle_perc < 95, then use circle with radius = radius - 1.
            radius = radius_increment
            circle_poly = self.draw_circle(area_cent,radius)
            #area_perc = circle_intersection_area(circle_poly,area_poly)/area_poly.area*100
            circle_perc = self.circle_intersection_area(circle_poly,area_poly)/circle_poly.area*100
            
            # While the circle can be made bigger without the target UTLA content
            # dropping below 95%, do this
            while circle_perc >= min_circle_perc_tot:
                radius = radius + radius_increment
                circle_poly = self.draw_circle(area_cent,radius)
                #area_perc = circle_intersection_area(circle_poly,area_poly)/area_poly.area*100
                circle_perc = self.circle_intersection_area(circle_poly,area_poly)/circle_poly.area*100
            
            # Draw max size circle that satisfies the requirements
            all_poly = self.draw_circle(area_cent,radius-radius_increment)
            
            # Add details of circle to df_utla
            dict_area = {'area': area_name, 'lat': area_cent.y, 
                          'long': area_cent.x, 'radius': radius-radius_increment}
            df_area = df_area.append(dict_area, ignore_index = True)
        #ADD ELSE: CHOOSE RANDOM POINT IN POLY
        
        # Calculate the area % of the target UTLA that the circle covers
        area_perc_tot = all_poly.intersection(area_poly).area/area_poly.area*100
        circle_perc_tot = all_poly.intersection(area_poly).area/all_poly.area*100
        
        return all_poly,df_area,area_perc_tot,circle_perc_tot
    
    def dec_radius_increment(self,radius_increment,n_bad,n):
        # Every 50 consecutive unsuccessful attempts to draw a circle, half the
        # radius_increment
        if (n_bad > 0) & (round(n_bad/n) == n_bad/n):
            radius_increment = radius_increment/2
        else:
            radius_increment = radius_increment
        
        return radius_increment
    
    def poly_random_point(self,all_poly,area_poly,area_name):
        #	3. Take a random point from the circumference of the first circle and
        # check it is in the target UTLA.
        rand_no = randrange(len(all_poly.exterior.coords))
        
        rand_point = Point(list(all_poly.exterior.coords)[rand_no][0],
                           list(all_poly.exterior.coords)[rand_no][1])
        
        #	  a. If it is not, choose another random point from the circ. of the
        #   first circle and check it is in the target UTLA. Proceed to step 4 
        #   when a point on the circ. of the first circle that is in the target
        #   UTLA has been identified.
        while rand_point.within(area_poly) == False:
            print("Find a new random point that is in",area_name)
            rand_no = randrange(len(all_poly.exterior.coords))
            rand_point = Point(list(all_poly.exterior.coords)[rand_no][0],
                               list(all_poly.exterior.coords)[rand_no][1])
        
        return rand_point
    
    #	4. Repeat step 2, using the point identified in step 3. Circle_perc 
    # should be updated to be the area covered by any of the circles.
    def calc_area_perc(self,all_poly,area_poly):
        circle_area_area_tot = self.circle_intersection_area(all_poly,area_poly)
        # % of poly that's in the area
        circle_perc_tot_work = circle_area_area_tot/all_poly.area*100
        # % of area that's in the poly
        area_perc_tot_work = circle_area_area_tot/area_poly.area*100
        
        return circle_perc_tot_work, area_perc_tot_work
    
    def draw_save_circle(self,min_circle_perc_tot,circle_perc_tot_work,radius,radius_increment,rand_point,
                          area_poly,all_poly,circle_area_area_tot,n_bad,area_name,
                          df_area):
        #circle_perc_tot_work = circle_perc_tot
        while circle_perc_tot_work >= min_circle_perc_tot:
                
            radius = radius + radius_increment
            circle_poly = self.draw_circle(rand_point,radius)
            #circle_area_area = circle_intersection_area(circle_poly,area_poly)
            #problem here
            all_poly_test = cascaded_union([all_poly,circle_poly])
            #circle_area_area_tot = circle_intersection_area(all_poly,area_poly)
            circle_perc_tot_work, area_perc_tot_work = self.calc_area_perc(all_poly_test,
                                                                      area_poly)
            
            old_circle_perc_tot, old_area_perc_tot = self.calc_area_perc(all_poly,area_poly)
            
            print('radius_increment:',radius_increment,' radius:',radius)
            print('old area_perc:',old_area_perc_tot,' area_perc:',area_perc_tot_work)
            print('old circle_perc:',old_circle_perc_tot,' circle_perc:',circle_perc_tot_work)
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
    
    def fill_area_with_circles(self,area_perc_tot,circle_perc_tot,min_area_perc_tot,radius_increment,
                               n_bad,n,all_poly,area_poly,area_name,df_area,min_circle_perc_tot):
        while area_perc_tot < min_area_perc_tot:
            # Every 50 consecutive unsuccessful attempts to draw a circle, half the
            # radius_increment
            radius_increment = self.dec_radius_increment(radius_increment,n_bad,n)
            
            # Select a random point on the polygon
            rand_point = self.poly_random_point(all_poly,area_poly,area_name)
            
            #	4. Repeat step 2, using the point identified in step 3. Circle_perc 
            # should be updated to be the area covered by any of the circles.
            circle_area_area_tot = self.circle_intersection_area(all_poly,area_poly)
            circle_perc_tot_work, area_perc_tot_work = self.calc_area_perc(all_poly,
                                                                      area_poly)
            
            radius = 0
            
            all_poly, area_perc_tot, df_area, n_bad = self.draw_save_circle(
                    min_circle_perc_tot,circle_perc_tot,radius,radius_increment,rand_point,
                    area_poly,all_poly,circle_area_area_tot,n_bad,area_name,
                    df_area)
            
        return df_area, all_poly
    
    def map_circle(self,all_poly,area_name,min_area_perc_tot,lat,long,map_folder):
        # Map circle-based approximation and save
        all_geoj = folium.GeoJson(data=all_poly,
                                  style_function=lambda x: {'fillColor': 'blue'})
        
        c = folium.Map(location=[lat,long],tiles = 'CartoDB positron')
        all_geoj.add_to(c)
        c.save("{0}/{1}_circles_{2}_{3}.html".format(map_folder,area_name,
               min_area_perc_tot,min_circle_perc_tot))
    
    def areas_circles(self,areas,radius_increment,df_polygons,area_var,poly_var,
                      min_area_perc_tot,min_circle_perc_tot,map_folder):
        df_areas = pd.DataFrame()
        for area in areas:
            print(area)
            
            # Set initial radius increment to 1km
            radius_increment = radius_increment
            # Set number of consecutive unsuccessful attempts to draw a circle to 0
            n_bad = 0
            
            # Create empty dataframe, to contain centroids and radiuses of target UTLA
            # circles
            df_area = pd.DataFrame(columns = ['area','lat','long','radius'])
            
            # Keep target UTLA data only
            gdfd = df_polygons.loc[df_polygons[area_var]==area].copy()
            
            area_poly = self.extract_area_poly(gdfd[poly_var],gdfd)
            
            lat = df_polygons[df_polygons[area_var]==area].LAT
            long = df_polygons[df_polygons[area_var]==area].LONG
            
            self.map_poly(area_poly,lat,long,area,map_folder)
            
            initial_poly, df_area, area_perc_tot, circle_perc_tot = self.initial_circle(
                    lat,long,area_poly,area,df_area,radius_increment,min_circle_perc_tot)
            
            df_area, all_poly = self.fill_area_with_circles(area_perc_tot,
                                                       circle_perc_tot,
                                                       min_area_perc_tot,
                                                       radius_increment,n_bad,50,
                                                       initial_poly,area_poly,
                                                       area,df_area,
                                                       min_circle_perc_tot)
            
            # Append the circles for the target UTLA to those for all UTLAs
            df_areas = df_areas.append(df_area)
            
            # Map circle-based approximation and save
            self.map_circle(all_poly,area,min_area_perc_tot,lat,long,map_folder)
            
        return df_areas