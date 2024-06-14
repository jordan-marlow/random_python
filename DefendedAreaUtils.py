#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:24:55 2019

@author: klinetry
"""
import os
import sys
import numpy as np


if os.name == 'posix':
    rootpathname = '/'
else:
    rootpathname = os.path.splitdrive(sys.executable)[0]
if os.path.isfile(os.path.realpath(__file__)):
    RELATIVE_LIB_PATH = os.path.realpath(__file__)
    while os.path.basename(RELATIVE_LIB_PATH) != 'src' and RELATIVE_LIB_PATH != rootpathname:
        RELATIVE_LIB_PATH = os.path.dirname(RELATIVE_LIB_PATH)
    if __name__ == '__main__':
        sys.path.append(RELATIVE_LIB_PATH)
        sys.path.pop(0)
else:
    RELATIVE_LIB_PATH = os.path.dirname(sys.executable)

    
from utils.UnitConversionUtils import convertUnit,radianalias

class Circle_DA(object):
    '''
    This is a standard template for Circle DAs.
    Standard units:
        Angles - degrees
        Distances - kilometers
    
    Input:
        name - name of the DA
        lat - the latitude of the point of reference [Differs for slanted and slaved]
                Also, the latitude needs to be in degrees.
        lon - the longitude of the point of reference [Differs for slanted and slaved]
                Also, the longitude needs to be in degrees.
        radius - the radius of the DA in km
    Kwargs:
        N/A
    
    Notes:
        For setting or getting in this class, the units keyword should be used
        if you are giving or wanting values in different units than the standard.
    '''
    def __init__(self,name,lat=0,lon=0,radius=0):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.radius = radius
    
    def setLat(self,lat,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            lat = np.rad2deg(lat)
        self.lat = lat
    
    def setLon(self,lon,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            lon = np.rad2deg(lon)
        self.lon = lon
    
    def setRadius(self,radius,**kwargs):
        units = kwargs.get('units','km')
        if units != 'km':
            radius = convertUnit(radius, units, 'km')
        self.radius = radius
            
    def getName(self,**kwargs):
        upper = kwargs.get('upper',False)
        lower = kwargs.get('lower',False)
        if upper:
            return self.name.upper()
        elif lower:
            return self.name.lower()
        else:
            return self.name
        
    def getRadius(self,**kwargs):
        units = kwargs.get('units','km')
        if units != 'km':
            return convertUnit(self.radius, 'km', units)
        return self.radius
    
    def getLat(self,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            return np.deg2rad(self.lat)
        else:
            return self.lat
    
    def getLon(self,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            return np.deg2rad(self.lon)
        else:
            return self.lon
        
class Ellipse_DA(object):
    '''
    This is a standard template for Ellipse DAs.
    Standard units:
        Angles - degrees
        Distances - kilometers
    
    Input:
        name - name of the DA
        lat - the latitude of the point of reference.
                Also, the latitude needs to be in degrees.
        lon - the longitude of the point of reference.
                Also, the longitude needs to be in degrees.
        major- The half-vertical extent of the ellipse at orientation 0.  Needs to be in km
        minor - The half-horizontal extent of the ellipse at orientation 0.  Needs to be in km
        orientation - the orientation of the ellipse about the point of reference
                Also, orientation needs to be in degrees
    Kwargs:
        N/A
    
    Notes:
        For setting or getting in this class, the units keyword should be used
        if you are giving or wanting values in different units than the standard.
    '''
    def __init__(self,name,lat=0,lon=0,major=0,minor=0,orientation=0):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.major = major
        self.minor = minor
        self.orientation = orientation
        
    def setLat(self,lat,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            lat = np.rad2deg(lat)
        self.lat = lat
    
    def setLon(self,lon,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            lon = np.rad2deg(lon)
        self.lon = lon
    
    def setMajor(self,major,**kwargs):
        units = kwargs.get('units','km')
        if units != 'km':
            major = convertUnit(major, units, 'km')
        self.major = major
    
    def setMinor(self,minor,**kwargs):
        units = kwargs.get('units','km')
        if units != 'km':
            minor = convertUnit(minor, units, 'km')
        self.minor = minor
        
    def setOrientation(self,orientation,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            orientation = np.rad2deg(orientation)
        self.orientation = orientation
        
    def getName(self,**kwargs):
        upper = kwargs.get('upper',False)
        lower = kwargs.get('lower',False)
        if upper:
            return self.name.upper()
        elif lower:
            return self.name.lower()
        else:
            return self.name
    
    def getLat(self,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            return np.deg2rad(self.lat)
        return self.lat
    
    def getLon(self,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            return np.deg2rad(self.lon)
        return self.lon
    
    def getMajor(self,**kwargs):
        units = kwargs.get('units','km')
        if units != 'km':
            return convertUnit(self.major,'km',units)
        return self.major
    
    def getMinor(self,**kwargs):
        units = kwargs.get('units','km')
        if units != 'km':
            return convertUnit(self.minor, 'km', units)
        return self.minor
    
    def getOrientation(self,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            return np.deg2rad(self.orienation)
        return self.orientation
    
class Rectangle_DA(object):
    '''
    This is a standard template for Rectangle DAs.
    Standard units:
        Angles - degrees
        Distances - kilometers
    
    Input:
        name - name of the DA
        lat - the latitude of the point of reference [Differs for slanted and slaved]
                Also, the latitude needs to be in degrees.
        lon - the longitude of the point of reference [Differs for slanted and slaved]
                Also, the longitude needs to be in degrees.
        width - The full width of the rectangle.  Needs to be in km
        height - The full height of the rectangle.  Needs to be in km
        orientation - the orientation of the rectangle about the point of reference
                Also, orientation needs to be in degrees
    Kwargs:
        N/A
    
    Notes:
        For setting or getting in this class, the units keyword should be used
        if you are giving or wanting values in different units than the standard.
    '''
    def __init__(self,name,lat=0,lon=0,width=0,height=0,orientation=0):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.width = width
        self.height = height
        self.orientation = orientation
    
    def setLat(self,lat,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            lat = np.rad2deg(lat)
        self.lat = lat
    
    def setLon(self,lon,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            lon = np.rad2deg(lon)
        self.lon = lon
    
    def setWidth(self,width,**kwargs):
        units = kwargs.get('units','km')
        if units != 'km':
            width = convertUnit(width, units, 'km')
        self.width = width
    
    def setHeight(self,height,**kwargs):
        units = kwargs.get('units','km')
        if units != 'km':
            height = convertUnit(height, units, 'km')
        self.height = height
    
    def setOrientation(self,orientation,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            orientation = np.rad2deg(orientation)
        self.orientation = orientation
    
    def getName(self,**kwargs):
        upper = kwargs.get('upper',False)
        lower = kwargs.get('lower',False)
        if upper:
            return self.name.upper()
        elif lower:
            return self.name.lower()
        else:
            return self.name
    
    def getLat(self,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            return np.deg2rad(self.lat)
        return self.lat
    
    def getLon(self,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            return np.deg2rad(self.lon)
        return self.lon
    
    def getWidth(self,**kwargs):
        units = kwargs.get('units','km')
        if units != 'km':
            return convertUnit(self.width,'km',units)
        return self.width
    
    def getHeight(self,**kwargs):
        units = kwargs.get('units','km')
        if units != 'km':
            return convertUnit(self.height, 'km', units)
        return self.height
    
    def getOrientation(self,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            return np.deg2rad(self.orienation)
        return self.orientation
        
class Polygon_DA(object):
    '''
    This is a standard template for Polygon DAs.
    Standard units:
        Angles - degrees
    
    Input:
        name - name of the DA
        lats - iterable of latitudes.  The index should match with the corresponding longitude index
        lons - iterable of longitudes.  The index should match with the corresponding latitude index
    Kwargs:
        N/A
    
    Notes:
        For setting or getting in this class, the units keyword should be used
        if you are giving or wanting values in different units.
    '''
    def __init__(self,name,lats=[],lons=[]):
        self.name = name
        self.lats = np.array(lats)
        self.lons = np.array(lons)
        
    def setLats(self,lats,**kwargs):
        units = kwargs.get('units','deg')
        lats = np.array(lats)
        if units in radianalias:
            lats = np.rad2deg(lats)
        self.lats = lats
    
    def setLons(self,lons,**kwargs):
        units = kwargs.get('units','deg')
        lons = np.array(lons)
        if units in radianalias:
            lons = np.rad2deg(lons)
        self.lons = lons
    
    def getName(self,**kwargs):
        upper = kwargs.get('upper',False)
        lower = kwargs.get('lower',False)
        if upper:
            return self.name.upper()
        elif lower:
            return self.name.lower()
        else:
            return self.name
    
    def getLats(self,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            return np.deg2rad(self.lats)
        return self.lats
    
    def getLons(self,**kwargs):
        units = kwargs.get('units','deg')
        if units in radianalias:
            return np.deg2rad(self.lons)
        return self.lons
    
if __name__ == '__main__':
    x = Circle_DA('thisone')
    x.setLat(50)
    x.setLon(50)
    x.setRadius(100,units='NM')
    print(x.getLat())
    
    y = Circle_DA('thatone')
    
    z = {'thatone':y,'thisone':x}
    import pickle
    pickle.dump(x, open('/home/marlowcj/Desktop/Aegis_Japan.pkl','wb'))
#     import dill
#     dill.dump(z, open('/home/klinetry/Desktop/DA.da','wb'))