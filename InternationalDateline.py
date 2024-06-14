#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 10:54:08 2019

@author: klinetry
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:07:44 2019

@author: Jordan
"""
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import split,nearest_points
# from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from itertools import combinations,product
from pyproj import Geod

geod = Geod(ellps='WGS84')

def orderPolygon(xs,ys):
    xs = np.array(xs)
    ys = np.array(ys)
    angles = np.arctan2(ys-np.average(ys),xs-np.average(xs))
    array = np.array([xs,ys,angles])
    array = array[:,array[2].argsort()]
    return array[0],array[1]

def circleLatLons(lat, lon, radiuskm,n_samples=180):
    """
    Return the coordinates of a geodetic circle of a given
    radius about a lon/lat point.

    Radius is in meters in the geodetic's coordinate system.

    """
    radius = radiuskm * 1000.
    lons, lats, back_azim = geod.fwd(np.repeat(lon, n_samples),
                                     np.repeat(lat, n_samples),
                                     np.linspace(0, 360, n_samples),
                                     np.repeat(radius, n_samples),
                                     radians=False,
                                     )
    return lats, lons

def ellipseLatLons(lat,lon,major,minor,orientation,n_samples=180,units='km'):
    """
    Return the coordinates of a geodetic ellipse of a given
    radius about a lon/lat point.  Radius is calculated using
    the polar coordinate equation for an ellipse, utilizing major
    and minor axis.

    Radius is in meters in the geodetic's coordinate system.

    """
    if units == 'km':
        major = major*1000.
        minor = minor*1000.
    orientation = np.deg2rad(orientation)
    bearing = np.linspace(0,2*np.pi,n_samples)
    radius = np.true_divide(np.multiply(major,minor),np.sqrt(np.square(np.multiply(major,np.sin(bearing+orientation))) + np.square(np.multiply(minor,np.cos(bearing+orientation)))))

    lons, lats, back_azim = geod.fwd(np.repeat(lon, n_samples),
                                     np.repeat(lat, n_samples),
                                     np.rad2deg(bearing),
                                     radius,
                                     radians=False,
                                     )
    return lats,lons

def rectangleLatLons(lat,lon,height,width,orientation,**kwargs):
    units = kwargs.get('units','km')
    slave = kwargs.get('slave',False)
    if units == 'km':
        width = width*1000.
        height = height*1000.
#    orientation = np.deg2rad(orientation)
    if slave:
        TRangle = np.rad2deg(np.arctan2(height,width))
        bearings = np.array([90.+TRangle,90.-TRangle,270.+TRangle,270.-TRangle])-orientation
        distance = np.linalg.norm([height/2.,width/2.])
        lons,lats,back_azim = geod.fwd(np.repeat(lon,4),
                                       np.repeat(lat,4),
                                       bearings,
                                       np.repeat(distance,4),
                                       radians=False,
                                       )
    else:
        TRangle = np.rad2deg(np.arctan(2*width/(height)))
        bearings = np.array([90.,90.-TRangle,270.+TRangle,270.])-orientation
        outerD = np.linalg.norm([height,width/2.])
        distance = np.array([width/2.,outerD,outerD,width/2.])
        lons,lats,back_azim = geod.fwd(np.repeat(lon,4),
                                       np.repeat(lat,4),
                                       bearings,
                                       distance,
                                       radians=False,
                                       )
    return lats,lons

def handle_InternationalDateline(iLat,iLon):
    '''
    iLat and iLon must be iterables
    if there are no points that lie around the ID then I'll return the originals in a list [[lats]],[[lons]] that make a closed polygon
    if there are points that lie around the ID then I'll return a list of lists of separated points with new points on the dateline
    [[lats on left][lats on right]],[[lons on left],[[lons on right]]
    This keeps the output consistent no matter what happens in the function
    '''
    #Just convert to numpy arrays off the wiffle ball bat
    nlats = np.array(list(iLat))
    nlons = np.array(list(iLon))

    #Correct the points
    nlons = np.where(nlons >= 0 ,nlons,nlons+360)
    nlons = np.where(nlons <= 360,nlons,nlons-360)
    #Return the points
    nlons = np.where(nlons <= 180,nlons,nlons-360)
    rlats,rlons = [],[]
    if max(nlons)-min(nlons) > 300:
        posidx = nlons > 0
        negidx = nlons < 0
        poslons = nlons[posidx]
        poslats = nlats[posidx]
        neglons = nlons[negidx]
        neglats = nlats[negidx]
        if posidx.any():
            poslons -= 180
        if negidx.any():
            neglons += 180
        alllons = np.append(poslons,neglons)
        alllats = np.append(poslats,neglats)
        alllons,alllats = orderPolygon(alllons,alllats)
        poly = Polygon(list(zip(alllons,alllats)))
        line = LineString([(0,-5000),(0,5000)])
        spoly = split(poly,line)
        for polygon in spoly:
            xx,yy = np.array(polygon.exterior.xy[0]),np.array(polygon.exterior.xy[1])
            posidx = xx > 0
            negidx = xx < 0
            if posidx.any():
                xx += -180
            if negidx.any():
                xx -= -180
            rlats.append(yy)
            rlons.append(xx)
        return rlats,rlons
    else:
        nlons,nlats = orderPolygon(nlons,nlats)
        polygon = Polygon(list(zip(nlons,nlats)))
        xx,yy = np.array(polygon.exterior.xy[0]),np.array(polygon.exterior.xy[1])
        rlats.append(yy)
        rlons.append(xx)
        return rlats,rlons

def LatLon2MultiPolygon(rlats,rlons):
    '''
    This takes the direct output of handle_InternationalDateline

    This will just make sure that every polygon that gets split up by the handle_InternationalDateline
    function will stay in a single collection because these should be identified
    as single objects
    Example:  Hawaii DA will be split into at least 2 parts, but it is considered
    a single DA.

    FAQ:  What if I have the polygons already made?  You can pass a list of
    polygons that should be regarded as a single object into the

    Polygon2MultiPolygon

    function.
    '''
    polylist = []
    for x,y in zip(rlons,rlats):
        polylist.append(Polygon(list(zip(x,y))))
    return MultiPolygon(polylist)

def Polygon2MultiPolygon(polylist):
    '''
    This will take a list of polygons and just slap them into a multipolygon.
    This is primarily used to find the closest points between different DAs.
    For instance if you have a DA in Hawaii that is split into 2 or more sections
    you don't want to compare the individual sections to each other, because they
    are in essence considered a single DA.
    '''
    return MultiPolygon(polylist)

def getClosesestPolygonPoints(poly1,poly2):
    '''
    This function will find the line that connects two polygon objects at minimum
    distance between them.  After this line is found, the endpoints are returned.
    The first endpoint is a point that lies on the first polygon's perimeter and the second
    is the endpoint that lies on the second polygon's perimeter.

    This function will take in either two single Polygon objects or two Multipolygon
    objects.  If the object is a Polygon object it will be converted to a Multipolygon
    for general purposes.

    Input:  poly1 - Polygon or MultiPolygon object
            poly2 - Polygon or MultiPolygon object

    Output: p1,p2 - These are of the form ((x1,y1),(x2,y2)) and correspond
                    to the endpoints of the shortest distance line that connects
                    the two polygon objects.
    '''
    if isinstance(poly1,Polygon):
        poly1 = MultiPolygon([poly1])
    if isinstance(poly2,Polygon):
        poly2 = MultiPolygon([poly2])
    distances = []
    coords = []
    for poly in poly1:
        for otherpoly in poly2:
            distances.append(poly.distance(otherpoly))
            coords.append(nearest_points(poly,otherpoly))
    index = distances.index(min(distances))
    p1 = coords[index][0].coords.xy
    p2 = coords[index][1].coords.xy
    return p1,p2

def LatLonDistance(p1,p2):
    '''
    Uses WGS84 and pyproj

    This function takes in two Lon,Lat points and returns the WGS84 distance
    between them in km.

    Input:  p1 (lon,lat) or in math terms (x,y)
            p2 (lon,lat) or in math terms (x,y)
    Output: distance in kilometers
    '''
    lon1,lat1 = p1
    lon2,lat2 = p2
    azimuth1, azimuth2, distance = geod.inv(lon1, lat1, lon2, lat2)
    if isinstance(distance,float):
        return np.divide(distance,1000.)
    else:
        return np.divide(distance,1000.)[0]

def minDistancePolygons(poly1,poly2,**kwargs):
    '''
    Just a wrapper to get the minimun distance in km between two polygons
    '''
    show = kwargs.pop('show',False)
    p1,p2 = getClosesestPolygonPoints(poly1,poly2)
    if show:
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]])
    return LatLonDistance(p1,p2)

def BulkMinDistancePolygons(mpolys,**kwargs):
    '''
    This function wants a list or dict of multipolygons.  It will take the list
    and find all possible unique combinations of 2 polygons that can be obtained and
    find the minimum distance between then 2 polygons along with some other characteristics.

    Input:  mpolys - A list or dictionary of Multipolygon objects

    Output: if mpolys is a list- Returns a list of the following type
                                [polygon1,polygon2,bool,bool]
            if mpolys is a dict- Returns a list of the following type
                                [Dict Key1, Dict Key2, bool, bool]
            where the first bool value tells you if the distance is less than
            the specified threshold

            the second bool value tells you if the two polygons intersect

    The dict version allows the user to keep track of named polygons, probably
    the best way to use this function for record keeping.
    '''
    threshold = kwargs.get('thresh',50)
    show = kwargs.pop('show',False)
    if isinstance(mpolys,dict):
        useID = True
    elif isinstance(mpolys,list):
        useID = False
    else:
        print(("The mpolys should be either a list or dict, but you passed {}"
              .format(type(mpolys))))
        return 0
    passfail = {'Poly1':[],'Poly2':[],'Within {} km'.format(threshold):[],'Intersect':[]}
    if not useID:
        combs = combinations(mpolys,2)
        for comb in combs:
            d = minDistancePolygons(comb[0],comb[1],show=show)
            passfail['Poly1'].append(comb[0])
            passfail['Poly2'].append(comb[1])
            passfail['Within {} km'.format(threshold)].append(d<threshold)
            passfail['Intersect'].append(d==0)
    else:
        combs = combinations(list(mpolys.keys()),2)
        for comb in combs:
            d = minDistancePolygons(mpolys[comb[0]],mpolys[comb[1]],show=show)
            passfail['Poly1'].append(comb[0])
            passfail['Poly2'].append(comb[1])
            passfail['Within {} km'.format(threshold)].append(d<threshold)
            passfail['Intersect'].append(d==0)
    return pd.DataFrame(passfail)

def BulkMinDistancePolygonsCompare(mpolys,mpolys1,**kwargs):
    '''
    This function wants two lists or dicts of multipolygons.  It will take the list
    and find all possible unique combinations of the 2 iterables that can be obtained and
    find the minimum distance between then 2 polygons along with some other characteristics.
    This essentially bypasses comparing polygons within the same dictionary with one another.

    Input:  mpolys - A list or dictionary of Multipolygon objects
            mpolys1 - A list or dictionary of Multipolygon objects

    Output: if mpolys is a list- Returns a list of the following type
                                [polygon1,polygon2,bool,bool]
            if mpolys is a dict- Returns a list of the following type
                                [Dict Key1, Dict Key2, bool, bool]
            where the first bool value tells you if the distance is less than
            the specified threshold

            the second bool value tells you if the two polygons intersect

    The dict version allows the user to keep track of named polygons, probably
    the best way to use this function for record keeping.
    '''
    threshold = kwargs.get('thresh',50)
    show = kwargs.pop('show',False)
    if isinstance(mpolys,dict):
        useID = True
    elif isinstance(mpolys,list):
        useID = False
    else:
        print(("The mpolys should be either a list or dict, but you passed {}"
              .format(type(mpolys))))
        return pd.DataFrame()
    if isinstance(mpolys1,dict):
        useID = True
    elif isinstance(mpolys1,list):
        useID = False
    else:
        print(("The mpolys should be either a list or dict, but you passed {}"
              .format(type(mpolys1))))
        return pd.DataFrame()
    
    passfail = {'Poly1':[],'Poly2':[],'Within {} km'.format(threshold):[],'Intersect':[]}
    if not useID:
        products = product(mpolys,mpolys1)
        for prod in products:
            d = minDistancePolygons(prod[0],prod[1],show=show)
            passfail['Poly1'].append(prod[0])
            passfail['Poly2'].append(prod[1])
            passfail['Within {} km'.format(threshold)].append(d<threshold)
            passfail['Intersect'].append(d==0)
    else:
        products = product(list(mpolys.keys()),list(mpolys1.keys()))
        for prod in products:
            d = minDistancePolygons(mpolys[prod[0]],mpolys1[prod[1]],show=show)
            passfail['Poly1'].append(prod[0])
            passfail['Poly2'].append(prod[1])
            passfail['Within {} km'.format(threshold)].append(d<threshold)
            passfail['Intersect'].append(d==0)
    return pd.DataFrame(passfail)

def unitTest(numpolys=10,**kwargs):
    nri = np.random.randint
    show = kwargs.pop('show',False)
    dictopolys = {}
    dictopolys1 = {}
    for i in range(numpolys):
#        lats,lons = ellipseLatLons(nri(-80,80),nri(-179,179),nri(500,2000),nri(500,2000),0)
#        lats,lons = ellipseLatLons(0,0,1000,500,20)
        lats,lons = rectangleLatLons(0.,0.,50.,50.,0,slave=False)
        y,x = handle_InternationalDateline(lats,lons)
        poly = LatLon2MultiPolygon(y,x)
        if not i%2:
            dictopolys['poly{}'.format(i)] = poly
        if i % 2:
            dictopolys1['poly{}'.format(i)] = poly
        if show:
            for polygon in poly:
                xx,yy = polygon.exterior.coords.xy
                plt.plot(xx,yy)
    result = BulkMinDistancePolygonsCompare(dictopolys,dictopolys1,show=show)
    return result


if __name__ == '__main__':
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
#    #This will take the lats and lons from handle_internationalDateline
#    #and make it into a poly or multipolygon and find the coordinates
#    #of the shortest distance between the two polygons
#    lats,lons = ellipseLatLons(80,180,500,200,10)
#    y,x = handle_InternationalDateline(lats,lons)
#    poly1 = LatLon2MultiPolygon(y,x)
#
#    lats1,lons1 = ellipseLatLons(0,180,600,3200,0)
#    y,x = handle_InternationalDateline(lats1,lons1)
#    poly2 = LatLon2MultiPolygon(y,x)
#
#    lats,lons = ellipseLatLons(80,20,500,500,0)
#    y,x = handle_InternationalDateline(lats,lons)
#    poly3 = LatLon2MultiPolygon(y,x)
#
#    lats,lons = ellipseLatLons(35,30,4500,6500,0)
#    y,x = handle_InternationalDateline(lats,lons)
#    poly4 = LatLon2MultiPolygon(y,x)
#
#    dictopolys = dict(ArcticPoly=poly1,PacificPoly=poly2,ArcticPoly2=poly3,AtlanticPoly=poly4)
#
#    results = BulkMinDistancePolygons(dictopolys)
#    print results
#    distance = minDistancePolygons(poly1,poly2)
#    print distance
#
#    ##Just for plotting purposes
#    p1,p2 = getClosesestPolygonPoints(poly1,poly2)
#    for poly in poly1:
#        x,y = poly.exterior.coords.xy
#        plt.plot(x,y)
#    for poly in poly2:
#        x,y = poly.exterior.coords.xy
#        plt.plot(x,y)
#    for poly in poly3:
#        x,y = poly.exterior.coords.xy
#        plt.plot(x,y)
#    for poly in poly4:
#        x,y = poly.exterior.coords.xy
#        plt.plot(x,y)
#    result = unitTest(10,show=True)
    results = unitTest(1,show=True)
#    import timeit
#    import functools
#    rmins = []
#    rmaxs = []
#    maxpolys = 100
#    for i in range(maxpolys):
#        t = timeit.Timer(functools.partial(unitTest,i))
#        r = t.repeat(repeat=3,number=1)
#        rmins.append(min(r))
#        rmaxs.append(max(r))
#        plt.scatter(i,min(r),color='g')
#        plt.scatter(i,max(r),color='r')
#    from scipy.optimize import curve_fit
#    def poly_fit(x, a, b, c):
##        return a*np.exp(-b*x) + c
#        return a*x**2 + b+c
#
#    x = np.arange(1,len(rmins))
#    y1 = np.array(rmins[:-1])
#    y2 = np.array(rmaxs[:-1])
#    fitting_parameters, covariance = curve_fit(poly_fit, x, y1,maxfev=10000)
#    fitting_parameters1,covariance1 = curve_fit(poly_fit, x, y2,maxfev=10000)
#    a, b, c = fitting_parameters
#    a1,b1,c1 = fitting_parameters1
#    print(fitting_parameters)
#    print(fitting_parameters1)
#
#    next_x = maxpolys
#    next_ymins = poly_fit(next_x, a, b, c)
#    next_ymaxs = poly_fit(next_x, a1,b1,c1)
#    npfit = np.vectorize(poly_fit)
#    fitys = npfit(x,a,b,c)
#    plt.plot(fitys,'purple')
#    plt.plot([maxpolys],[next_ymins], 'ro',ms=20)
#    plt.plot([maxpolys],[next_ymaxs],'g',ms=20)


#    plt.gca().set_aspect('equal')
    plt.show()