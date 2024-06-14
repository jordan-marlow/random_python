#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 23:07:06 2021

@author: marlowcj
"""
import numpy as np
import pandas as pd
earth_angular_velocity = 7.2921151e-5 #rad/s
earth_equitorial_radius = 6378137.0 #m
earth_polar_radius = 6356752.31424518 #m
earth_first_eccentricity = 0.08181919084262032 # sqrt ((earth_equitorial_radius^2 - earth_polar_radius^2)/earth_equitorial_radius^2)
earth_second_eccentricity = 0.0820944379496945 # sqrt ((earth_equitorial_radius^2 - earth_polar_radius^2)/earth_polar_radius^2)
ese2 = earth_second_eccentricity**2
efe2 = earth_first_eccentricity**2
eer2_epr2 = earth_equitorial_radius**2 - earth_polar_radius**2
one_efe2 = 1-efe2
one_ese2 = 1-ese2
one_over_eer = 1/earth_equitorial_radius
one_over_eer2 = 1/earth_equitorial_radius**2
l = efe2/2
l2 = l**2
one_efe2_over_eer2 = one_efe2/earth_equitorial_radius**2
one_efe2_over_epr = one_efe2/earth_polar_radius
one_over_cuberoot2 = .5**(1/3)

ndtypes = (np.intp, np.int8, np.intc, np.int32, np.int16, np.floating, np.int0, np.integer, np.int64, np.float64,
           np.int_, np.float16, np.float32, np.float_,int,float)
preferred_dtype = np.float64

#%% Direct Conversions
def ecef2eci(x,y,z,t,*args,**kwargs):
    '''
    Converts ECEF to ECI Coordinate System
    Position,Velocity, and Acceleration distance units do not matter
    for this conversion.

    e.g. ecef in {km,km/s,km/s^2} -> eci {km,km/s,km/s^2}
         ecef in {m,m/s,m/s^2}    -> eci {m,m/s,m/s^2}

    Parameters
    ----------
    x : Numpy Array, Pandas Series
        The X Position of the ECEF coordinates
    y : Numpy Array, Pandas Series
        The Y Position of the ECEF coordinates
    z : Numpy Array, Pandas Series
        The Y Position of the ECEF coordinates
    t : Numpy Array, Pandas Series
        Time difference between cooincident time in seconds
    *args : Numpy Array, Pandas Series
        Additional args passed must be in sets of 3.
            If you want Velocities from ecef2eci you must give
                vx,vy,vz in ECEF
            If you want Accelerations from ecef2eci you must give
                vx,vy,vz,ax,ay,az in ECEF
            This is because for speed velocity calculations depend on calculated
            positions and accelerations depend on calculated velocities and positions.
            There are of course ways to do this with the original data, however,
            that requires many more math operations than the current method.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    Pandas.DataFrame
        If only x,y,z, and t given:
            returns DataFrame with ECI x,y,z
        If x,y,z,t,vx,vy, and vz given:
            return DataFrame with ECI x,y,z,vx,vy,vz
        If x,y,z,t,vx,vy,vz,ax,ay, and az given:
            return DataFrame with ECI x,y,z,vx,vy,vz,ax,ay,az

    '''
    iunits = kwargs.get('iunits','m')
    ounits = kwargs.get('ounits','m')
    d = {}
    if isinstance(x,ndtypes):
        x = np.array([x],dtype=preferred_dtype)
        y = np.array([y],dtype=preferred_dtype)
        z = np.array([z],dtype=preferred_dtype)
        t = np.array([t],dtype=preferred_dtype)
    else:
        x = np.array(x,dtype=preferred_dtype)
        y = np.array(y,dtype=preferred_dtype)
        z = np.array(z,dtype=preferred_dtype)
        t = np.array(t,dtype=preferred_dtype)

    rsize = 3 + len(args)
    rsize = int(rsize/3)*3


    #How much the earth as moved in some time t
    w = earth_angular_velocity
    angular_displacement = w*t
    cos_wt = np.cos(angular_displacement)
    sin_wt = np.sin(angular_displacement)

    ecix = cos_wt * x - sin_wt * y
    eciy = sin_wt * x + cos_wt * y

    d.update({'PosXi':ecix,'PosYi':eciy,'PosZi':z})

    if rsize > 3:
        vx = args[0]
        vy = args[1]
        vz = args[2]
        if isinstance(vx,ndtypes):
            vx = np.array([vx]*t.shape[0],dtype=preferred_dtype)
            vy = np.array([vy]*t.shape[0],dtype=preferred_dtype)
            vz = np.array([vz]*t.shape[0],dtype=preferred_dtype)
        else:
            vx = np.array(vx,dtype=preferred_dtype)
            vy = np.array(vy,dtype=preferred_dtype)
            vz = np.array(vz,dtype=preferred_dtype)
        ecivx = cos_wt*vx - sin_wt*vy - eciy*w
        ecivy = cos_wt*vy + sin_wt*vx + ecix*w

        d.update({'VelXi':ecivx,'VelYi':ecivy,'VelZi':vz})

    if rsize == 9:
        w2 = w**2
        ax = args[3]
        ay = args[4]
        az = args[5]
        if isinstance(ax,ndtypes):
            ax = np.array([ax]*t.shape[0],dtype=preferred_dtype)
            ay = np.array([ay]*t.shape[0],dtype=preferred_dtype)
            az = np.array([az]*t.shape[0],dtype=preferred_dtype)
        else:
            ax = np.array(ax,dtype=preferred_dtype)
            ay = np.array(ay,dtype=preferred_dtype)
            az = np.array(az,dtype=preferred_dtype)

        eciax = cos_wt*ax - sin_wt*ay - 2*ecivy*w + ecix*w2
        eciay = cos_wt*ay + sin_wt*ax + 2*ecivx*w + eciy*w2

        d.update({'AccXi':eciax,'AccYi':eciay,'AccZi':az})
    df = pd.DataFrame(d)
    if ounits == iunits:
        return df
    elif iunits == 'm' and ounits == 'km':
        return df*.001
    # elif iunits == 'km' and ounits == 'm':
    else:
        return df*1000


def eci2ecef(x,y,z,t,*args,**kwargs):
    '''
    Converts ECI to ECEF Coordinate System
    Position,Velocity, and Acceleration distance units do not matter
    for this conversion.

    e.g. eci in {km,km/s,km/s^2} -> ecef {km,km/s,km/s^2}
         eci in {m,m/s,m/s^2}    -> ecef {m,m/s,m/s^2}

    Parameters
    ----------
    x : Numpy Array, Pandas Series
        The X Position of the ECI coordinates
    y : Numpy Array, Pandas Series
        The Y Position of the ECI coordinates
    z : Numpy Array, Pandas Series
        The Y Position of the ECI coordinates
    t : Numpy Array, Pandas Series
        Time difference between cooincident time in seconds
    *args : Numpy Array, Pandas Series
        Additional args passed must be in sets of 3.
            If you want Velocities from eci2ecef you must give
                vx,vy,vz in ECEF
            If you want Accelerations from eci2ecef you must give
                vx,vy,vz,ax,ay,az in ECEF
            This is because for speed velocity calculations depend on calculated
            positions and accelerations depend on calculated velocities and positions.
            There are of course ways to do this with the original data, however,
            that requires many more math operations than the current method.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    Pandas.DataFrame
        If only x,y,z, and t given:
            returns DataFrame with ECEF x,y,z
        If x,y,z,t,vx,vy, and vz given:
            return DataFrame with ECEF x,y,z,vx,vy,vz
        If x,y,z,t,vx,vy,vz,ax,ay, and az given:
            return DataFrame with ECEF x,y,z,vx,vy,vz,ax,ay,az

    '''
    iunits = kwargs.get('iunits','m')
    ounits = kwargs.get('ounits','m')
    d = {}
    if isinstance(x,ndtypes):
        x = np.array([x],dtype=preferred_dtype)
        y = np.array([y],dtype=preferred_dtype)
        z = np.array([z],dtype=preferred_dtype)
        t = np.array([t],dtype=preferred_dtype)
    else:
        x = np.array(x,dtype=preferred_dtype)
        y = np.array(y,dtype=preferred_dtype)
        z = np.array(z,dtype=preferred_dtype)
        t = np.array(t,dtype=preferred_dtype)

    rsize = 3 + len(args)
    rsize = int(rsize/3)*3
    w = earth_angular_velocity
    angular_displacement = w*t
    cos_wt = np.cos(angular_displacement)
    sin_wt = np.sin(angular_displacement)

    ecefx = cos_wt*x + sin_wt*y
    ecefy = cos_wt*y - sin_wt*x

    d.update({'PosX':ecefx,'PosY':ecefy,'PosZ':z})

    if rsize > 3:
        vx = args[0]
        vy = args[1]
        vz = args[2]
        if isinstance(vx,ndtypes):
            vx = np.array([vx]*t.shape[0],dtype=preferred_dtype)
            vy = np.array([vy]*t.shape[0],dtype=preferred_dtype)
            vz = np.array([vz]*t.shape[0],dtype=preferred_dtype)
        else:
            vx = np.array(vx,dtype=preferred_dtype)
            vy = np.array(vy,dtype=preferred_dtype)
            vz = np.array(vz,dtype=preferred_dtype)

        ecefvx = cos_wt*vx + sin_wt*vy + ecefy*w
        ecefvy = cos_wt*vy - sin_wt*vx - ecefx*w

        d.update({'VelX':ecefvx,'VelY':ecefvy,'VelZ':vz})

    if rsize == 9:
        w2 = w**2
        ax = args[3]
        ay = args[4]
        az = args[5]
        if isinstance(ax,ndtypes):
            ax = np.array([ax]*t.shape[0],dtype=preferred_dtype)
            ay = np.array([ay]*t.shape[0],dtype=preferred_dtype)
            az = np.array([az]*t.shape[0],dtype=preferred_dtype)
        else:
            ax = np.array(ax,dtype=preferred_dtype)
            ay = np.array(ay,dtype=preferred_dtype)
            az = np.array(az,dtype=preferred_dtype)

        ecefax = cos_wt*ax + sin_wt*ay + 2*ecefvy*w - ecefx*w2
        ecefay = cos_wt*ay - sin_wt*ax - 2*ecefvx*w - ecefy*w2
        d.update({'AccX':ecefax,'AccY':ecefay,'AccZ':az})
    df = pd.DataFrame(d)
    if iunits == ounits:
        return df
    elif iunits == 'm' and ounits == 'km':
        return df/1000
    else:
        return df*1000


def ecef2enu(x,y,z,lat0,lon0,alt0,*args,**kwargs):
    '''
    Convert ECEF coordinates to ENU (East North Up).
    Position,Velocity, and Acceleration distance units do not matter
    for this conversion.

    e.g. ecef in {km,km/s,km/s^2} -> enu {km,km/s,km/s^2}
         ecef in {m,m/s,m/s^2}    -> enu {m,m/s,m/s^2}
    Parameters
    ----------
    x : Numpy Array, Pandas Series
        The X Position of the ECEF coordinates
    y : Numpy Array, Pandas Series
        The Y Position of the ECEF coordinates
    z : Numpy Array, Pandas Series
        The Y Position of the ECEF coordinates
    t : Numpy Array, Pandas Series
        Time difference between cooincident time in seconds
    *args : Numpy Array, Pandas Series
        Additional args passed must be in sets of 3.
            If you want Velocities from ecef2enu you must give
                vx,vy,vz in ECEF
            If you want Accelerations from ecef2enu you must give
                vx,vy,vz,ax,ay,az in ECEF
            This is because for speed velocity calculations depend on calculated
            positions and accelerations depend on calculated velocities and positions.
            There are of course ways to do this with the original data, however,
            that requires many more math operations than the current method.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    Pandas.DataFrame
        If only x,y,z, and t given:
            returns DataFrame with ENU east, north, up
        If x,y,z,t,vx,vy, and vz given:
            return DataFrame with ENU east, north, up, veast, vnorth, vup
        If x,y,z,t,vx,vy,vz,ax,ay, and az given:
            return DataFrame with ENU east, north, up, veast, vnorth, vup, aeast, anorth, aup

    '''
    d = {}
    deg = kwargs.get('deg',True)
    iunits = kwargs.get('m')
    ounits = kwargs.get('m')

    if isinstance(x,ndtypes):
        x = np.array([x],dtype=preferred_dtype)
        y = np.array([y],dtype=preferred_dtype)
        z = np.array([z],dtype=preferred_dtype)
    else:
        x = np.array(x,dtype=preferred_dtype)
        y = np.array(y,dtype=preferred_dtype)
        z = np.array(z,dtype=preferred_dtype)

    if deg:
        lat0 = np.deg2rad(lat0)
        lon0 = np.deg2rad(lon0)


    rsize = 3 + len(args)
    rsize = int(rsize/3)*3

    ecef0 = geodetic2ecef(lat0, lon0, alt0, deg=False)
    xi = x-ecef0.iloc[:,0].values
    yi = y-ecef0.iloc[:,1].values
    zi = z-ecef0.iloc[:,2].values
    coslon = np.cos(lon0)
    coslat = np.cos(lat0)
    sinlon = np.sin(lon0)
    sinlat = np.sin(lat0)

    east = coslon*yi - sinlon*xi
    north = coslat*zi - coslon*sinlat*xi - sinlat*sinlon*yi
    up = coslat*coslon*xi + coslat*sinlon*yi + sinlat*zi

    d.update({'East':east,'North':north,'Up':up})

    if rsize > 3:
        ecefvx = args[0]
        ecefvy = args[1]
        ecefvz = args[2]
        if isinstance(ecefvx,ndtypes):
            ecefvx = np.array([ecefvx]*x.shape[0],dtype=preferred_dtype)
            ecefvy = np.array([ecefvy]*x.shape[0],dtype=preferred_dtype)
            ecefvz = np.array([ecefvz]*x.shape[0],dtype=preferred_dtype)
        else:
            ecefvx = np.array(ecefvx,dtype=preferred_dtype)
            ecefvy = np.array(ecefvy,dtype=preferred_dtype)
            ecefvz = np.array(ecefvz,dtype=preferred_dtype)

        veast = -sinlon*ecefvx + coslon*ecefvy
        vnorth = -coslon*sinlat*ecefvx - sinlat*sinlon*ecefvy + coslat*ecefvz
        vup = coslat*coslon*ecefvx + coslat*sinlon*ecefvy + sinlat*ecefvz

        d.update({'EastRate':veast,'NorthRate':vnorth,'UpRate':vup})

    if rsize == 9:
        ecefax = args[3]
        ecefay = args[4]
        ecefaz = args[5]
        if isinstance(ecefax,ndtypes):
            ecefax = np.array([ecefax]*x.shape[0],dtype=preferred_dtype)
            ecefay = np.array([ecefay]*x.shape[0],dtype=preferred_dtype)
            ecefaz = np.array([ecefaz]*x.shape[0],dtype=preferred_dtype)
        else:
            ecefax = np.array(ecefax,dtype=preferred_dtype)
            ecefay = np.array(ecefay,dtype=preferred_dtype)
            ecefaz = np.array(ecefaz,dtype=preferred_dtype)

        aeast = -sinlon*ecefax + coslon*ecefay
        anorth = coslat*ecefaz - coslon*ecefax*sinlat - ecefay*sinlat*sinlon
        aup = coslat*coslon*ecefax + coslat*ecefay*sinlon + ecefaz*sinlat

        d.update({'EastAcc':aeast,'NorthAcc':anorth,'UpAcc':aup})
    return pd.DataFrame(d)

def enu2ecef(east,north,up,lat0,lon0,alt0,*args,**kwargs):
    d = {}
    deg = kwargs.get('deg',True)
    if isinstance(east,ndtypes):
        east = np.array([east],dtype=preferred_dtype)
        north = np.array([north],dtype=preferred_dtype)
        up = np.array([up],dtype=preferred_dtype)
    else:
        east = np.array(east,dtype=preferred_dtype)
        north = np.array(north,dtype=preferred_dtype)
        up = np.array(up,dtype=preferred_dtype)

    if deg:
        lat0 = np.deg2rad(lat0)
        lon0 = np.deg2rad(lon0)

    rsize = 3 + len(args)
    rsize = int(rsize/3)*3
    ecef0 = geodetic2ecef(lat0, lon0, alt0, deg=False)
    x0 = ecef0.iloc[:,0].values
    y0 = ecef0.iloc[:,1].values
    z0 = ecef0.iloc[:,2].values
    coslon = np.cos(lon0)
    coslat = np.cos(lat0)
    sinlon = np.sin(lon0)
    sinlat = np.sin(lat0)

    ecefx = coslat*coslon*up - coslon*sinlat*north - sinlon*east + x0
    ecefy = coslat*sinlon*up - sinlat*sinlon*north + coslon*east + y0
    ecefz = sinlat*up + coslat*north + z0

    d.update({'PosX':ecefx,'PosY':ecefy,'PosZ':ecefz})

    if rsize > 3:
        veast = args[0]
        vnorth = args[1]
        vup = args[2]
        if isinstance(veast,ndtypes):
            veast = np.array([veast]*east.shape[0],dtype=preferred_dtype)
            vnorth = np.array([vnorth]*east.shape[0],dtype=preferred_dtype)
            vup = np.array([vup]*east.shape[0],dtype=preferred_dtype)
        else:
            veast = np.array(veast,dtype=preferred_dtype)
            vnorth = np.array(vnorth,dtype=preferred_dtype)
            vup = np.array(vup,dtype=preferred_dtype)

        ecefvx = coslat*coslon*vup - coslon*sinlat*vnorth - sinlon*veast
        ecefvy = coslat*sinlon*vup + coslon*veast - sinlat*sinlon*vnorth
        ecefvz = coslat*vnorth + vup*sinlat

        d.update({'VelX':ecefvx,'VelY':ecefvy,'VelZ':ecefvz})

    if rsize == 9:
        aeast = args[3]
        anorth = args[4]
        aup = args[5]
        if isinstance(aeast,ndtypes):
            aeast = np.array([aeast]*east.shape[0],dtype=preferred_dtype)
            anorth = np.array([anorth]*east.shape[0],dtype=preferred_dtype)
            aup = np.array([aup]*east.shape[0],dtype=preferred_dtype)
        else:
            aeast = np.array(aeast,dtype=preferred_dtype)
            anorth = np.array(anorth,dtype=preferred_dtype)
            aup = np.array(aup,dtype=preferred_dtype)

        ecefax = coslat*coslon*aup - coslon*anorth*sinlat - aeast*sinlon
        ecefay = coslat*aup*sinlon + coslon*aeast - anorth*sinlat*sinlon
        ecefaz = coslat*anorth + aup*sinlat
        d.update({'AccX':ecefax,'AccY':ecefay,'AccZ':ecefaz})
    return pd.DataFrame(d)

def geodetic2ecef(lat,lon,alt,*args,**kwargs):
    d = {}
    deg = kwargs.get('deg',True)

    rsize = 3 + len(args)
    rsize = int(rsize/3)*3

    if isinstance(lat,ndtypes):
        lat = np.array([lat],dtype=preferred_dtype)
        lon = np.array([lon],dtype=preferred_dtype)
        alt = np.array([alt],dtype=preferred_dtype)
    else:
        lat = np.array(lat,dtype=preferred_dtype)
        lon = np.array(lon,dtype=preferred_dtype)
        alt = np.array(alt,dtype=preferred_dtype)

    if deg:
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
    sinlon = np.sin(lon)
    sinlat = np.sin(lat)
    coslon = np.cos(lon)
    coslat = np.cos(lat)

    N = earth_equitorial_radius/np.sqrt(1-efe2*sinlat*sinlat)
    NtA = (N+alt)
    const = NtA*coslat
    NetA = (N*one_efe2+alt)

    ecefx = const*coslon
    ecefy = const*sinlon
    ecefz = NetA*sinlat

    d.update({'PosX':ecefx,'PosY':ecefy,'PosZ':ecefz})

    #For something else
    # if rsize > 3:
    #     speed = args[0]
    #     heading = args[1]
    #     vertical_v = args[2]
    #     if isinstance(speed,ndtypes):
    #         speed = np.array([speed]*lat.shape[0],dtype=preferred_dtype)
    #         heading = np.array([heading]*lat.shape[0],dtype=preferred_dtype)
    #         vertical_v = np.array([vertical_v]*lat.shape[0],dtype=preferred_dtype)
    #     else:
    #         speed = np.array(speed,dtype=preferred_dtype)
    #         heading = np.array(heading,dtype=preferred_dtype)
    #         vertical_v = np.array(vertical_v,dtype=preferred_dtype)

    #     sinheading = np.sin(heading)
    #     cosheading = np.cos(heading)
    #     zenith = (np.pi/2)-lat
    #     coszenith = np.cos(zenith)
    #     sinzenith = np.sin(zenith)
    #     ecefvx = -speed*sinheading*sinlon - speed*cosheading*coslon*coszenith + vertical_v*coslon*sinzenith
    #     ecefvy = speed*sinheading*coslon - speed*cosheading*sinlon*coszenith + vertical_v*sinlon*sinzenith
    #     ecefvz = speed*coslon*sinzenith + vertical_v*coszenith

    #     d.update({'VelX':ecefvx,'VelY':ecefvy,'VelZ':ecefvz})

    if rsize > 3:
        lat_dot = args[0]
        lon_dot = args[1]
        alt_dot = args[2]

        if isinstance(lat_dot,ndtypes):
            lat_dot = np.array([lat_dot],dtype=preferred_dtype)
            lon_dot = np.array([lon_dot],dtype=preferred_dtype)
            alt_dot = np.array([alt_dot],dtype=preferred_dtype)
        else:
            lat_dot = np.array(lat_dot,dtype=preferred_dtype)
            lon_dot = np.array(lon_dot,dtype=preferred_dtype)
            alt_dot = np.array(alt_dot,dtype=preferred_dtype)

        if deg:
            lat_dot = np.deg2rad(lat_dot)
            lon_dot = np.deg2rad(lon_dot)

        N_dot = N**3 * efe2 * coslat*sinlat*lat_dot/(earth_equitorial_radius**2)
        NdtAd = N_dot+alt_dot
        const2 = NdtAd*coslat
        NdetAd = N_dot*one_efe2 + alt_dot

        vx = const2*coslon - NtA*lat_dot*sinlat*coslon - ecefy*lon_dot
        vy = const2*sinlon - NtA*lat_dot*sinlat*sinlon + ecefx*lon_dot
        vz = NdetAd*sinlat + NetA*coslat*lat_dot

        d.update({'VelX':vx,'VelY':vy,'VelZ':vz})

    if rsize == 9:
        lat_ddot = args[3]
        lon_ddot = args[4]
        alt_ddot = args[5]

        if isinstance(lat_ddot,ndtypes):
            lat_ddot = np.array([lat_ddot],dtype=preferred_dtype)
            lon_ddot = np.array([lon_ddot],dtype=preferred_dtype)
            alt_ddot = np.array([alt_ddot],dtype=preferred_dtype)
        else:
            lat_ddot = np.array(lat_ddot,dtype=preferred_dtype)
            lon_ddot = np.array(lon_ddot,dtype=preferred_dtype)
            alt_ddot = np.array(alt_ddot,dtype=preferred_dtype)

        if deg:
            lat_ddot = np.deg2rad(lat_ddot)
            lon_ddot = np.deg2rad(lon_ddot)

        N_ddot = 3*N_dot**2/N + (efe2*N**3/(earth_equitorial_radius**2))*(lat_dot**2*np.cos(2*lat) + coslat*sinlat*lat_ddot)
        NddtAdd = N_ddot + alt_ddot
        const3 = NddtAdd*coslat
        NddetAdd = N_ddot*one_efe2+alt_ddot

        ax = const3*coslon - 2*NdtAd*(lat_dot*sinlat*coslon+lon_dot*coslat*sinlon) - NtA*(lat_ddot*sinlat*coslon - 2*lon_dot*lat_dot*sinlat*sinlon) - ecefy*lon_ddot - ecefx*(lat_dot**2+lon_dot**2)
        ay = const3*sinlon - 2*NdtAd*(lat_dot*sinlat*sinlon-lon_dot*coslat*coslon) - NtA*(lat_ddot*sinlat*sinlon + 2*lon_dot*lat_dot*sinlat*coslon) + ecefx*lon_ddot - ecefy*(lat_dot**2+lon_dot**2)
        az = NddetAdd*sinlat + 2*NdetAd*lat_dot*coslat + NetA*lat_ddot*coslat - ecefz*lat_dot**2

        d.update({'AccX':ax,'AccY':ay,'AccZ':az})

    return pd.DataFrame(d)

def ecef2geodeticIter(x,y,z,**kwargs):
    '''
    U.S. Navy Iterative Method.  Considered by Carl to be the most accurate conversion,
    however it is the slowest do to iterations.

    source: https://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    deg = kwargs.get('deg',True)
    if isinstance(x,ndtypes):
        x = np.array([x],dtype=preferred_dtype)
        y = np.array([y],dtype=preferred_dtype)
        z = np.array([z],dtype=preferred_dtype)
    else:
        x = np.array(x,dtype=preferred_dtype)
        y = np.array(y,dtype=preferred_dtype)
        z = np.array(z,dtype=preferred_dtype)

    lon = np.arctan2(y,x)
    x2 = x*x
    y2 = y*y
    # z2 = z*z

    # r = np.sqrt(x2+y2+z2)
    p = np.sqrt(x2+y2)
    lat = np.arctan2(p,z) #Geocentric Lat initally
    for i in range(30):
        sinlat = np.sin(lat)
        N = earth_equitorial_radius/np.sqrt(1-efe2*sinlat*sinlat)
        h = p/np.cos(lat) - N
        lat = np.arctan2(z,p*(1-efe2*N/(N+h)))
    sinlat = np.sin(lat)
    N = earth_equitorial_radius/np.sqrt(1-efe2*sinlat*sinlat)
    alt = p/np.cos(lat) - N
    if deg:
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)
    return pd.DataFrame({'Lat':lat,'Lon':lon,'Alt':alt})

def ecef2geodeticModifiedZhu(x,y,z,**kwargs):
    '''
    This is the modified Zhu Version of ecef to geodetic.  It is faster
    and more accurate than the Zhu method especially around Latitudes of
    +/- 45.288 degrees.  This is by far the 2nd best option only to be
    dethroned by Ferrari's method in ecef2geodetic.

    source: https://hal.archives-ouvertes.fr/hal-01704943v2/document (last method mentioned)

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    deg = kwargs.get('deg',True)
    if isinstance(x,ndtypes):
        x = np.array([x],dtype=preferred_dtype)
        y = np.array([y],dtype=preferred_dtype)
        z = np.array([z],dtype=preferred_dtype)
    else:
        x = np.array(x,dtype=preferred_dtype)
        y = np.array(y,dtype=preferred_dtype)
        z = np.array(z,dtype=preferred_dtype)

    x2 = x*x
    y2 = y*y
    z2 = z*z

    w2 = x2+y2
    m = w2/earth_equitorial_radius**2
    n = z2*one_efe2_over_eer2

    M_plus_N = m+n
    M_minus_N = m-n
    p = (M_plus_N-4*l2)/6
    G = m*n*l2
    H = 2*p**3 + G
    C = (H+G+2*np.sqrt(H*G))**(1/3)*one_over_cuberoot2
    i = -(2*l2+M_plus_N)/2
    B = i/3 - C - p*p/C
    k = l2*(l2 - M_plus_N)
    t = np.sqrt(np.sqrt(B*B-k)-.5*(B+i)) - np.sign(M_minus_N)*np.sqrt(0.5*np.abs(B-i))
    con1 = 2*i*t
    con2 = 2*l*M_minus_N
    t2 = t*t
    F = t2*t2 + con1*t + con2*t + k
    DFDT = 4*t2*t + 2*con1 + con2
    dt = -F/DFDT
    common = t + dt
    u = common+l
    v = common-l
    lat = np.arctan2(z*u,np.sqrt(w2)*v)
    dw2 = w2*(1-1/u)**2
    dz2 = z2*(1-one_efe2/v)**2
    alt = np.sign(u-1)*np.sqrt(dw2 + dz2)
    lon = np.arctan2(y,x)
    if deg:
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)
    return pd.DataFrame({'Lat':lat,'Lon':lon,'Alt':alt})

def ecef2geodetic(x,y,z,*args,**kwargs):
    rsize = 3 + len(args)
    rsize = int(rsize/3)*3

    d = {}

    deg = kwargs.get('deg',True)
    tol = False
    
    if rsize == 3:
        if isinstance(x,ndtypes):
            x = np.array([x],dtype=preferred_dtype)
            y = np.array([y],dtype=preferred_dtype)
            z = np.array([z],dtype=preferred_dtype)
        else:
            x = np.array(x,dtype=preferred_dtype)
            y = np.array(y,dtype=preferred_dtype)
            z = np.array(z,dtype=preferred_dtype)
    
        rho = np.sqrt(x**2+y**2) 
        
        lon = np.arctan2(y,x)
        lat = np.arctan2(rho,z)
        alt = 0
        
        while not tol:
            old_alt = alt
            old_lat = lat
            
            R_n = earth_equitorial_radius/np.sqrt(1 - efe2 * np.sin(lat)**2)
            r0 = R_n/(R_n+alt)
            denom = rho*(1-efe2*r0)
            
            alt = rho/np.cos(lat) - R_n
            lat = np.arctan2(z,denom)
            
            idx_alt = np.all(np.abs(old_alt-alt)<1.10e-8)
            idx_lat = np.all(np.abs(old_lat-lat)<3.6e-8)
            tol = idx_alt and idx_lat
            
        
        if deg:
            lat = np.rad2deg(lat)
            lon = np.rad2deg(lon)
        d.update({'latitude':lat,'longitude':lon,'altitude':alt})
    
    if rsize == 6:
        if isinstance(x,ndtypes):
            x = np.array([x],dtype=preferred_dtype)
            y = np.array([y],dtype=preferred_dtype)
            z = np.array([z],dtype=preferred_dtype)
        else:
            x = np.array(x,dtype=preferred_dtype)
            y = np.array(y,dtype=preferred_dtype)
            z = np.array(z,dtype=preferred_dtype)
            
        x_dot = args[0]
        y_dot = args[1]
        z_dot = args[2]

        if isinstance(x_dot,ndtypes):
            x_dot = np.array([x_dot],dtype=preferred_dtype)
            y_dot = np.array([y_dot],dtype=preferred_dtype)
            z_dot = np.array([z_dot],dtype=preferred_dtype)
        else:
            x_dot = np.array(x_dot,dtype=preferred_dtype)
            y_dot = np.array(y_dot,dtype=preferred_dtype)
            z_dot = np.array(z_dot,dtype=preferred_dtype)
    
        rho = np.sqrt(x**2+y**2) 
        rho_dot = (x_dot*x + y_dot*y)/rho
        
        
        lon = np.arctan2(y,x)
        lat = np.arctan2(rho,z)
        alt = 0
        
        lon_dot = (-x_dot*y + y_dot*x)/rho**2
        lat_dot = (-z_dot*rho + rho_dot*z)/(rho**2 + z**2)
        alt_dot = 0
        
        while not tol:
            old_alt = alt
            old_lat = lat
            
            R_n = earth_equitorial_radius/np.sqrt(1 - efe2 * np.sin(lat)**2)
            r0 = R_n/(R_n+alt)
            denom = rho*(1-efe2*r0)
            
            R_n_dot = (R_n**3 * efe2 *np.sin(lat)*np.cos(lat)*lat_dot)* (1 / earth_equitorial_radius**2)
            r0_dot = (R_n_dot - r0*(R_n_dot+alt_dot))/(R_n + alt)
            denom_dot = rho_dot*(1-efe2*r0) - rho*efe2*r0_dot
            
            alt_dot = (alt+R_n)*(rho_dot/rho + np.tan(lat)*lat_dot) - R_n_dot
            lat_dot = (-denom_dot*z + z_dot*denom)/(z**2 + denom**2)
            
            alt = rho/np.cos(lat) - R_n
            lat = np.arctan2(z,denom)
            
            idx_alt = np.all(np.abs(old_alt-alt)<1.10e-8)
            idx_lat = np.all(np.abs(old_lat-lat)<3.6e-8)
            tol = idx_alt and idx_lat
            
        
        if deg:
            lat = np.rad2deg(lat)
            lon = np.rad2deg(lon)
            lat_dot = np.rad2deg(lat_dot)
            lon_dot = np.rad2deg(lon_dot)
        d.update({'latitude':lat,'longitude':lon,'altitude':alt,
                  'latitude_rate':lat_dot,'longitude_rate':lon_dot,'altitude_rate':alt_dot})    
    
    if rsize == 9:
        if isinstance(x,ndtypes):
            x = np.array([x],dtype=preferred_dtype)
            y = np.array([y],dtype=preferred_dtype)
            z = np.array([z],dtype=preferred_dtype)
        else:
            x = np.array(x,dtype=preferred_dtype)
            y = np.array(y,dtype=preferred_dtype)
            z = np.array(z,dtype=preferred_dtype)
            
        x_dot = args[0]
        y_dot = args[1]
        z_dot = args[2]

        if isinstance(x_dot,ndtypes):
            x_dot = np.array([x_dot],dtype=preferred_dtype)
            y_dot = np.array([y_dot],dtype=preferred_dtype)
            z_dot = np.array([z_dot],dtype=preferred_dtype)
        else:
            x_dot = np.array(x_dot,dtype=preferred_dtype)
            y_dot = np.array(y_dot,dtype=preferred_dtype)
            z_dot = np.array(z_dot,dtype=preferred_dtype)
        
        x_ddot = args[3]
        y_ddot = args[4]
        z_ddot = args[5]

        if isinstance(x_ddot,ndtypes):
            x_ddot = np.array([x_ddot],dtype=preferred_dtype)
            y_ddot = np.array([y_ddot],dtype=preferred_dtype)
            z_ddot = np.array([z_ddot],dtype=preferred_dtype)
        else:
            x_ddot = np.array(x_ddot,dtype=preferred_dtype)
            y_ddot = np.array(y_ddot,dtype=preferred_dtype)
            z_ddot = np.array(z_ddot,dtype=preferred_dtype)
    
        rho = np.sqrt(x**2+y**2) 
        rho_dot = (x_dot*x + y_dot*y)/rho
        rho_ddot = (x_ddot*x + y_ddot*y + x_dot**2 + y_dot**2 - rho_dot**2)/rho
        
        
        lon = np.arctan2(y,x)
        lat = np.arctan2(rho,z)
        alt = 0
        
        lon_dot = (-x_dot*y + y_dot*x)/rho**2
        lat_dot = (-z_dot*rho + rho_dot*z)/(rho**2 + z**2)
        alt_dot = 0
        
        lon_ddot = (-x_ddot*y + y_ddot*x - 2*lon_dot*rho*rho_dot)/(rho**2)
        lat_ddot = (-z_ddot*rho + rho_ddot*z - 2*lat_dot*(rho*rho_dot + z*z_dot))/(rho**2 + z**2)
        alt_ddot = 0
        while not tol:
            old_alt = alt
            old_lat = lat
            
            R_n = earth_equitorial_radius/np.sqrt(1 - efe2 * np.sin(lat)**2)
            r0 = R_n/(R_n+alt)
            denom = rho*(1-efe2*r0)
            
            R_n_dot = (R_n**3 * efe2 *np.sin(lat)*np.cos(lat)*lat_dot)* (1 / earth_equitorial_radius**2)
            r0_dot = (R_n_dot - r0*(R_n_dot+alt_dot))/(R_n + alt)
            denom_dot = rho_dot*(1-efe2*r0) - rho*efe2*r0_dot
            
            R_n_ddot = (efe2*R_n**2/earth_equitorial_radius**2) * (R_n*(lat_dot**2 *(np.cos(lat)**2 - np.sin(lat)**2) + lat_ddot*np.sin(lat)*np.cos(lat)) + 3*R_n_dot*lat_dot*np.sin(lat)*np.cos(lat))
            r0_ddot = (R_n_ddot - 2*r0_dot*(R_n_dot+alt_dot) - r0*(R_n_ddot+alt_ddot))/(R_n+alt)
            denom_ddot = rho_ddot*(1-efe2*r0) - 2*rho_dot*efe2*r0_dot - rho*efe2*r0_ddot
            
            alt_ddot = (alt_dot+R_n_dot)*(rho_dot/rho + np.tan(lat)*lat_dot) + (alt+R_n)*((rho_ddot*rho-rho_dot**2)/rho**2 + np.tan(lat)*lat_ddot + (1/np.cos(lat))**2 * lat_dot**2) - R_n_ddot
            lat_ddot = (-denom_ddot*z + z_ddot*denom - 2*lat_dot*(z*z_dot + denom*denom_dot))/(denom**2 + z**2)
            
            
            alt_dot = (alt+R_n)*(rho_dot/rho + np.tan(lat)*lat_dot) - R_n_dot
            lat_dot = (-denom_dot*z + z_dot*denom)/(z**2 + denom**2)
            
            
            
            alt = rho/np.cos(lat) - R_n
            lat = np.arctan2(z,denom)
            
            
            
            idx_alt = np.all(np.abs(old_alt-alt)<1.10e-8)
            idx_lat = np.all(np.abs(old_lat-lat)<3.6e-8)
            tol = idx_alt and idx_lat
            
        
        if deg:
            lat = np.rad2deg(lat)
            lon = np.rad2deg(lon)
            lat_dot = np.rad2deg(lat_dot)
            lon_dot = np.rad2deg(lon_dot)
            lat_ddot = np.rad2deg(lat_ddot)
            lon_ddot = np.rad2deg(lon_ddot)
        d.update({'latitude':lat,'longitude':lon,'altitude':alt,
                  'latitude_rate':lat_dot,'longitude_rate':lon_dot,'altitude_rate':alt_dot,
                  'latitude_acc':lat_ddot,'longitude_acc':lon_ddot,'altitude_acc':alt_ddot})
    
    return pd.DataFrame(d)
    
    
def ecef2geodeticmeh(x,y,z,*args,**kwargs):
    rsize = 3 + len(args)
    rsize = int(rsize/3)*3

    d = {}

    deg = kwargs.get('deg',True)
    if isinstance(x,ndtypes):
        x = np.array([x],dtype=preferred_dtype)
        y = np.array([y],dtype=preferred_dtype)
        z = np.array([z],dtype=preferred_dtype)
    else:
        x = np.array(x,dtype=preferred_dtype)
        y = np.array(y,dtype=preferred_dtype)
        z = np.array(z,dtype=preferred_dtype)

    if rsize > 3:
        x_dot = args[0]
        y_dot = args[1]
        z_dot = args[2]

        if isinstance(x_dot,ndtypes):
            x_dot = np.array([x_dot],dtype=preferred_dtype)
            y_dot = np.array([y_dot],dtype=preferred_dtype)
            z_dot = np.array([z_dot],dtype=preferred_dtype)
        else:
            x_dot = np.array(x_dot,dtype=preferred_dtype)
            y_dot = np.array(y_dot,dtype=preferred_dtype)
            z_dot = np.array(z_dot,dtype=preferred_dtype)
    else:
        x_dot = 0
        y_dot = 0
        z_dot = 0
    
    if rsize == 9:
        x_ddot = args[3]
        y_ddot = args[4]
        z_ddot = args[5]

        if isinstance(x_ddot,ndtypes):
            x_ddot = np.array([x_ddot],dtype=preferred_dtype)
            y_ddot = np.array([y_ddot],dtype=preferred_dtype)
            z_ddot = np.array([z_ddot],dtype=preferred_dtype)
        else:
            x_ddot = np.array(x_ddot,dtype=preferred_dtype)
            y_ddot = np.array(y_ddot,dtype=preferred_dtype)
            z_ddot = np.array(z_ddot,dtype=preferred_dtype)
    else:
        x_ddot = 0
        y_ddot = 0
        z_ddot = 0

    rho = np.sqrt(x**2 + y**2)
    rho_dot = (x * x_dot + y * y_dot)/rho
    rho_ddot = (x**2 *(x * x_ddot + y_dot**2) + x * y* (x * y_ddot - 2* x_dot *y_dot) + y**2 *(x *x_ddot + x_dot**2) + y**3 *y_ddot)/(x**2 + y**2)**(3/2)
    F = 54*earth_polar_radius**2*z**2
    F_dot = 108 *earth_polar_radius**2 *z *z_dot
    F_ddot = 108 *earth_polar_radius**2 *(z_dot**2 + z* z_ddot)
    G = rho**2 + (1-earth_first_eccentricity**2)*z**2 - earth_first_eccentricity*(earth_equitorial_radius**2-earth_polar_radius**2)
    G_dot = (2 - 2* earth_first_eccentricity**2)* z* z_dot + 2* rho* rho_dot
    G_ddot = 2* (-((-1 + earth_first_eccentricity**2)* z_dot**2) + rho_dot**2 - (-1 + earth_first_eccentricity**2)* z* z_ddot + rho* rho_ddot)
    c = (earth_first_eccentricity**4 * F * rho**2)/G**3
    c_dot = c*(F_dot/F + 2*rho_dot/rho - 3*G_dot/G)
    # c_dot = (earth_first_eccentricity**4 *rho *(rho *G* F_dot + F *(2* G* rho_dot - 3* rho* G_dot)))/G**4
    c_ddot = (earth_first_eccentricity**4 *rho *(F *(-rho_dot* G_dot + 2* G* rho_ddot - 3 *rho* G_ddot) + G *rho_dot* F_dot + rho *G_dot *F_dot + (2* G *rho_dot - 3* rho *G_dot) *F_dot + rho *G *F_ddot))/G**4 + (earth_first_eccentricity**4 *rho_dot *(rho *G *F_dot + F *(2 *G *rho_dot - 3 *rho *G_dot)))/G**4 - (4 *earth_first_eccentricity**4 *rho *G_dot *(rho *G *F_dot + F *(2 *G *rho_dot - 3 *rho* G_dot)))/G**5
    s = np.cbrt(1 + c + np.sqrt(c**2+2*c))
    s_dot = (s* c_dot)/(3 *np.sqrt(c *(c + 2)))
    print(s_dot.dtype)
    s_ddot = -((c + np.sqrt(c *(c + 2)) + 1)**(1/3) *c_dot *(c *c_dot + (c + 2) *c_dot))/(6 *(c *(c + 2))**(3/2)) + (c_dot *(c_dot + (c *c_dot + (c + 2)* c_dot)/(2 *np.sqrt(c *(c + 2)))))/(9 *np.sqrt(c *(c + 2))* (c + np.sqrt(c *(c + 2)) + 1)**(2/3)) + ((c + np.sqrt(c *(c + 2)) + 1)**(1/3) *c_ddot)/(3 *np.sqrt(c *(c + 2)))
    k = s + 1 + (1/s)
    k_dot = (s_dot - s_dot/s**2)
    k_ddot = (2* s_dot**2)/s**3 + (1 - 1/s**2) *s_ddot
    P = F/(3*k**2*G**2)
    P_dot = P*(F_dot/F - 2*k_dot/k - 2*G_dot/G)
    # P_dot = (G *(k *F_dot - 2 *F* k_dot) - 2 *F *k *G_dot)/(3* G**3 *k**3)
    P_ddot = (G *(-F_dot *k_dot + k *F_ddot - 2 *F *k_ddot) - 2 *k *F_dot *G_dot - 2 *F *k_dot *G_dot + (k *F_dot - 2 *F *k_dot) *G_dot - 2 *F *k *G_ddot)/(3 *G**3 *k**3) - (G_dot *(G *(k *F_dot - 2 *F *k_dot) - 2 *F *k *G_dot))/(G**4 *k**3) - (k_dot *(G *(k *F_dot - 2 *F *k_dot) - 2 *F *k *G_dot))/(G**3 *k**4)
    Q = np.sqrt(1 + 2*earth_first_eccentricity**4*P)
    Q_dot = (earth_first_eccentricity**4 *P_dot)/np.sqrt(2 *earth_first_eccentricity**4 *P + 1)
    Q_ddot = (earth_first_eccentricity**4 *((2 *earth_first_eccentricity**4 *P + 1) *P_ddot - earth_first_eccentricity**4 *P_dot**2))/(2 *earth_first_eccentricity**4 *P + 1)**(3/2)
    alpha = -P * earth_first_eccentricity**2 * rho/(1+Q)
    alpha_dot = -(earth_first_eccentricity**2 *(rho *(Q + 1) *P_dot + P *((Q + 1) *rho_dot - rho *Q_dot)))/(Q + 1)**2
    alpha_ddot = (2 *earth_first_eccentricity**2 *Q_dot *(rho *(Q + 1) *P_dot + P *((Q + 1) *rho_dot - rho* Q_dot)))/(Q + 1)**3 - (earth_first_eccentricity**2 *((Q + 1) *rho_dot *P_dot + rho *Q_dot *P_dot + ((Q + 1) *rho_dot - rho *Q_dot)* P_dot + rho *(Q + 1) *P_ddot + P *((Q + 1) *rho_ddot - rho *Q_ddot)))/(Q + 1)**2
    beta = earth_equitorial_radius**2/2
    gamma= beta/Q
    gamma_dot = -(beta* Q_dot)/Q**2
    gamma_dot = -Q_dot*gamma/Q
    gamma_ddot = (beta *(2 *Q_dot**2 - Q *Q_ddot))/Q**3
    delta = ((1-earth_first_eccentricity**2)*P*z**2)/(Q*(1+Q))
    delta_dot = (2*(1-earth_first_eccentricity**2)*P *z* z_dot)/(Q*(Q+1))+((1-earth_first_eccentricity**2)*z**2 *P_dot)/(Q*(Q+1))-((1-earth_first_eccentricity**2)*P *z**2 *Q_dot)/(Q**2 *(Q+1))-((1-earth_first_eccentricity**2) *P *z**2 *Q_dot)/(Q*(Q+1)**2)
    delta_ddot = (1/(Q**3 *(1+Q)**3))*(-2 *(-1+earth_first_eccentricity**2) *P *z**2 *Q_dot**2+Q *z *(2 *(-2+2 *earth_first_eccentricity**2) *P *Q_dot *z_dot+z *((-2+2 *earth_first_eccentricity**2) *P_dot *Q_dot+P *(-((-6+6 *earth_first_eccentricity**2) *Q_dot**2)+(-1+earth_first_eccentricity**2) *Q_ddot)))-(-1+earth_first_eccentricity**2) *Q**4 *(2 *P *z_dot**2+z**2 *P_ddot+2 *z *(2 *P_dot *z_dot+P *z_ddot))+Q**3 *(-4 *(-1+earth_first_eccentricity**2) *P *z_dot**2+z**2 *((-4+4 *earth_first_eccentricity**2) *P_dot *Q_dot-2 *(-1+earth_first_eccentricity**2) *P_ddot+(-2+2 *earth_first_eccentricity**2) *P *Q_ddot)+z *(-8 *(-1+earth_first_eccentricity**2) *P_dot *z_dot+2 *P *((-4+4 *earth_first_eccentricity**2) *Q_dot *z_dot-2 *(-1+earth_first_eccentricity**2) *z_ddot)))+Q**2 *(-2 *(-1+earth_first_eccentricity**2) *P *z_dot**2+z**2 *(2 *(-3+3 *earth_first_eccentricity**2) *P_dot *Q_dot-(-1+earth_first_eccentricity**2) *P_ddot+P *(-3 *(-2+2 *earth_first_eccentricity**2) *Q_dot**2+(-3+3 *earth_first_eccentricity**2) *Q_ddot))+2 *z *(-2 *(-1+earth_first_eccentricity**2) *P_dot *z_dot+P *(2 *(-3+3 *earth_first_eccentricity**2) *Q_dot *z_dot-(-1+earth_first_eccentricity**2) *z_ddot))))
    zeta = (P * rho**2)/2
    zeta_dot = 1/2 *rho *(2 *P *rho_dot + rho *P_dot)
    zeta_ddot = P *(rho_dot**2 + rho *rho_ddot) + 1/2 *rho *(4 *rho_dot *P_dot + rho *P_ddot)
    r0 = alpha + np.sqrt(beta + gamma - delta - zeta)
    r0_dot = alpha_dot + (gamma_dot - delta_dot - zeta_dot)/(2 *np.sqrt(gamma - delta - zeta + beta))
    r0_ddot = -(gamma_dot - delta_dot - zeta_dot)**2/(4 *(gamma - delta - zeta + beta)**(3/2)) + alpha_ddot + (gamma_ddot - delta_ddot - zeta_ddot)/(2 *np.sqrt(gamma - delta - zeta + beta))
    U = np.sqrt((rho - earth_first_eccentricity**2 * r0)**2 + z**2)
    U_dot = ((rho - earth_first_eccentricity**2 *r0)* (rho_dot - earth_first_eccentricity**2 *r0_dot) + z* z_dot)/np.sqrt((rho - earth_first_eccentricity**2 *r0)**2 + z**2)
    U_ddot = ((rho_dot - earth_first_eccentricity**2 *r0_dot)**2 + (rho - earth_first_eccentricity**2 *r0) *(rho_ddot - earth_first_eccentricity**2 *r0_ddot) + z *z_ddot + z_dot**2)/np.sqrt((rho - earth_first_eccentricity**2 *r0)**2 + z**2) - (((rho - earth_first_eccentricity**2 *r0) *(rho_dot - earth_first_eccentricity**2 *r0_dot) + z *z_dot) *(2 *(rho - earth_first_eccentricity**2 *r0) *(rho_dot - earth_first_eccentricity**2 *r0_dot) + 2 *z* z_dot))/(2 *((rho - earth_first_eccentricity**2 *r0)**2 + z**2)**(3/2))
    V = np.sqrt((rho - earth_first_eccentricity**2 * r0)**2 + (1 - earth_first_eccentricity**2)*z**2)
    V_dot = (2 *(rho - earth_first_eccentricity**2 *r0)* (rho_dot - earth_first_eccentricity**2 *r0_dot) + 2 *(1 - earth_first_eccentricity**2) *z *z_dot)/(2 *np.sqrt((rho - earth_first_eccentricity**2* r0)**2 + (1 - earth_first_eccentricity**2)* z**2))
    V_ddot = (2 *(rho_dot - earth_first_eccentricity**2 *r0_dot)**2 + 2 *(rho - earth_first_eccentricity**2 *r0) *(rho_ddot - earth_first_eccentricity**2 *r0_ddot) + 2 *(1 - earth_first_eccentricity**2) *z *z_ddot + 2 *(1 - earth_first_eccentricity**2) *z_dot**2)/(2 *np.sqrt((rho - earth_first_eccentricity**2 *r0)**2 + (1 - earth_first_eccentricity**2) *z**2)) - (2 *(rho - earth_first_eccentricity**2 *r0) *(rho_dot - earth_first_eccentricity**2 *r0_dot) + 2 *(1 - earth_first_eccentricity**2) *z *z_dot)**2/(4 *((rho - earth_first_eccentricity**2 *r0)**2 + (1 - earth_first_eccentricity**2) *z**2)**(3/2))
    z0 = earth_polar_radius**2 * z/ (earth_equitorial_radius * V)
    z0_dot = (earth_polar_radius**2 *(V *z_dot - z *V_dot))/(earth_equitorial_radius *V**2)
    z0_ddot = (earth_polar_radius**2 *(V**2 *z_ddot - V *(2 *V_dot *z_dot + z *V_ddot) + 2 *z *V_dot**2))/(earth_equitorial_radius *V**3)
    rho_0 = z + earth_second_eccentricity**2*z0
    rho_0_dot = earth_second_eccentricity**2 *z0_dot + z_dot
    rho_0_ddot = earth_second_eccentricity**2 *z0_ddot + z_ddot
    lat = np.arctan2(rho_0,rho)
    lat_dot = (rho *rho_0_dot - rho_0 *rho_dot)/(rho**2 + rho_0**2)
    lat_ddot = (2 *(rho_0 *rho_dot - rho *rho_0_dot) *(rho *rho_dot + rho_0 *rho_0_dot) + (rho**2 + rho_0**2) *(rho *rho_0_ddot - rho_0 *rho_ddot))/(rho**2 + rho_0**2)**2
    lon = np.arctan2(y,x)
    lon_dot = (y *x_dot - x *y_dot)/(x**2 + y**2)
    lon_ddot = ((x**2 + y**2) *(y *x_ddot - x *y_ddot) - 2 *(y *x_dot - x *y_dot) *(x* x_dot + y *y_dot))/(x**2 + y**2)**2
    alt = U * (1 - (earth_polar_radius**2/(earth_equitorial_radius*V))) 
    alt_dot = (earth_polar_radius**2 *U *V_dot)/(earth_equitorial_radius *V**2) + (1 - earth_polar_radius**2/(earth_equitorial_radius *V)) *U_dot
    alt_ddot = (earth_polar_radius**2 *V *(2 *U_dot *V_dot + U *V_ddot) - 2 *earth_polar_radius**2 *U *V_dot**2 - earth_polar_radius**2 *V**2 *U_ddot + earth_equitorial_radius *V**3 *U_ddot)/(earth_equitorial_radius *V**3)


    #Old 
    
    x2 = x*x
    y2 = y*y
    z2 = z*z
    epr2 = earth_polar_radius**2
    eer2 = earth_equitorial_radius**2
    efe22 = efe2*efe2

    r2 = x2+y2
    r = np.sqrt(r2)
    epr2 = earth_polar_radius**2
    eer2 = earth_equitorial_radius**2
    F = 54*epr2*z2
    G = r2 + (one_efe2)*z2 - efe2*(eer2_epr2)
    G2 = G*G
    efe22 = efe2*efe2
    c = efe22 * F *r2/(G2*G)
    s = np.cbrt(1+c+np.sqrt(c*c+2*c))
    k = (s+1+1/s)
    k2 = k*k
    P = F/(3*k2*G2)
    Q = np.sqrt(1+2*efe22*P)
    # one_plus_Q = 1+Q
    # r0 = -P*efe2*r/one_plus_Q + np.sqrt(.5*eer2*(1+1/Q)-P*(one_efe2)*z2/(Q*one_plus_Q) - .5*P*r2)
    alpha = -P*efe2*r/(1+Q)
    beta = eer2/2
    gamma = beta/Q
    delta = one_efe2*P*z**2/(Q*(1+Q))
    zeta = P*r**2/2
    r0 = alpha + np.sqrt(beta+gamma-delta-zeta)
    cu1 = (r-efe2*r0)
    cu2 = cu1*cu1
    U = np.sqrt(cu2+z2)
    V = np.sqrt(cu2+z2*(one_efe2))
    divisor = 1/(earth_equitorial_radius*V)
    z0 = epr2*z*divisor
    rho_0 = z + ese2*z0
    alt = U*(1 - epr2*divisor)
    lat = np.arctan2(rho_0,r)
    lon = np.arctan2(y,x)
    
    
    
    r_dot = (x_dot*x + y_dot*y)/r
    oF_dot = 108*epr2*z*z_dot
    oG_dot = 2*(r*r_dot + z*z_dot*one_efe2)
    oc_dot = c*(oF_dot/F + 2*r_dot/r - 3*oG_dot/G)
    # os_dot = (oc_dot/3)*(s-(c/s**2))
    os_dot = oc_dot*s/(3*np.sqrt(c**2+2*c))
    ok_dot = os_dot - os_dot/s**2
    nP_dot = (G *(k *oF_dot - 2 *F* ok_dot) - 2 *F *k *oG_dot)/(3* G**3 *k**3)
    oP_dot = P*(oF_dot/F - 2*ok_dot/k - 2*oG_dot/G)
    print(max(abs(nP_dot - oP_dot)))
    oQ_dot = efe22*oP_dot / Q
    oalpha_dot = alpha * (oP_dot/P + r_dot/r - oQ_dot/(1+Q))
    ogamma_dot = -oQ_dot*gamma/Q
    odelta_dot = delta * (oP_dot/P + 2*z_dot/z - oQ_dot*(1+2*Q)/(Q*(1+Q)))
    ozeta_dot = zeta*oP_dot/P + 2*zeta*r_dot/r
    or0_dot = oalpha_dot + (ogamma_dot-odelta_dot-ozeta_dot)/(2*(r0-alpha))
    oU_dot = ((r-efe2*r0)*(r_dot-efe2*or0_dot)+z*z_dot)/U
    oV_dot = ((r-efe2*r0)*(r_dot-efe2*or0_dot)+one_efe2*z*z_dot)/V
    oz0_dot = z0*(z_dot/z - oV_dot/V)
    orho_0_dot = z_dot + ese2*z0_dot

    olat_dot = (r*orho_0_dot - rho_0*r_dot)/(rho_0**2+r**2)
    olon_dot = (x*y_dot-y*x_dot)/r**2
    oalt_dot = oU_dot* alt/U + (earth_polar_radius**2/earth_equitorial_radius)*U*(oV_dot/V**2)
    
    print('rho_dot',np.isclose(rho_dot,r_dot).all())
    print('F',np.isclose(F_dot,oF_dot).all())
    print('G',np.isclose(G_dot,oG_dot).all())
    print('c',np.isclose(c_dot,oc_dot).all())
    print('s',np.isclose(s_dot,os_dot).all())
    print('k',np.isclose(k_dot,ok_dot).all())
    print('P',np.isclose(P_dot,oP_dot).all())
    print('Q',np.isclose(Q_dot,oQ_dot).all())
    print('alpha',np.isclose(alpha_dot,oalpha_dot).all())
    print('gamma',np.isclose(gamma_dot,ogamma_dot).all())
    print('delta',np.isclose(delta_dot,odelta_dot).all())
    print('zeta',np.isclose(zeta_dot,ozeta_dot).all())
    print('r0',np.isclose(r0_dot,or0_dot).all())
    print('U',np.isclose(U_dot,oU_dot).all())
    print('V',np.isclose(V_dot,oV_dot).all())
    print('z0',np.isclose(z0_dot,oz0_dot).all())
    print('rho_0',np.isclose(rho_0_dot,orho_0_dot).all())
    print('lat',np.isclose(lat_dot,olat_dot).all())
    print('lon',np.isclose(lon_dot,olon_dot).all())
    print('alt',np.isclose(alt_dot,oalt_dot).all())
    
    print('p',max(abs(P_dot-oP_dot)))
    print('alpha',max(abs(alpha_dot-oalpha_dot)))
    print('gamma',max(abs(gamma_dot-ogamma_dot)))
    print('delta',max(abs(delta_dot-odelta_dot)))
    print('zeta',max(abs(zeta_dot-ozeta_dot)))
    print(max(abs(s_dot - os_dot)))

    
    if deg:
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)
    d.update({'Lat':lat,'Lon':lon,'Alt':alt})
    if rsize > 3:
        d.update({'LatRate':lat_dot,'LonRate':lon_dot,'AltRate':alt_dot})
    if rsize == 9:
        d.update({'LatAcc':lat_ddot,'LonAcc':lon_ddot,'AltAcc':alt_ddot})
    
    return pd.DataFrame(d)
    
    
def ecef2geodetic22(x,y,z,*args,**kwargs):
    '''
    This is Ferrari's method (or Heikkinen's method: but Ferrari sounds cooler)
    is the following method to convert ecef2geodetic coordinates.  This is the most
    accurate and the fastest method we currently have.

    source: https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#The_application_of_Ferrari's_solution

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    rsize = 3 + len(args)
    rsize = int(rsize/3)*3

    d = {}

    deg = kwargs.get('deg',True)
    if isinstance(x,ndtypes):
        x = np.array([x],dtype=preferred_dtype)
        y = np.array([y],dtype=preferred_dtype)
        z = np.array([z],dtype=preferred_dtype)
    else:
        x = np.array(x,dtype=preferred_dtype)
        y = np.array(y,dtype=preferred_dtype)
        z = np.array(z,dtype=preferred_dtype)

    x2 = x*x
    y2 = y*y
    z2 = z*z

    r2 = x2+y2
    r = np.sqrt(r2)
    epr2 = earth_polar_radius**2
    eer2 = earth_equitorial_radius**2
    F = 54*epr2*z2
    G = r2 + (one_efe2)*z2 - efe2*(eer2_epr2)
    G2 = G*G
    efe22 = efe2*efe2
    c = efe22 * F *r2/(G2*G)
    s = np.cbrt(1+c+np.sqrt(c*c+2*c))
    k = (s+1+1/s)
    k2 = k*k
    P = F/(3*k2*G2)
    Q = np.sqrt(1+2*efe22*P)
    # one_plus_Q = 1+Q
    # r0 = -P*efe2*r/one_plus_Q + np.sqrt(.5*eer2*(1+1/Q)-P*(one_efe2)*z2/(Q*one_plus_Q) - .5*P*r2)
    alpha = -P*efe2*r/(1+Q)
    beta = eer2/2
    gamma = beta/Q
    delta = one_efe2*P*z**2/(Q*(1+Q))
    zeta = P*r**2/2
    r0 = alpha + np.sqrt(beta+gamma-delta-zeta)
    cu1 = (r-efe2*r0)
    cu2 = cu1*cu1
    U = np.sqrt(cu2+z2)
    V = np.sqrt(cu2+z2*(one_efe2))
    divisor = 1/(earth_equitorial_radius*V)
    z0 = epr2*z*divisor
    rho_0 = z + ese2*z0
    alt = U*(1 - epr2*divisor)
    lat = np.arctan2(rho_0,r)
    lon = np.arctan2(y,x)
    if deg:
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)
    d.update({'Lat':lat,'Lon':lon,'Alt':alt})

    if rsize > 3:
        x_dot = args[0]
        y_dot = args[1]
        z_dot = args[2]

        if isinstance(x_dot,ndtypes):
            x_dot = np.array([x_dot],dtype=preferred_dtype)
            y_dot = np.array([y_dot],dtype=preferred_dtype)
            z_dot = np.array([z_dot],dtype=preferred_dtype)
        else:
            x_dot = np.array(x_dot,dtype=preferred_dtype)
            y_dot = np.array(y_dot,dtype=preferred_dtype)
            z_dot = np.array(z_dot,dtype=preferred_dtype)

        r_dot = (x_dot*x + y_dot*y)/r
        F_dot = 108*epr2*z*z_dot
        G_dot = 2*(r*r_dot + z*z_dot*one_efe2)
        c_dot = c*(F_dot/F + 2*r_dot/r - 3*G_dot/G)
        s_dot = (c_dot/3)*(s-(c/s**2))
        k_dot = s_dot - s_dot/s**2
        P_dot = P*(F_dot/F - 2*k_dot/k - 2*G_dot/G)
        Q_dot = efe22*P_dot / Q
        alpha_dot = alpha * (P_dot/P + r_dot/r - Q_dot/(1+Q))
        gamma_dot = -Q_dot*gamma/Q
        delta_dot = delta * (P_dot/P + 2*z_dot/z - Q_dot*(1+2*Q)/(Q*(1+Q)))
        zeta_dot = zeta*P_dot/P + 2*zeta*r_dot/r
        r0_dot = alpha_dot + (gamma_dot-delta_dot-zeta_dot)/(2*(r0-alpha))
        U_dot = ((r-efe2*r0)*(r_dot-efe2*r0_dot)+z*z_dot)/U
        V_dot = ((r-efe2*r0)*(r_dot-efe2*r0_dot)+one_efe2*z*z_dot)/V
        z0_dot = z0*(z_dot/z - V_dot/V)
        rho_0_dot = z_dot + ese2*z0_dot

        lat_dot = (r*rho_0_dot - rho_0*r_dot)/(rho_0**2+r**2)
        lon_dot = (x*y_dot-y*x_dot)/r**2
        if deg:
            lat_dot = np.rad2deg(lat_dot)
            lon_dot = np.rad2deg(lon_dot)
        alt_dot = U_dot* alt/U + (earth_polar_radius**2/earth_equitorial_radius)*U*(V_dot/V**2)

        d.update({'LatRate':lat_dot,'LonRate':lon_dot,'AltRate':alt_dot})

    if rsize == 9:
        x_ddot = args[3]
        y_ddot = args[4]
        z_ddot = args[5]

        if isinstance(x_ddot,ndtypes):
            x_ddot = np.array([x_ddot],dtype=preferred_dtype)
            y_ddot = np.array([y_ddot],dtype=preferred_dtype)
            z_ddot = np.array([z_ddot],dtype=preferred_dtype)
        else:
            x_ddot = np.array(x_ddot,dtype=preferred_dtype)
            y_ddot = np.array(y_ddot,dtype=preferred_dtype)
            z_ddot = np.array(z_ddot,dtype=preferred_dtype)

        r_ddot = (x_ddot*x + y_ddot*y + x_dot**2 + y_dot**2 - r_dot**2)/r
                
                
        F_ddot = 108*epr2*(z_dot**2+z*z_ddot)
        
        
        G_ddot = 2*(r_dot**2+r*r_ddot+one_efe2*(z_ddot*z+z_dot**2))
        
        
        cddF = (F_ddot*F-F_dot**2)/F**2
        cddr = (r_ddot*r-r_dot**2)/r**2
        cddG = (G_ddot*G-G_dot**2)/G**2
        c_ddot = c_dot**2/c + c*(cddF+2*cddr-3*cddG)
        
        s1 = (c_ddot*s + c_dot*s_dot)/3
        s2 = -(c_ddot*c+c_dot**2)/(3*s**2)
        s3 = (2*c_dot*s_dot*c)/(3*s**3)
        s_ddot = (s1+s2+s3)
        
        
        k_ddot = s_ddot - s_ddot/s**2 + 2*s_dot**2/s**3
        
        
        pF = cddF
        pG = cddG
        pk = (k_ddot*k-k_dot**2)/k**2
        P_ddot = P_dot**2/P + P*(pF-2*pG-2*pk)
        
        
        Q_ddot = Q_dot*(P_ddot/P_dot - Q_dot/Q)
        
        
        
        aP = (P_ddot*P-P_dot**2)/P**2
        ar = cddr
        aq = ((1+Q)*Q_ddot - Q_dot**2)/(1+Q)**2
        alpha_ddot = alpha_dot**2/alpha + alpha * (aP + ar - aq)
        
        
        
        gamma_ddot = -2*gamma_dot*Q_dot - gamma*Q_ddot
        
        
        
        deltaP = (P_ddot*P-P_dot**2)/P**2
        deltaz = (z_ddot*z-z_dot**2)/z**2
        deltaQ = (Q*(1+Q)*(Q_ddot*(1+2*Q) + 2*Q_dot**2) - Q_dot*(1+2*Q)*(Q_dot*(1+Q)+Q*Q_dot))/(Q**2*(1+Q)**2)
        delta_ddot = delta_dot**2/delta + delta*(deltaP + 2*deltaz - deltaQ)
        
        
        zetaP = deltaP
        zetar = ar
        zeta_ddot = zeta_dot**2/zeta + zeta * (zetaP + 2*zetar)
        
        
        r01 = (r0-alpha)*(gamma_ddot-delta_ddot-zeta_ddot)
        r02 = (r0_dot-alpha_dot)*(gamma_dot-delta_dot-zeta_dot)
        r0_denom = 2*(r0-alpha)**2
        r0_ddot = alpha_ddot + (r01+r02)/r0_denom
        
        
        U_ddot = ((r_dot-efe2*r0_dot)**2 + (r-efe2*r0)*(r_ddot-efe2*r0_ddot) + z_dot**2 + z*z_ddot - U_dot**2)/U
        
        V_ddot = ((r_dot-efe2*r0_dot)**2 + (r-efe2*r0)*(r_ddot-efe2*r0_ddot) + (one_efe2)*(z_dot**2+z*z_ddot) - V_dot**2)/V
        
        z0z = (z_ddot*z - z_dot**2)/z**2
        z0V = (V_ddot*V - V_dot**2)/V**2
        z0_ddot = z0_dot**2/z0 + z0* (z0z - z0V)
        
        rho_0_ddot = z_ddot + ese2*z0_ddot

        lat_ddot = (r*rho_0_ddot - rho_0*r_ddot - 2*lat_dot*(r*r_dot+rho_0*rho_0_dot))/(rho_0**2 + r**2)
        
        lon_ddot = (x*y_ddot - y*x_ddot - 2*lon_dot*(x*x_dot+y*y_dot))/(r**2)
        
        if deg:
            lat_ddot = np.rad2deg(lat_ddot)
            lon_ddot = np.rad2deg(lon_ddot)
        rr = earth_polar_radius**2/earth_equitorial_radius
        alt1 = rr*(V_dot*U_dot+U*V_ddot)/V**2
        alt2 = 2*rr*(U*V_dot**2)/V**3
        alt3 = (U*alt_dot*U_dot - alt*U_dot**2+alt*U*U_ddot)/U**2
        alt_ddot = alt1 - alt2 + alt3

        d.update({'LatAcc':lat_ddot,'LonAcc':lon_ddot,'AltAcc':alt_ddot})



    return pd.DataFrame(d)


def ecef2geodeticZhu(x,y,z,**kwargs):
    '''
    This is the Zhu Algorithm for ecef to geodetic.  This is the original method
    found by Zhu in 1993.  This is known to be inaccurate around latitudes of
    +/- 45.288 degrees.  This method is slower and more inaccurate than the
    Ferrari method (ecef2geodetic) and the modified Zhu method (ecef2geodeticModifiedZhu)

    source: https://hal.archives-ouvertes.fr/hal-01704943v2/document (first method)

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    deg = kwargs.get('deg',True)
    if isinstance(x,ndtypes):
        x = np.array([x],dtype=preferred_dtype)
        y = np.array([y],dtype=preferred_dtype)
        z = np.array([z],dtype=preferred_dtype)
    else:
        x = np.array(x,dtype=preferred_dtype)
        y = np.array(y,dtype=preferred_dtype)
        z = np.array(z,dtype=preferred_dtype)

    x2 = x*x
    y2 = y*y
    z2 = z*z

    w = np.sqrt(x2+y2)
    m = w*w/(earth_equitorial_radius**2)
    n = z2*one_efe2_over_epr**2
    i = -0.5*(2*l2+m+n)
    k = l2*(l2-m-n)
    q = (m+n-4*l2)**3/216+m*n*l2
    D = np.sqrt((2*q-n*m*l2)*m*n*l2)
    B = i/3 - (q+D)**(1/3) - (q-D)**(1/3)
    t = np.sqrt(np.sqrt(B*B-k)-0.5*(B+i)) - np.sign(m-n)*np.sqrt(.5*np.abs((B-i)))
    w1 = w/(t+l)
    z1 = one_efe2*z/(t-l)
    lat = np.arctan2(z1,one_efe2*w1)
    alt = np.sign(t-1+l)*np.sqrt((w-w1)**2+(z-z1)**2)
    lon = np.arctan2(y,x)

    if deg:
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)

    return pd.DataFrame({'Lat':lat,'Lon':lon,'Alt':alt})

def ecef2geodeticPymap(x,y,z,**kwargs):
    '''
    This method seems to be inaccurate for some ecef positions and was the main
    source of investigations into other methods.  This is not to be used unless
    it is known that all data points are well outside the earth's surface.  Inaccuracy
    comes from the parameters "m" and "n" being so close together within machine
    precision.  Given the precision is good enough, this method could be the fastest
    that we have.

    source: https://www.researchgate.net/publication/240359424_Transformation_of_Cartesian_to_Geodetic_Coordinates_without_Iterations

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    deg = kwargs.get('deg',True)
    if isinstance(x,ndtypes):
        x = np.array([x],dtype=preferred_dtype)
        y = np.array([y],dtype=preferred_dtype)
        z = np.array([z],dtype=preferred_dtype)
    else:
        x = np.array(x,dtype=preferred_dtype)
        y = np.array(y,dtype=preferred_dtype)
        z = np.array(z,dtype=preferred_dtype)

    x2 = x*x
    y2 = y*y
    z2 = z*z

    E2 = eer2_epr2
    r2_E2 = x2+y2+z2-E2
    u = np.sqrt(0.5*r2_E2 + 0.5*np.sqrt(r2_E2*r2_E2 + 4*E2*z2))
    Q = np.sqrt(x2+y2)
    huE = np.sqrt(u*u+E2)
    huEEER = huE*earth_equitorial_radius
    Beta = np.arctan2(huE*z,Q*u)
    sinBeta = np.sin(Beta)
    cosBeta = np.cos(Beta)
    eps = ((earth_polar_radius * u - huEEER + E2) * sinBeta) / (huEEER/cosBeta - E2 * cosBeta)
    Beta += eps

    lat = np.arctan2(earth_equitorial_radius*np.tan(Beta), earth_polar_radius)
    lon = np.arctan2(y, x)
    alt = np.sqrt((z-earth_polar_radius*np.sin(Beta))**2+(Q-earth_equitorial_radius*np.cos(Beta))**2)

    inside = np.array(earth_polar_radius**2*(x2+y2) + earth_equitorial_radius**2*z2 < earth_equitorial_radius**2*earth_polar_radius**2)
    if inside.any():
        alt[inside] = -alt[inside]

    if deg:
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)

    return pd.DataFrame({'Lat':lat,'Lon':lon,'Alt':alt})

def enu2rae(east,north,up,*args,**kwargs):
    d = {}
    deg = kwargs.get('deg',True)
    if isinstance(east,ndtypes):
        east = np.array([east],dtype=preferred_dtype)
        north = np.array([north],dtype=preferred_dtype)
        up = np.array([up],dtype=preferred_dtype)
    else:
        east = np.array(east,dtype=preferred_dtype)
        north = np.array(north,dtype=preferred_dtype)
        up = np.array(up,dtype=preferred_dtype)

    rsize = 3 + len(args)
    rsize = int(rsize/3)*3

    east2 = east*east
    north2 = north*north
    east2north2 = east2+north2
    up2 = up*up
    r = np.sqrt(east2north2+up2)
    azimuth = np.arctan2(east,north)
    elevation = np.arctan2(up,np.sqrt(east2north2))
    if deg:
        azimuth = np.rad2deg(azimuth)
        elevation = np.rad2deg(elevation)

    d.update({'Range':r,'Azimuth':azimuth,'Elevation':elevation})
    if rsize > 3:
        veast = args[0]
        vnorth = args[1]
        vup = args[2]
        if isinstance(veast,ndtypes):
            veast = np.array([veast]*east.shape[0],dtype=preferred_dtype)
            vnorth = np.array([vnorth]*east.shape[0],dtype=preferred_dtype)
            vup = np.array([vup]*east.shape[0],dtype=preferred_dtype)
        else:
            veast = np.array(veast,dtype=preferred_dtype)
            vnorth = np.array(vnorth,dtype=preferred_dtype)
            vup = np.array(vup,dtype=preferred_dtype)

        one_over_r = 1/r
        r_dot = (east*veast + north*vnorth + up*vup)*one_over_r
        az_dot = (north*veast - east*vnorth)/(east2north2)
        el_dot = (vup - up*r_dot*one_over_r)/np.sqrt(east2north2)
        if deg:
            az_dot = np.rad2deg(az_dot)
            el_dot = np.rad2deg(el_dot)
        d.update({'RangeRate':r_dot,'AzimuthRate':az_dot,'ElevationRate':el_dot})

    if rsize == 9:
        aeast = args[3]
        anorth = args[4]
        aup = args[5]
        if isinstance(aeast,ndtypes):
            aeast = np.array([aeast]*east.shape[0],dtype=preferred_dtype)
            anorth = np.array([anorth]*east.shape[0],dtype=preferred_dtype)
            aup = np.array([aup]*east.shape[0],dtype=preferred_dtype)
        else:
            aeast = np.array(aeast,dtype=preferred_dtype)
            anorth = np.array(anorth,dtype=preferred_dtype)
            aup = np.array(aup,dtype=preferred_dtype)

        r_ddot = (veast**2 + vnorth**2 + vup**2 + east*aeast + north*anorth + up*aup - r_dot**2)*one_over_r
        az_ddot = ((north*aeast-east*anorth)-2*az_dot*(east*veast+north*vnorth))/east2north2
        el_ddot = (r*r*aup-r*r_dot*vup-r*r_ddot*up+up*r_dot*r_dot)/(np.sqrt(east2north2)*r*r) - (el_dot*(east*veast+north*vnorth))/east2north2
        if deg:
            az_ddot = np.rad2deg(az_ddot)
            el_ddot = np.rad2deg(el_ddot)
        d.update({'RangeAcc':r_ddot,'AzimuthAcc':az_ddot,'ElevationAcc':el_ddot})

    return pd.DataFrame(d)

def rae2enu(r,azimuth,elevation,*args,**kwargs):
    d = {}
    deg = kwargs.get('deg',True)
    if isinstance(r,ndtypes):
        r = np.array([r],dtype=preferred_dtype)
        azimuth = np.array([azimuth],dtype=preferred_dtype)
        elevation = np.array([elevation],dtype=preferred_dtype)
    else:
        r = np.array(r,dtype=preferred_dtype)
        azimuth = np.array(azimuth,dtype=preferred_dtype)
        elevation = np.array(elevation,dtype=preferred_dtype)

    if deg:
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)

    rsize = 3 + len(args)
    rsize = int(rsize/3)*3

    cosel = np.cos(elevation)
    sinel = np.sin(elevation)
    cosaz = np.cos(azimuth)
    sinaz = np.sin(azimuth)

    rcosel = r*cosel

    east = rcosel*sinaz
    north = rcosel*cosaz
    up = r*sinel

    d.update({'East':east,'North':north,'Up':up})

    if rsize > 3:
        r_dot = args[0]
        az_dot = args[1]
        el_dot = args[2]
        if isinstance(r_dot,ndtypes):
            r_dot = np.array([r_dot],dtype=preferred_dtype)
            az_dot = np.array([az_dot],dtype=preferred_dtype)
            el_dot = np.array([el_dot],dtype=preferred_dtype)
        else:
            r_dot = np.array(r_dot,dtype=preferred_dtype)
            az_dot = np.array(az_dot,dtype=preferred_dtype)
            el_dot = np.array(el_dot,dtype=preferred_dtype)

        r_dot_over_r = r_dot/r
        up_el_dot = up*el_dot

        veast = east*r_dot_over_r + north*az_dot - up_el_dot*sinaz
        vnorth = north*r_dot_over_r - east*az_dot - up_el_dot*cosaz
        vup = up*r_dot_over_r + r*el_dot*cosel
        d.update({'EastRate':veast,'NorthRate':vnorth,'UpRate':vup})

    if rsize == 9:
        r_ddot = args[3]
        az_ddot = args[4]
        el_ddot = args[5]
        if isinstance(r_ddot,ndtypes):
            r_ddot = np.array([r_ddot],dtype=preferred_dtype)
            az_ddot = np.array([az_ddot],dtype=preferred_dtype)
            el_ddot = np.array([el_ddot],dtype=preferred_dtype)
        else:
            r_ddot = np.array(r_ddot,dtype=preferred_dtype)
            az_ddot = np.array(az_ddot,dtype=preferred_dtype)
            el_ddot = np.array(el_ddot,dtype=preferred_dtype)
        one_over_r = 1/r

        aeast = (r*(east*r_ddot+r_dot*veast)-east*r_dot**2)*one_over_r**2 + vnorth*az_dot+north*az_ddot-vup*el_dot*sinaz-up*el_ddot*sinaz-up*el_dot*az_dot*cosaz
        anorth = (r*(north*r_ddot+r_dot*vnorth)-north*r_dot**2)*one_over_r**2 - veast*az_dot-east*az_ddot-vup*el_dot*cosaz-up*el_ddot*cosaz+up*el_dot*az_dot*sinaz
        aup = (r*(up*r_ddot+r_dot*vup)-up*r_dot**2)*one_over_r**2 + r_dot*el_dot*cosel + r*el_ddot*cosel - r*el_dot**2*sinel
        d.update({'EastAcc':aeast,'NorthAcc':anorth,'UpAcc':aup})

    return pd.DataFrame(d)

def ruv2rfc(r,u,v,*args,**kwargs):
    d = {}
    if isinstance(r,ndtypes):
        r = np.array([r],dtype=preferred_dtype)
        u = np.array([u],dtype=preferred_dtype)
        v = np.array([v],dtype=preferred_dtype)
    else:
        r = np.array(r,dtype=preferred_dtype)
        u = np.array(u,dtype=preferred_dtype)
        v = np.array(v,dtype=preferred_dtype)

    rsize = 3 + len(args)
    rsize = int(rsize/3)*3

    #Rounding to 15 decimal places inside sqrt.  This propogates error by +/- .1 meters in some cases
    #This is for all intents and purposes 0.  A machine precision error for 32 bit float
    w = np.sqrt(np.around(1.-u**2-v**2,15))

    x = r*u
    y = r*v
    z = r*w

    d={'x':x,'y':y,'z':z}
    if rsize > 3:
        r_dot = args[0]
        u_dot = args[1]
        v_dot = args[2]
        if isinstance(r_dot,ndtypes):
            r_dot = np.array([r_dot],dtype=preferred_dtype)
            u_dot = np.array([u_dot],dtype=preferred_dtype)
            v_dot = np.array([v_dot],dtype=preferred_dtype)
        else:
            r_dot = np.array(r_dot,dtype=preferred_dtype)
            u_dot = np.array(u_dot,dtype=preferred_dtype)
            v_dot = np.array(v_dot,dtype=preferred_dtype)

        w_dot = (-u*u_dot - v*v_dot)/w

        vx = r*u_dot + r_dot*u
        vy = r*v_dot + r_dot*v
        vz = r*w_dot + r_dot*w
        d.update({'vx':vx,'vy':vy,'vz':vz})

    if rsize == 9:
        r_ddot = args[3]
        u_ddot = args[4]
        v_ddot = args[5]
        if isinstance(r_ddot,ndtypes):
            r_ddot = np.array([r_ddot],dtype=preferred_dtype)
            u_ddot = np.array([u_ddot],dtype=preferred_dtype)
            v_ddot = np.array([v_ddot],dtype=preferred_dtype)
        else:
            r_ddot = np.array(r_ddot,dtype=preferred_dtype)
            u_ddot = np.array(u_ddot,dtype=preferred_dtype)
            v_ddot = np.array(v_ddot,dtype=preferred_dtype)

        w_ddot = (-u_dot**2-u*u_ddot-v_dot**2-v*v_ddot-w_dot**2)/w

        ax = r_ddot*u + r*u_ddot + 2*r_dot*u_dot
        ay = r_ddot*v + r*v_ddot + 2*r_dot*v_dot
        az = r_ddot*w + r*w_ddot + 2*r_dot*w_dot

        d.update({'ax':ax,'ay':ay,'az':az})

    return pd.DataFrame(d)

def rfc2ruv(x,y,z,*args,**kwargs):
    d = {}
    if isinstance(x,ndtypes):
        x = np.array([x],dtype=preferred_dtype)
        y = np.array([y],dtype=preferred_dtype)
        z = np.array([z],dtype=preferred_dtype)
    else:
        x = np.array(x,dtype=preferred_dtype)
        y = np.array(y,dtype=preferred_dtype)
        z = np.array(z,dtype=preferred_dtype)

    rsize = 3 + len(args)
    rsize = int(rsize/3)*3
    sign = np.sign(z)
    sign = np.where(sign==0,sign+1,sign)
    r = np.sqrt(x*x+y*y+z*z)*sign
    one_over_r = 1/r
    u = x*one_over_r
    v = y*one_over_r

    d.update({'range':r,'u':u,'v':v})

    if rsize > 3:
        vx = args[0]
        vy = args[1]
        vz = args[2]
        if isinstance(vx,ndtypes):
            vx = np.array([vx],dtype=preferred_dtype)
            vy = np.array([vy],dtype=preferred_dtype)
            vz = np.array([vz],dtype=preferred_dtype)
        else:
            vx = np.array(vx,dtype=preferred_dtype)
            vy = np.array(vy,dtype=preferred_dtype)
            vz = np.array(vz,dtype=preferred_dtype)

        w = z*one_over_r
        r_dot = u*vx + v*vy + w*vz
        u_dot = (vx - r_dot*u)*one_over_r
        v_dot = (vy - r_dot*v)*one_over_r

        d.update({'r_dot':r_dot,'u_dot':u_dot,'v_dot':v_dot})
    if rsize == 9:
        ax = args[3]
        ay = args[4]
        az = args[5]
        if isinstance(ax,ndtypes):
            ax = np.array([ax],dtype=preferred_dtype)
            ay = np.array([ay],dtype=preferred_dtype)
            az = np.array([az],dtype=preferred_dtype)
        else:
            ax = np.array(ax,dtype=preferred_dtype)
            ay = np.array(ay,dtype=preferred_dtype)
            az = np.array(az,dtype=preferred_dtype)
        w_dot = (r*vz - z*r_dot)/r**2
        r_ddot = u_dot*vx + u*ax + v_dot*vy + v*ay + w_dot*vz + w*az
        u_ddot = (ax*r - vx*r_dot - r_ddot*r*u - r_dot*u_dot*r + r_dot*r_dot*u)*one_over_r**2
        v_ddot = (ay*r - vy*r_dot - r_ddot*r*u - r_dot*u_dot*r + r_dot*r_dot*u)*one_over_r**2

        d.update({'r_ddot':r_ddot,'u_ddot':u_ddot,'v_ddot':v_ddot})

    return pd.DataFrame(d)

def rfc2enu(x,y,z,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    '''
    For SBX give clocking angle = 0

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    boresightAz : TYPE
        DESCRIPTION.
    boresightEl : TYPE
        DESCRIPTION.
    clockingangle : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    d = {}
    deg = kwargs.get('deg',True)
    if isinstance(x,ndtypes):
        x = np.array([x],dtype=preferred_dtype)
        y = np.array([y],dtype=preferred_dtype)
        z = np.array([z],dtype=preferred_dtype)
    else:
        x = np.array(x,dtype=preferred_dtype)
        y = np.array(y,dtype=preferred_dtype)
        z = np.array(z,dtype=preferred_dtype)

    rsize = 3 + len(args)
    rsize = int(rsize/3)*3

    if deg:
        boresightAz = np.deg2rad(boresightAz)
        boresightEl = np.deg2rad(boresightEl)
        clockingangle = np.deg2rad(clockingangle)

    cosc = np.cos(clockingangle)
    sinc = np.sin(clockingangle)
    cosbA = np.cos(boresightAz)
    sinbA = np.sin(boresightAz)
    cosbE = np.cos(boresightEl)
    sinbE = np.sin(boresightEl)

    P11 = -cosc*cosbA - sinc*sinbE*sinbA
    P12 = sinc*cosbA - cosc*sinbE*sinbA
    P13 = cosbE*sinbA

    P21 = cosc*sinbA - sinc*sinbE*cosbA
    P22 = -sinc*sinbA - cosc*sinbE*cosbA
    P23 = cosbE*cosbA

    P31 = sinc*cosbE
    P32 = cosc*cosbE
    P33 = sinbE

    east = x*P11 + y*P12 + z*P13
    north = x*P21 + y*P22 + z*P23
    up = x*P31 + y*P32 + z*P33

    d.update({'East':east,'North':north,'Up':up})

    if rsize > 3:
        vx = args[0]
        vy = args[1]
        vz = args[2]
        if isinstance(vx,ndtypes):
            vx = np.array([vx],dtype=preferred_dtype)
            vy = np.array([vy],dtype=preferred_dtype)
            vz = np.array([vz],dtype=preferred_dtype)
        else:
            vx = np.array(vx,dtype=preferred_dtype)
            vy = np.array(vy,dtype=preferred_dtype)
            vz = np.array(vz,dtype=preferred_dtype)

        veast = vx*P11 + vy*P12 + vz*P13
        vnorth = vx*P21 + vy*P22 + vz*P23
        vup = vx*P31 + vy*P32 + vz*P33

        d.update({'EastRate':veast,'NorthRate':vnorth,'UpRate':vup})

    if rsize == 9:
        ax = args[3]
        ay = args[4]
        az = args[5]
        if isinstance(ax,ndtypes):
            ax = np.array([ax],dtype=preferred_dtype)
            ay = np.array([ay],dtype=preferred_dtype)
            az = np.array([az],dtype=preferred_dtype)
        else:
            ax = np.array(ax,dtype=preferred_dtype)
            ay = np.array(ay,dtype=preferred_dtype)
            az = np.array(az,dtype=preferred_dtype)

        aeast = ax*P11 + ay*P12 + az*P13
        anorth = ax*P21 + ay*P22 + az*P23
        aup = ax*P31 + ay*P32 + az*P33

        d.update({'EastAcc':aeast,'NorthAcc':anorth,'UpAcc':aup})

    return pd.DataFrame(d)

def enu2rfc(east,north,up,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    d = {}
    deg = kwargs.get('deg',True)
    if isinstance(east,ndtypes):
        east = np.array([east],dtype=preferred_dtype)
        north = np.array([north],dtype=preferred_dtype)
        up = np.array([up],dtype=preferred_dtype)
    else:
        east = np.array(east,dtype=preferred_dtype)
        north = np.array(north,dtype=preferred_dtype)
        up = np.array(up,dtype=preferred_dtype)

    rsize = 3 + len(args)
    rsize = int(rsize/3)*3

    if deg:
        boresightAz = np.deg2rad(boresightAz)
        boresightEl = np.deg2rad(boresightEl)
        clockingangle = np.deg2rad(clockingangle)

    cosc = np.cos(clockingangle)
    sinc = np.sin(clockingangle)
    cosbA = np.cos(boresightAz)
    sinbA = np.sin(boresightAz)
    cosbE = np.cos(boresightEl)
    sinbE = np.sin(boresightEl)

    P11 = -cosc*cosbA - sinc*sinbE*sinbA
    P12 = cosc*sinbA - sinc*sinbE*cosbA
    P13 = sinc*cosbE

    P21 = sinc*cosbA - cosc*sinbE*sinbA
    P22 = -sinc*sinbA - cosc*sinbE*cosbA
    P23 = cosc*cosbE

    P31 = cosbE*sinbA
    P32 = cosbE*cosbA
    P33 = sinbE

    x = east*P11 + north*P12 + up*P13
    y = east*P21 + north*P22 + up*P23
    z = east*P31 + north*P32 + up*P33

    d.update({'rfc_x':x,'rfc_y':y,'rfc_z':z})
    if rsize>3:
        veast = args[0]
        vnorth = args[1]
        vup = args[2]
        if isinstance(veast, ndtypes):
            veast = np.array([veast],dtype=preferred_dtype)
            vnorth = np.array([vnorth],dtype=preferred_dtype)
            vup = np.array([vup],dtype=preferred_dtype)
        else:
            veast = np.array(veast,dtype=preferred_dtype)
            vnorth = np.array(vnorth,dtype=preferred_dtype)
            vup = np.array(vup,dtype=preferred_dtype)

        x_dot = veast*P11 + vnorth*P12 + vup*P13
        y_dot = veast*P21 + vnorth*P22 + vup*P23
        z_dot = veast*P31 + vnorth*P32 + vup*P33

        d.update({'rfc_x_dot':x_dot,'rfc_y_dot':y_dot,'rfc_z_dot':z_dot})

    if rsize == 9:
        aeast = args[3]
        anorth = args[4]
        aup = args[5]
        if isinstance(aeast, ndtypes):
            aeast = np.array([aeast],dtype=preferred_dtype)
            anorth = np.array([anorth],dtype=preferred_dtype)
            aup = np.array([aup],dtype=preferred_dtype)
        else:
            aeast = np.array(aeast,dtype=preferred_dtype)
            anorth = np.array(anorth,dtype=preferred_dtype)
            aup = np.array(aup,dtype=preferred_dtype)

        x_ddot = aeast*P11 + anorth*P12 + aup*P13
        y_ddot = aeast*P21 + anorth*P22 + aup*P23
        z_ddot = aeast*P31 + anorth*P32 + aup*P33

        d.update({'rfc_x_ddot':x_ddot,'rfc_y_ddot':y_ddot,'rfc_z_ddot':z_ddot})

    return pd.DataFrame(d)

#%% ecef wrappers

def ecef2rae(x,y,z,lat0,lon0,alt0,*args,**kwargs):
    fdf = pd.DataFrame()
    df = ecef2enu(x, y, z, lat0, lon0, alt0, *args,**kwargs)
    if df.shape[1] == 3:
        fdf = enu2rae(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], **kwargs)
    elif df.shape[1] == 6:
        fdf = enu2rae(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2],df.iloc[:,3], df.iloc[:,4], df.iloc[:,5],**kwargs)
    return fdf

def ecef2rfc(x,y,z,lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    fdf = pd.DataFrame()
    df = ecef2enu(x, y, z, lat0, lon0, alt0, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = enu2rfc(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, *kwargs)
    elif df.shape[1] == 6:
        fdf = enu2rfc(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def ecef2ruv(x,y,z,lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    fdf = pd.DataFrame()
    idf = pd.DataFrame()
    df = ecef2enu(x,y,z,lat0,lon0,alt0,*args,**kwargs)
    if df.shape[1] == 3:
        idf = enu2rfc(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, **kwargs)
    elif df.shape[1] == 6:
        idf = enu2rfc(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2],boresightAz,boresightEl,clockingangle,df.iloc[:,3], df.iloc[:,4], df.iloc[:,5],**kwargs)
    if idf.shape[1] == 3:
        fdf = rfc2ruv(idf.iloc[:,0], idf.iloc[:,1], idf.iloc[:,2], **kwargs)
    elif idf.shape[1] == 6:
        fdf = rfc2ruv(idf.iloc[:,0], idf.iloc[:,1], idf.iloc[:,2], idf.iloc[:,3], idf.iloc[:,4], idf.iloc[:,5],**kwargs)
    return fdf

#%% eci wrappers
def eci2geodetic(x,y,z,t,*args,**kwargs):
    fdf = pd.DataFrame()
    df = eci2ecef(x, y, z, t, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = ecef2geodetic(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], **kwargs)
    elif df.shape[1] == 6:
        fdf = ecef2geodetic(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def eci2enu(x,y,z,t,lat0,lon0,alt0,*args,**kwargs):
    fdf = pd.DataFrame()
    df = eci2ecef(x, y, z, t, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = ecef2enu(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, **kwargs)
    elif df.shape[1] == 6:
        fdf = ecef2enu(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def eci2rae(x,y,z,t,lat0,lon0,alt0,*args,**kwargs):
    fdf = pd.DataFrame()
    df = eci2ecef(x, y, z, t, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = ecef2rae(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, **kwargs)
    elif df.shape[1] == 6:
        fdf = ecef2rae(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def eci2rfc(x,y,z,t,lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    fdf = pd.DataFrame()
    df = eci2ecef(x, y, z, t, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = ecef2rfc(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,**kwargs)
    elif df.shape[1] == 6:
        fdf = ecef2rfc(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,df.iloc[:,3], df.iloc[:,4], df.iloc[:,5],**kwargs)
    return fdf

def eci2ruv(x,y,z,t,lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    fdf = pd.DataFrame()
    df = eci2ecef(x, y, z, t, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = ecef2ruv(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,**kwargs)
    elif df.shape[1] == 6:
        fdf = ecef2ruv(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,df.iloc[:,3], df.iloc[:,4], df.iloc[:,5],**kwargs)
    return fdf

#%% geodetic wrappers
def geodetic2eci(lat,lon,alt,t,*args,**kwargs):
    fdf = pd.DataFrame()
    df = geodetic2ecef(lat, lon, alt, *args,**kwargs)
    if df.shape[1] == 3:
        fdf = ecef2eci(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], t, **kwargs)
    elif df.shape[1] == 6:
        fdf = ecef2eci(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], t, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def geodetic2enu(lat,lon,alt,lat0,lon0,alt0,*args,**kwargs):
    fdf = pd.DataFrame()
    df = geodetic2ecef(lat, lon, alt, *args,**kwargs)
    if df.shape[1] == 3:
        fdf = ecef2enu(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0,lon0,alt0, **kwargs)
    elif df.shape[1] == 6:
        fdf = ecef2enu(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0,lon0,alt0, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def geodetic2rae(lat,lon,alt,lat0,lon0,alt0,*args,**kwargs):
    fdf = pd.DataFrame()
    df = geodetic2ecef(lat, lon, alt, *args,**kwargs)
    if df.shape[1] == 3:
        fdf = ecef2rae(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0,lon0,alt0, **kwargs)
    elif df.shape[1] == 6:
        fdf = ecef2rae(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0,lon0,alt0, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def geodetic2rfc(lat,lon,alt,lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    fdf = pd.DataFrame()
    df = geodetic2ecef(lat,lon,alt, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = ecef2rfc(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,**kwargs)
    elif df.shape[1] == 6:
        fdf = ecef2rfc(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,df.iloc[:,3], df.iloc[:,4], df.iloc[:,5],**kwargs)
    return fdf

def geodetic2ruv(lat,lon,alt,lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    fdf = pd.DataFrame()
    df = geodetic2ecef(lat,lon,alt, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = ecef2ruv(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,**kwargs)
    elif df.shape[1] == 6:
        fdf = ecef2ruv(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,df.iloc[:,3], df.iloc[:,4], df.iloc[:,5],**kwargs)
    return fdf

#%% enu wrappers
def enu2eci(east,north,up,lat0,lon0,alt0,t,*args,**kwargs):
    fdf = pd.DataFrame()
    df = enu2ecef(east, north, up, lat0, lon0, alt0, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = ecef2eci(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], t, **kwargs)
    elif df.shape[1] == 6:
        fdf = ecef2eci(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], t, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def enu2geodetic(east,north,up,lat0,lon0,alt0,*args,**kwargs):
    fdf = pd.DataFrame()
    df = enu2ecef(east, north, up, lat0, lon0, alt0, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = ecef2geodetic(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], **kwargs)
    elif df.shape[1] == 6:
        fdf = ecef2geodetic(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def enu2ruv(east,north,up,lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    fdf = pd.DataFrame()
    df = enu2ecef(east, north, up, lat0, lon0, alt0, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = ecef2ruv(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,**kwargs)
    elif df.shape[1] == 6:
        fdf = ecef2ruv(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,df.iloc[:,3], df.iloc[:,4], df.iloc[:,5],**kwargs)
    return fdf

#%% rae wrappers
def rae2ecef(r,azimuth,elevation,lat0,lon0,alt0,*args,**kwargs):
    fdf = pd.DataFrame()
    df = rae2enu(r, azimuth, elevation, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = enu2ecef(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, **kwargs)
    elif df.shape[1] == 6:
        fdf = enu2ecef(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0,lon0,alt0, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def rae2eci(r,azimuth,elevation,lat0,lon0,alt0,t,*args,**kwargs):
    fdf = pd.DataFrame()
    df = rae2enu(r, azimuth, elevation, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = enu2eci(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, t, **kwargs)
    elif df.shape[1] == 6:
        fdf = enu2eci(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, t, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def rae2geodetic(r,azimuth,elevation,lat0,lon0,alt0,*args,**kwargs):
    fdf = pd.DataFrame()
    df = rae2enu(r, azimuth, elevation, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = enu2geodetic(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, **kwargs)
    elif df.shape[1] == 6:
        fdf = enu2geodetic(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def rae2rfc(r,azimuth,elevation,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    fdf = pd.DataFrame()
    df = rae2enu(r, azimuth, elevation, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = enu2rfc(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, **kwargs)
    elif df.shape[1] == 6:
        fdf = enu2rfc(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def rae2ruv(r,azimuth,elevation,lat0,lon0,alt0,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    fdf = pd.DataFrame()
    df = rae2enu(r, azimuth, elevation, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = enu2ruv(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, boresightAz, boresightEl, clockingangle, **kwargs)
    elif df.shape[1] == 6:
        fdf = enu2ruv(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, boresightAz, boresightEl, clockingangle, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

#%% rfc wrappers
def rfc2eci(x,y,z,boresightAz,boresightEl,clockingangle,lat0,lon0,alt0,t,*args,**kwargs):
    fdf = pd.DataFrame()
    df = rfc2enu(x, y, z, boresightAz, boresightEl, clockingangle, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = enu2eci(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, t, **kwargs)
    elif df.shape[1] == 6:
        fdf = enu2eci(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, t, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def rfc2ecef(x,y,z,boresightAz,boresightEl,clockingangle,lat0,lon0,alt0,*args,**kwargs):
    fdf = pd.DataFrame()
    df = rfc2enu(x, y, z, boresightAz, boresightEl, clockingangle, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = enu2ecef(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, **kwargs)
    elif df.shape[1] == 6:
        fdf = enu2ecef(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def rfc2geodetic(x,y,z,boresightAz,boresightEl,clockingangle,lat0,lon0,alt0,*args,**kwargs):
    fdf = pd.DataFrame()
    df = rfc2enu(x, y, z, boresightAz, boresightEl, clockingangle, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = enu2geodetic(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, **kwargs)
    elif df.shape[1] == 6:
        fdf = enu2geodetic(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], lat0, lon0, alt0, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def rfc2rae(x,y,z,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    fdf = pd.DataFrame()
    df = rfc2enu(x, y, z, boresightAz, boresightEl, clockingangle, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = enu2rae(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], **kwargs)
    elif df.shape[1] == 6:
        fdf = enu2rae(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

#%% ruv wrappers
def ruv2eci(r,u,v,boresightAz,boresightEl,clockingangle,lat0,lon0,alt0,t,*args,**kwargs):
    fdf = pd.DataFrame()
    df = ruv2rfc(r, u, v, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = rfc2eci(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, lat0, lon0, alt0, t, **kwargs)
    elif df.shape[1] == 6:
        fdf = rfc2eci(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, lat0, lon0, alt0, t, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def ruv2ecef(r,u,v,boresightAz,boresightEl,clockingangle,lat0,lon0,alt0,*args,**kwargs):
    fdf = pd.DataFrame()
    df = ruv2rfc(r, u, v, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = rfc2ecef(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, lat0, lon0, alt0, **kwargs)
    elif df.shape[1] == 6:
        fdf = rfc2ecef(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, lat0, lon0, alt0, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def ruv2geodetic(r,u,v,boresightAz,boresightEl,clockingangle,lat0,lon0,alt0,*args,**kwargs):
    fdf = pd.DataFrame()
    df = ruv2rfc(r, u, v, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = rfc2geodetic(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, lat0, lon0, alt0, **kwargs)
    elif df.shape[1] == 6:
        fdf = rfc2geodetic(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, lat0, lon0, alt0, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def ruv2rae(r,u,v,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    fdf = pd.DataFrame()
    df = ruv2rfc(r, u, v, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = rfc2rae(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, **kwargs)
    elif df.shape[1] == 6:
        fdf = rfc2rae(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

def ruv2enu(r,u,v,boresightAz,boresightEl,clockingangle,*args,**kwargs):
    fdf = pd.DataFrame()
    df = ruv2rfc(r, u, v, *args, **kwargs)
    if df.shape[1] == 3:
        fdf = rfc2enu(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, **kwargs)
    elif df.shape[1] == 6:
        fdf = rfc2enu(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], boresightAz, boresightEl, clockingangle, df.iloc[:,3], df.iloc[:,4], df.iloc[:,5], **kwargs)
    return fdf

#%% Covariance Transforms
def Cov_enu2ecef(df,lat0,lon0,alt0,**kwargs):
    deg = kwargs.get('deg',True)
    if deg:
        lat0 = np.deg2rad(lat0)
        lon0 = np.deg2rad(lon0)
    if isinstance(df,pd.DataFrame):
        cov_fields = []
        for i in range(6):
            for j in range(6):
                cov_fields.append(f'p{i}{j}')
        if not set(cov_fields).issubset(df.columns):
            return pd.DataFrame()
        cov_matrix = df[cov_fields].values.reshape(df.shape[0],6,6)
    elif isinstance(df,np.ndarray):
        if df.shape[1] == 36:
            #Assuming contains position and velocity .
            cov_matrix = df.reshape(df.shape[0], 6, 6)

        elif df.shape[1] == 81:
        #Assuming contains position, velocity and acceleration .
            cov_matrix = df.reshape(df.shape[0], 9, 9)
            cov_matrix = df[:,:6,:6]
            cov_matrix = df.reshape(df.shape[0], 36)
            cov_matrix = df.reshape(df.shape[0], 6, 6)
    else:
        return pd.DataFrame()
    sinlat = np.sin(lat0)
    coslat = np.cos(lat0)
    sinlon = np.sin(lon0)
    coslon = np.cos(lon0)
    i00 = -sinlon
    i01 = -coslon * sinlat
    i02 = coslon * coslat
    i10 = coslon
    i11 = -sinlon * sinlat
    i12 = sinlon * coslat
    i21 = coslat
    i22 = sinlat

    jacobian = np.zeros(cov_matrix.shape)
    jacobian[:,0,0] = i00
    jacobian[:,0,1] = i01
    jacobian[:,0,2] = i02
    jacobian[:,1,0] = i10
    jacobian[:,1,1] = i11
    jacobian[:,1,2] = i12
    jacobian[:,2,1] = i21
    jacobian[:,2,2] = i22

    jacobian[:,3:,3:] = jacobian[:,:3,:3]
    jacobianT = np.empty((jacobian.shape))
    for i in range(6):
        for j in range(6):
            jacobianT[:,i,j] = jacobian[:,j,i]
    matmul = np.matmul(np.matmul(jacobian, cov_matrix), jacobianT)
    matmul = matmul.reshape((df.shape[0],36))
    if isinstance(df,pd.DataFrame):
        return pd.DataFrame(matmul,columns=cov_fields)
    else:
        return matmul

def Cov_rfc2enu(df,boresightAz,boresightEl,clockingangle,**kwargs):
    deg = kwargs.get('deg',True)
    if deg:
        boresightAz = np.deg2rad(boresightAz)
        boresightEl = np.deg2rad(boresightEl)
        clockingangle = np.deg2rad(clockingangle)
    if isinstance(df,pd.DataFrame):
        cov_fields = []
        for i in range(6):
            for j in range(6):
                cov_fields.append(f'p{i}{j}')
        if not set(cov_fields).issubset(df.columns):
            return pd.DataFrame()
        cov_matrix = df[cov_fields].values.reshape(df.shape[0],6,6)
    elif isinstance(df,np.ndarray):
        if df.shape[1] == 36:
            #Assuming contains position and velocity .
            cov_matrix = df.reshape(df.shape[0], 6, 6)

        elif df.shape[1] == 81:
        #Assuming contains position, velocity and acceleration .
            cov_matrix = df.reshape(df.shape[0], 9, 9)
            cov_matrix = df[:,:6,:6]
            cov_matrix = df.reshape(df.shape[0], 36)
            cov_matrix = df.reshape(df.shape[0], 6, 6)
    cosc = np.cos(clockingangle)
    sinc = np.sin(clockingangle)
    cosbA = np.cos(boresightAz)
    sinbA = np.sin(boresightAz)
    cosbE = np.cos(boresightEl)
    sinbE = np.sin(boresightEl)

    P00 = cosc*cosbA + sinc*sinbE*sinbA
    P01 = -sinc*cosbA + cosc*sinbE*sinbA
    P02 = cosbE*sinbA

    P10 = -cosc*sinbA + sinc*sinbE*cosbA
    P11 = sinc*sinbA + cosc*sinbE*cosbA
    P12 = cosbE*cosbA

    P20 = -sinc*cosbE
    P21 = -cosc*cosbE
    P22 = sinbE

    jacobian = np.zeros(cov_matrix.shape)
    jacobian[:,0,0] = P00
    jacobian[:,0,1] = P01
    jacobian[:,0,2] = P02
    jacobian[:,1,0] = P10
    jacobian[:,1,1] = P11
    jacobian[:,1,2] = P12
    jacobian[:,2,0] = P20
    jacobian[:,2,1] = P21
    jacobian[:,2,2] = P22

    jacobian[:,3:,3:] = jacobian[:,:3,:3]
    jacobianT = np.empty((jacobian.shape))
    for i in range(6):
        for j in range(6):
            jacobianT[:,i,j] = jacobian[:,j,i]
    matmul = np.matmul(np.matmul(jacobian, cov_matrix), jacobianT)
    matmul = matmul.reshape((df.shape[0],36))
    if isinstance(df,pd.DataFrame):
        return pd.DataFrame(matmul,columns=cov_fields)
    else:
        return matmul

#%% Main
if __name__ == '__main__':
    size = 1000
    import time
    d = {'x':np.random.randint(5000000,6000000,size),
         'y':np.random.randint(5000000,6000000,size),
         'z':np.random.randint(5000000,6000000,size),
         'vx':np.random.randint(3000,4000,size),
         'vy':np.random.randint(3000,4000,size),
         'vz':np.random.randint(3000,4000,size),
         'ax':np.random.randint(30,40,size),
         'ay':np.random.randint(30,40,size),
         'az':np.random.randint(30,40,size),
         't':np.linspace(0,500,size),
         'lat':np.linspace(-89,89,size),
         'lon':np.linspace(-179,179,size),
         'alt':np.linspace(0,400000,size)}

    df = pd.DataFrame(d)
    start = time.time()
    rfc = enu2rfc(df['x'],df['y'],df['z'],0,0,0)
    enu = rfc2enu(rfc.iloc[:,0],rfc.iloc[:,1],rfc.iloc[:,2],0,0,0)
    j = ecef2ruv(df['x'],df['y'],df['z'],80,80,0,0,0,0)
    k = ruv2ecef(j.iloc[:,0],j.iloc[:,1],j.iloc[:,2],0,0,0,80,80,0)
    u = rfc2ruv(df['x'],df['y'],df['z'])
    v = ruv2rfc(u.iloc[:,0],u.iloc[:,1],u.iloc[:,2])

    a = ruv2rfc(10,.5,.6)
    b = rfc2ruv(a.iloc[:,0],a.iloc[:,1],a.iloc[:,2])

    c = rfc2ruv(-10,10,-10)
    d = ruv2rfc(c.iloc[:,0],c.iloc[:,1],c.iloc[:,2])



    # #Eci ecef testing
    # start = time.time()
    # j = ecef2eci(df['x'], df['y'], df['z'], df['t'],df['vx'],df['vy'],df['vz'],df['ax'],df['ay'],df['az'])
    # print('ECEF2ECI Linear Equations', time.time()-start)
    # start = time.time()
    # n = eci2ecef(j['x'],j['y'],j['z'],df['t'],j['vx'],j['vy'],j['vz'],j['ax'],j['ay'],j['az'])
    # print('ECI2ECEF Linear Equations',time.time()-start)


    # #ECEF ENU testing
    # start = time.time()
    # n = ecef2enu(df['x'],df['y'],df['z'],df['lat'],df['lon'],df['alt'])
    # print('ECEF2ENU',time.time()-start)
    # start = time.time()
    # c = pymap3d.ecef2enu(df['x'],df['y'],df['z'],df['lat'],df['lon'],df['alt'])
    # print('Pymap3d',time.time()-start)
    # start = time.time()
    # m = enu2ecef(n['East'], n['North'], n['Up'], 34,34,0,5,5,5)
    # print('enu2ecef',time.time()-start)

    #ecef geodetic testing
    # start = time.time()
    # test = geodetic2ecef(df['lat'], df['lon'], df['alt'])
    # print('geodetic2ecef',time.time()-start)
    # start = time.time()
    # rtest = ecef2geodetic(test['x'], test['y'], test['z'])
    # print('ecef2geodetic [Ferrari]',time.time()-start)
    # start = time.time()
    # rrtest = ecef2geodeticModifiedZhu(test['x'],test['y'],test['z'])
    # print('ecef2geodeticModifiedZhu',time.time()-start)
    # start = time.time()
    # rrrtest = ecef2geodeticIter(test['x'],test['y'],test['z'])
    # print('ecef2geodeticIter',time.time()-start)
    # start = time.time()
    # rrrrtest = ecef2geodeticZhu(test['x'],test['y'],test['z'])
    # print('ecef2geodeticZhu',time.time()-start)
    # start = time.time()
    # rrrrr_test = ecef2geodeticPymap(test['x'], test['y'], test['z'])
    # print('ecef2geodeticPymap3d',time.time()-start)

    # rrrval = rrrtest.values
    # rrval = rrtest.values
    # rval = rtest.values
    # rrrrval = rrrrtest.values
    # rrrrrval = rrrrr_test.values
    # r_rr = np.isclose(rval,rrval)
    # r_rrr = np.isclose(rval,rrrval)
    # r_rrrr = np.isclose(rval,rrrrval)
    # r_rrrrr = np.isclose(rval,rrrrrval)
    # idxs = np.where(r_rrrr[:,2]==False)[0]
    # print('is my orig equal to modified zhu?',r_rr.all())
    # print()
    # print()
    # print('is my orig equal to iterative method?',r_rrr.all())
    # print()
    # print()
    # print('the following should be False... but only at lats at +/- 45.288 which is noticed in the reference paper. But the error is small')
    # print()
    # print()
    # print('is my orig equal to zhu?',r_rrrr.all())
    # # for idx in idxs:
    # #     print()
    # #     print()
    # #     print(f'True Values for LLA are {df[["lat","lon","alt"]].iloc[idx].values} and the calculated LLA is {rrrrval[idx,:]} with Alt error {100*np.abs(df["alt"].iloc[idx]-rrrrval[idx,2])/df["alt"].iloc[idx]}%')

    # print()
    # print()
    # print('is my orig equal to ferrari solution?',r_rrrrr.all())
    # #rae enu testing
    # start = time.time()
    # test = enu2rae(n['East'], n['North'], n['Up'])
    # print('enu2rae',time.time()-start)
    # start = time.time()
    # rtest = rae2enu(test['range'], test['azimuth'], test['elevation'])
    # print('rae2enu',time.time()-start)

