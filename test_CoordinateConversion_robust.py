# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:45:41 2021

@author: Jordan
"""
import os,sys
import numpy as np
import pandas as pd
import pytest


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


from utils.geo_spatial import CoordinateConversions as CC


earth_equitorial_radius = 6378137.0 #m
# This is the Point at the North Pole in ECEF.  It will be used for testing
ivory_coast = (0,earth_equitorial_radius,0)
size = 1000000
ivory_coast = (np.random.uniform(earth_equitorial_radius,earth_equitorial_radius*2,(size,)),
               np.random.uniform(earth_equitorial_radius,earth_equitorial_radius*2,(size,)),
               np.random.uniform(earth_equitorial_radius,earth_equitorial_radius*2,(size,)))
#This is a random lat lon alt location for a radar location
random_lla = (np.random.uniform(low=-89, high=89, size=(1,))[0],
              np.random.uniform(low=-179, high=179, size=(1,))[0],
              np.random.uniform(low=0, high=10000, size=(1,))[0])

#This is a random az,el,clocking angle for a radar
random_boresight = (np.random.uniform(low=-10, high=10, size=(1,))[0],
                    np.random.uniform(low=-10, high=10, size=(1,))[0],
                    np.random.uniform(low=-10, high=10, size=(1,))[0])

random_time = np.random.uniform(low=0, high=10000, size=(1,))[0]

def test_position_whole_enchilada():
    x,y,z = ivory_coast
    radar_lat,radar_lon,radar_alt = random_lla
    radar_az,radar_el,radar_clocking = random_boresight
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z]}
    else:
        data = {'x':x,'y':y,'z':z}
    df = pd.DataFrame(data)
    end_test_pts = df.values

    #ECEF2ENU
    enu = CC.ecef2enu(df['x'],df['y'],df['z'],radar_lat,radar_lon,radar_alt).values
    #go back
    test_pts = df.values
    ecef = CC.enu2ecef(enu[:,0],enu[:,1],enu[:,2],radar_lat,radar_lon,radar_alt).values
    print(f'ECEF <-> ENU Match: {np.isclose(test_pts,ecef).all()}')

    #ENU2RAE
    rae = CC.enu2rae(enu[:,0],enu[:,1],enu[:,2]).values
    #go back
    test_pts = enu.copy()
    enu = CC.rae2enu(rae[:,0],rae[:,1],rae[:,2]).values
    print(f'ENU <-> RAE Match: {np.isclose(test_pts,enu).all()}')

    #RAE2LLA
    lla = CC.rae2geodetic(rae[:,0],rae[:,1],rae[:,2],radar_lat,radar_lon,radar_alt).values
    #go back
    test_pts = rae.copy()
    rae = CC.geodetic2rae(lla[:,0],lla[:,1],lla[:,2],radar_lat,radar_lon,radar_alt).values
    print(f'RAE <-> LLA Match: {np.isclose(test_pts,rae).all()}')

    #LLA2RFC
    rfc = CC.geodetic2rfc(lla[:,0],lla[:,1],lla[:,2],radar_lat,radar_lon,radar_alt,radar_az,radar_el,radar_clocking).values
    #go back
    test_pts = lla.copy()
    lla = CC.rfc2geodetic(rfc[:,0],rfc[:,1],rfc[:,2],radar_az,radar_el,radar_clocking,radar_lat,radar_lon,radar_alt).values
    print(f'LLA <-> RFC Match: {np.isclose(test_pts,lla).all()}')

    #RFC2RUV
    ruv = CC.rfc2ruv(rfc[:,0],rfc[:,1],rfc[:,2]).values
    #go back
    test_pts = rfc.copy()
    rfc = CC.ruv2rfc(ruv[:,0],ruv[:,1],ruv[:,2]).values
    print(f'RFC <-> RUV Match: {np.isclose(test_pts,rfc).all()}')

    #RUV2RAE
    rae = CC.ruv2rae(ruv[:,0],ruv[:,1],ruv[:,2],radar_az,radar_el,radar_clocking).values
    #go back
    test_pts = ruv.copy()
    ruv = CC.rae2ruv(rae[:,0],rae[:,1],rae[:,2],radar_lat,radar_lon,radar_alt,radar_az,radar_el,radar_clocking).values
    print(f'RUV <-> RAE Match: {np.isclose(test_pts,ruv).all()}')

    #RAE2ECEF
    ecef = CC.rae2ecef(rae[:,0],rae[:,1],rae[:,2],radar_lat,radar_lon,radar_alt).values
    #go back
    test_pts = rae.copy()
    rae = CC.ecef2rae(ecef[:,0],ecef[:,1],ecef[:,2],radar_lat,radar_lon,radar_alt).values
    print(f'RAE <-> ECEF Match: {np.isclose(test_pts,rae).all()}')

    #ECEF2LLA
    lla = CC.ecef2geodetic(ecef[:,0],ecef[:,1],ecef[:,2]).values
    #go back
    test_pts = ecef.copy()
    ecef = CC.geodetic2ecef(lla[:,0],lla[:,1],lla[:,2]).values
    print(f'ECEF <-> LLA Match: {np.isclose(test_pts,ecef).all()}')

    #LLA2ENU
    enu = CC.geodetic2enu(lla[:,0],lla[:,1],lla[:,2],radar_lat,radar_lon,radar_alt).values
    #go back
    test_pts = lla.copy()
    lla = CC.enu2geodetic(enu[:,0],enu[:,1],enu[:,2],radar_lat,radar_lon,radar_alt).values
    print(f'LLA <-> ENU Match: {np.isclose(test_pts,lla).all()}')

    #ENU2RUV
    ruv = CC.enu2ruv(enu[:,0],enu[:,1],enu[:,2],radar_lat,radar_lon,radar_alt,radar_az,radar_el,radar_clocking).values
    #go back
    test_pts = enu.copy()
    enu = CC.ruv2enu(ruv[:,0],ruv[:,1],ruv[:,2],radar_az,radar_el,radar_clocking).values
    print(f'ENU <-> RUV Match: {np.isclose(test_pts,enu).all()}')

    #RUV2RFC
    rfc = CC.ruv2rfc(ruv[:,0],ruv[:,1],ruv[:,2]).values
    #go back
    test_pts = ruv.copy()
    ruv = CC.rfc2ruv(rfc[:,0],rfc[:,1],rfc[:,2]).values
    print(f'RUV <-> RFC Match: {np.isclose(test_pts,ruv).all()}')

    #RFC2LLA
    lla = CC.rfc2geodetic(rfc[:,0],rfc[:,1],rfc[:,2],radar_az,radar_el,radar_clocking,radar_lat,radar_lon,radar_alt).values
    #go back
    test_pts = rfc.copy()
    rfc = CC.geodetic2rfc(lla[:,0],lla[:,1],lla[:,2],radar_lat,radar_lon,radar_alt,radar_az,radar_el,radar_clocking).values
    print(f'RFC <-> LLA Match: {np.isclose(test_pts,rfc).all()}')

    #LLA2RUV
    ruv = CC.geodetic2ruv(lla[:,0],lla[:,1],lla[:,2],radar_lat,radar_lon,radar_alt,radar_az,radar_el,radar_clocking).values
    #go back
    test_pts = lla.copy()
    lla = CC.ruv2geodetic(ruv[:,0],ruv[:,1],ruv[:,2],radar_az,radar_el,radar_clocking,radar_lat,radar_lon,radar_alt).values
    print(f'LLA <-> RUV Match: {np.isclose(test_pts,lla).all()}')

    #RUV2ECEF
    ecef = CC.ruv2ecef(ruv[:,0],ruv[:,1],ruv[:,2],radar_az,radar_el,radar_clocking,radar_lat,radar_lon,radar_alt).values
    #go back
    test_pts = ruv.copy()
    ruv = CC.ecef2ruv(ecef[:,0],ecef[:,1],ecef[:,2],radar_lat,radar_lon,radar_alt,radar_az,radar_el,radar_clocking).values
    print(f'RUV <-> ECEF Match: {np.isclose(test_pts,ruv).all()}')

    #ECEF2RAE
    rae = CC.ecef2rae(ecef[:,0],ecef[:,1],ecef[:,2],radar_lat,radar_lon,radar_alt).values
    #go back
    test_pts = ecef.copy()
    ecef = CC.rae2ecef(rae[:,0],rae[:,1],rae[:,2],radar_lat,radar_lon,radar_alt).values
    print(f'ECEF <-> RAE Match: {np.isclose(test_pts,ecef).all()}')

    #RAE2RFC
    rfc = CC.rae2rfc(rae[:,0],rae[:,1],rae[:,2],radar_az,radar_el,radar_clocking).values
    #go back
    test_pts = rae.copy()
    rae = CC.rfc2rae(rfc[:,0],rfc[:,1],rfc[:,2],radar_az,radar_el,radar_clocking).values
    print(f'RAE <-> RFC Match: {np.isclose(test_pts,rae).all()}')

    #RFC2ECEF
    ecef = CC.rfc2ecef(rfc[:,0],rfc[:,1],rfc[:,2],radar_az,radar_el,radar_clocking,radar_lat,radar_lon,radar_alt).values
    #go back
    test_pts = rfc.copy()
    rfc = CC.ecef2rfc(ecef[:,0],ecef[:,1],ecef[:,2],radar_lat,radar_lon,radar_alt,radar_az,radar_el,radar_clocking).values
    print(f'RFC <-> ECEF Match: {np.isclose(test_pts,rfc).all()}')

    #ECEF2RUV
    ruv = CC.ecef2ruv(ecef[:,0],ecef[:,1],ecef[:,2],radar_lat,radar_lon,radar_alt,radar_az,radar_el,radar_clocking).values
    #go back
    test_pts = ecef.copy()
    ecef = CC.ruv2ecef(ruv[:,0],ruv[:,1],ruv[:,2],radar_az,radar_el,radar_clocking,radar_lat,radar_lon,radar_alt).values
    print(f'ECEF <-> RUV Match: {np.isclose(test_pts,ecef).all()}')

    #RUV2LLA
    lla = CC.ruv2geodetic(ruv[:,0],ruv[:,1],ruv[:,2],radar_az,radar_el,radar_clocking,radar_lat,radar_lon,radar_alt).values
    #go back
    test_pts = ruv.copy()
    ruv = CC.geodetic2ruv(lla[:,0],lla[:,1],lla[:,2],radar_lat,radar_lon,radar_alt,radar_az,radar_el,radar_clocking).values
    print(f'RUV <-> LLA Match: {np.isclose(test_pts,ruv).all()}')

    #LLA2RAE
    rae = CC.geodetic2rae(lla[:,0],lla[:,1],lla[:,2],radar_lat,radar_lon,radar_alt).values
    #go back
    test_pts = lla.copy()
    lla = CC.rae2geodetic(rae[:,0],rae[:,1],rae[:,2],radar_lat,radar_lon,radar_alt).values
    print(f'LLA <-> RAE Match: {np.isclose(test_pts,lla).all()}')

    #RAE2RUV
    ruv = CC.rae2ruv(rae[:,0],rae[:,1],rae[:,2],radar_lat,radar_lon,radar_alt,radar_az,radar_el,radar_clocking).values
    #go back
    test_pts = rae.copy()
    rae = CC.ruv2rae(ruv[:,0],ruv[:,1],ruv[:,2],radar_az,radar_el,radar_clocking).values
    print(f'RAE <-> RUV Match: {np.isclose(test_pts,rae).all()}')

    #RUV2ENU
    enu = CC.ruv2enu(ruv[:,0],ruv[:,1],ruv[:,2],radar_az,radar_el,radar_clocking).values
    #go back
    test_pts = ruv.copy()
    ruv = CC.enu2ruv(enu[:,0],enu[:,1],enu[:,2],radar_lat,radar_lon,radar_alt,radar_az,radar_el,radar_clocking).values
    print(f'RUV <-> ENU Match: {np.isclose(test_pts,ruv).all()}')

    #ENU2LLA
    lla = CC.enu2geodetic(enu[:,0],enu[:,1],enu[:,2],radar_lat,radar_lon,radar_alt).values
    #go back
    test_pts = enu.copy()
    enu = CC.geodetic2enu(lla[:,0],lla[:,1],lla[:,2],radar_lat,radar_lon,radar_alt).values
    print(f'ENU <-> LLA Match: {np.isclose(test_pts,enu).all()}')

    #LLA2ECEF
    ecef = CC.geodetic2ecef(lla[:,0],lla[:,1],lla[:,2]).values
    #go back
    test_pts = lla.copy()
    lla = CC.ecef2geodetic(ecef[:,0],ecef[:,1],ecef[:,2]).values
    print(f'LLA <-> ECEF Match: {np.isclose(test_pts,lla).all()}')

    #ECEF2RFC
    rfc = CC.ecef2rfc(ecef[:,0],ecef[:,1],ecef[:,2],radar_lat,radar_lon,radar_alt,radar_az,radar_el,radar_clocking).values
    #go back
    test_pts = ecef.copy()
    ecef = CC.rfc2ecef(rfc[:,0],rfc[:,1],rfc[:,2],radar_az,radar_el,radar_clocking,radar_lat,radar_lon,radar_alt).values
    print(f'ECEF <-> RFC Match: {np.isclose(test_pts,ecef).all()}')

    #RFC2ENU
    enu = CC.rfc2enu(rfc[:,0],rfc[:,1],rfc[:,2],radar_az,radar_el,radar_clocking).values
    #go back
    test_pts = rfc.copy()
    rfc = CC.enu2rfc(enu[:,0],enu[:,1],enu[:,2],radar_az,radar_el,radar_clocking).values
    print(f'RFC <-> ENU Match: {np.isclose(test_pts,rfc).all()}')

    #ENU2RAE Again
    rae = CC.enu2rae(enu[:,0],enu[:,1],enu[:,2]).values
    #go back
    test_pts = enu.copy()
    enu = CC.rae2enu(rae[:,0],rae[:,1],rae[:,2]).values
    print(f'ENU <-> RAE Match: {np.isclose(test_pts,enu).all()}')

    #RAE2ENU
    enu = CC.rae2enu(rae[:,0],rae[:,1],rae[:,2]).values
    #go back
    test_pts = rae.copy()
    rae = CC.enu2rae(enu[:,0],enu[:,1],enu[:,2]).values
    print(f'RAE <-> ENU Match: {np.isclose(test_pts,rae).all()}')

    #ENU2RFC
    rfc = CC.enu2rfc(enu[:,0],enu[:,1],enu[:,2],radar_az,radar_el,radar_clocking).values
    #go back
    test_pts = enu.copy()
    enu = CC.rfc2enu(rfc[:,0],rfc[:,1],rfc[:,2],radar_az,radar_el,radar_clocking).values
    print(f'ENU <-> RFC Match: {np.isclose(test_pts,enu).all()}')

    #RFC2RAE
    rae = CC.rfc2rae(rfc[:,0],rfc[:,1],rfc[:,2],radar_az,radar_el,radar_clocking).values
    #go back
    test_pts = rfc.copy()
    rfc = CC.rae2rfc(rae[:,0],rae[:,1],rae[:,2],radar_az,radar_el,radar_clocking).values
    print(f'RFC <-> RAE Match: {np.isclose(test_pts,rfc).all()}')

    #RAE2ENU
    enu = CC.rae2enu(rae[:,0],rae[:,1],rae[:,2]).values
    #go back
    test_pts = rae.copy()
    rae = CC.enu2rae(enu[:,0],enu[:,1],enu[:,2]).values
    print(f'RAE <-> ENU Match: {np.isclose(test_pts,rae).all()}')

    #ENU2ECEF
    ecef = CC.enu2ecef(enu[:,0],enu[:,1],enu[:,2],radar_lat,radar_lon,radar_alt).values
    #FINAL TEST WITH ORIGINAL DATA
    print(f'ECEF -> ENU -> RAE -> LLA -> RFC -> RUV -> RAE -> ECEF -> LLA -> ENU -> RUV -> RFC -> LLA -> RUV -> ECEF -> RAE -> RFC -> ECEF -> RUV -> LLA -> RAE -> RUV -> ENU -> LLA -> ECEF -> RFC -> ENU -> RAE -> ENU -> RFC -> RAE -> ENU -> ECEF Match: {np.isclose(end_test_pts,ecef,rtol=1e-7,atol=1e-7).all()}')
    #Rounding to 7 decimal places.  I think after this many conversions, thats fair.
    assert np.isclose(end_test_pts,ecef,rtol=1e-7,atol=1e-7).all()






# NATIVE TESTS [These are the core for everything else to be internally consistent]
#%% Pos Test
def test_ecef2eci():
    x,y,z = ivory_coast
    t = random_time
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'t':[t]}
    else:
        data = {'x':x,'y':y,'z':z,'t':[t]*x.shape[0]}
    df = pd.DataFrame(data)
    ecef = CC.eci2ecef(df['x'],df['y'],df['z'],df['t']).values
    #go back
    test_pts = df.values[:,:-1]
    eci = CC.ecef2eci(ecef[:,0],ecef[:,1],ecef[:,2],df['t']).values
    print(f'ECEF <-> ECI Match: {np.isclose(test_pts,eci).all()}')
    assert np.isclose(test_pts,eci).all()

def test_ecef2enu():
    x,y,z = ivory_coast
    radar_lat,radar_lon,radar_alt = random_lla
    radar_az,radar_el,radar_clocking = random_boresight
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z]}
    else:
        data = {'x':x,'y':y,'z':z}
    df = pd.DataFrame(data)
    enu = CC.ecef2enu(df['x'],df['y'],df['z'],radar_lat,radar_lon,radar_alt).values
    #go back
    test_pts = df.values
    ecef = CC.enu2ecef(enu[:,0],enu[:,1],enu[:,2],radar_lat,radar_lon,radar_alt).values
    print(f'ECEF <-> ENU Match: {np.isclose(test_pts,ecef).all()}')
    assert np.isclose(test_pts,ecef).all()

def test_geodetic2ecef():
    x,y,z = ivory_coast
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z]}
    else:
        data = {'x':x,'y':y,'z':z}
    df = pd.DataFrame(data)
    lla = CC.ecef2geodetic(df['x'],df['y'],df['z']).values
    #go back
    test_pts = df.values
    ecef = CC.geodetic2ecef(lla[:,0],lla[:,1],lla[:,2]).values
    print(f'LLA <-> ECEF Match: {np.isclose(test_pts,ecef).all()}')
    assert np.isclose(test_pts,ecef).all()

def test_enu2rae():
    x,y,z = ivory_coast
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z]}
    else:
        data = {'x':x,'y':y,'z':z}
    df = pd.DataFrame(data)
    rae = CC.enu2rae(df['x'],df['y'],df['z']).values
    #go back
    test_pts = df.values
    enu = CC.rae2enu(rae[:,0],rae[:,1],rae[:,2]).values
    print(f'ENU <-> RAE Match: {np.isclose(test_pts,enu).all()}')
    assert np.isclose(test_pts,enu).all()

def test_ruv2rfc():
    x,y,z = ivory_coast
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z]}
    else:
        data = {'x':x,'y':y,'z':z}
    df = pd.DataFrame(data)
    ruv = CC.rfc2ruv(df['x'],df['y'],df['z']).values
    #go back
    test_pts = df.values
    rfc = CC.ruv2rfc(ruv[:,0],ruv[:,1],ruv[:,2]).values
    print(f'RUV <-> RFC Match: {np.isclose(test_pts,rfc).all()}')
    assert np.isclose(test_pts,rfc).all()

def test_rfc2enu():
    x,y,z = ivory_coast
    radar_az,radar_el,radar_clocking = random_boresight
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'vx':x,'vy':y,'vz':z}
    else:
        data = {'x':x,'y':y,'z':z}
    df = pd.DataFrame(data)
    enu = CC.rfc2enu(df['x'],df['y'],df['z'],radar_az,radar_el,radar_clocking).values
    #go back
    test_pts = df.values
    rfc = CC.enu2rfc(enu[:,0],enu[:,1],enu[:,2],radar_az,radar_el,radar_clocking).values
    print(f'RFC <-> ENU Match: {np.isclose(test_pts,rfc).all()}')
    assert np.isclose(test_pts,rfc).all()

#%% Vel Test

def test_ecef2eci_vel():
    x,y,z = ivory_coast
    t = random_time
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'t':[t],'vx':[x],'vy':[y],'vz':[z]}
    else:
        data = {'x':x,'y':y,'z':z,'t':[t]*x.shape[0],'vx':x,'vy':y,'vz':z}
    df = pd.DataFrame(data)
    ecef = CC.eci2ecef(df['x'],df['y'],df['z'],df['t'],df['vx'],df['vy'],df['vz']).values
    #go back
    test_pts = df.values[:,[0,1,2,4,5,6]]
    eci = CC.ecef2eci(ecef[:,0],ecef[:,1],ecef[:,2],df['t'],ecef[:,3],ecef[:,4],ecef[:,5]).values
    print(f'ECEF <-> ECI Vel Match: {np.isclose(test_pts,eci).all()}')
    assert np.isclose(test_pts,eci).all()

def test_ecef2enu_vel():
    x,y,z = ivory_coast
    radar_lat,radar_lon,radar_alt = random_lla
    radar_az,radar_el,radar_clocking = random_boresight
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'vx':[x],'vy':[y],'vz':[z]}
    else:
        data = {'x':x,'y':y,'z':z,'vx':x,'vy':y,'vz':z}
    df = pd.DataFrame(data)
    enu = CC.ecef2enu(df['x'],df['y'],df['z'],radar_lat,radar_lon,radar_alt,df['vx'],df['vy'],df['vz']).values
    #go back
    test_pts = df.values
    ecef = CC.enu2ecef(enu[:,0],enu[:,1],enu[:,2],radar_lat,radar_lon,radar_alt,enu[:,3],enu[:,4],enu[:,5]).values
    print(f'ECEF <-> ENU Vel Match: {np.isclose(test_pts,ecef).all()}')
    assert np.isclose(test_pts,ecef).all()

def test_geodetic2ecef_vel():
    x,y,z = ivory_coast
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'vx':[x]*.0013,'vy':[y]*.0023,'vz':[z]*.0033}
    else:
        data = {'x':x,'y':y,'z':z,'vx':x*.0013,'vy':y*.0023,'vz':z*.0033}
    df = pd.DataFrame(data)
    lla = CC.ecef2geodetic(df['x'],df['y'],df['z'],df['vx'],df['vy'],df['vz']).values
    #go back
    test_pts = df.values
    ecef = CC.geodetic2ecef(lla[:,0],lla[:,1],lla[:,2],lla[:,3],lla[:,4],lla[:,5]).values
    print(f'LLA <-> ECEF Vel Match: {np.isclose(test_pts,ecef).all()}')
    assert np.isclose(test_pts,ecef).all()

def test_enu2rae_vel():
    x,y,z = ivory_coast
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'vx':[x],'vy':[y],'vz':[z]}
    else:
        data = {'x':x,'y':y,'z':z,'vx':x,'vy':y,'vz':z}
    df = pd.DataFrame(data)
    rae = CC.enu2rae(df['x'],df['y'],df['z'],df['vx'],df['vy'],df['vz']).values
    #go back
    test_pts = df.values
    enu = CC.rae2enu(rae[:,0],rae[:,1],rae[:,2],rae[:,3],rae[:,4],rae[:,5]).values
    print(f'ENU <-> RAE Vel Match: {np.isclose(test_pts,enu).all()}')
    assert np.isclose(test_pts,enu).all()

def test_ruv2rfc_vel():
    x,y,z = ivory_coast
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'vx':[x],'vy':[y],'vz':[z]}
    else:
        data = {'x':x,'y':y,'z':z,'vx':x,'vy':y,'vz':z}
    df = pd.DataFrame(data)
    ruv = CC.rfc2ruv(df['x'],df['y'],df['z'],df['vx'],df['vy'],df['vz']).values
    #go back
    test_pts = df.values
    rfc = CC.ruv2rfc(ruv[:,0],ruv[:,1],ruv[:,2],ruv[:,3],ruv[:,4],ruv[:,5]).values
    print(f'RUV <-> RFC Vel Match: {np.isclose(test_pts,rfc).all()}')
    assert np.isclose(test_pts,rfc).all()

def test_rfc2enu_vel():
    x,y,z = ivory_coast
    radar_az,radar_el,radar_clocking = random_boresight
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'vx':[x],'vy':[y],'vz':[z]}
    else:
        data = {'x':x,'y':y,'z':z,'vx':x,'vy':y,'vz':z}
    df = pd.DataFrame(data)
    enu = CC.rfc2enu(df['x'],df['y'],df['z'],radar_az,radar_el,radar_clocking,df['vx'],df['vy'],df['vz']).values
    #go back
    test_pts = df.values
    rfc = CC.enu2rfc(enu[:,0],enu[:,1],enu[:,2],radar_az,radar_el,radar_clocking,enu[:,3],enu[:,4],enu[:,5]).values
    print(f'RFC <-> ENU Vel Match: {np.isclose(test_pts,rfc).all()}')
    assert np.isclose(test_pts,rfc).all()

#%% Acc Test

def test_ecef2eci_acc():
    x,y,z = ivory_coast
    t = random_time
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'t':[t],'vx':[x+10],'vy':[y+10],'vz':[z+10],'ax':[x+20],'ay':[y+20],'az':[z+20]}
    else:
        data = {'x':x,'y':y,'z':z,'t':[t]*x.shape[0],'vx':x+10,'vy':y+10,'vz':z+10,'ax':x+20,'ay':y+20,'az':z+20}
    df = pd.DataFrame(data)
    ecef = CC.eci2ecef(df['x'],df['y'],df['z'],df['t'],df['vx'],df['vy'],df['vz'],df['ax'],df['ay'],df['az']).values
    #go back
    test_pts = df.values[:,[0,1,2,4,5,6,7,8,9]]
    eci = CC.ecef2eci(ecef[:,0],ecef[:,1],ecef[:,2],df['t'],ecef[:,3],ecef[:,4],ecef[:,5],ecef[:,6],ecef[:,7],ecef[:,8]).values
    print(f'ECEF <-> ECI Acc Match: {np.isclose(test_pts,eci).all()}')
    assert np.isclose(test_pts,eci).all()

def test_ecef2enu_acc():
    x,y,z = ivory_coast
    radar_lat,radar_lon,radar_alt = random_lla
    radar_az,radar_el,radar_clocking = random_boresight
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'vx':[x+10],'vy':[y+10],'vz':[z+10],'ax':[x+20],'ay':[y+20],'az':[z+20]}
    else:
        data = {'x':x,'y':y,'z':z,'vx':x+10,'vy':y+10,'vz':z+10,'ax':x+20,'ay':y+20,'az':z+20}
    df = pd.DataFrame(data)
    enu = CC.ecef2enu(df['x'],df['y'],df['z'],radar_lat,radar_lon,radar_alt,df['vx'],df['vy'],df['vz'],df['ax'],df['ay'],df['az']).values
    #go back
    test_pts = df.values
    ecef = CC.enu2ecef(enu[:,0],enu[:,1],enu[:,2],radar_lat,radar_lon,radar_alt,enu[:,3],enu[:,4],enu[:,5],enu[:,6],enu[:,7],enu[:,8]).values
    print(f'ECEF <-> ENU Acc Match: {np.isclose(test_pts,ecef).all()}')
    assert np.isclose(test_pts,ecef).all()

def test_ecef2geodetic_acc():
    lla1 = pd.DataFrame( {'0':np.random.random(size)*80,
                '1':np.random.random(size)*80,
                '2':np.random.random(size)*10000,
                '3':np.random.random(size)*100,
                '4':np.random.random(size)*360,
            '5':np.random.random(size)*100,
            '6':np.random.random(size),
            '7':np.random.random(size),
            '8':np.random.random(size)}).values
    ecef = CC.geodetic2ecef(lla1[:,0],lla1[:,1],lla1[:,2],lla1[:,3],lla1[:,4],lla1[:,5],lla1[:,6],lla1[:,7],lla1[:,8]).values
    lla = CC.ecef2geodetic(ecef[:,0],ecef[:,1],ecef[:,2],ecef[:,3],ecef[:,4],ecef[:,5],ecef[:,6],ecef[:,7],ecef[:,8]).values
    assert np.isclose(lla,lla1,atol=1e-4).all()

def test_geodetic2ecef_acc():
    x,y,z = ivory_coast
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'vx':[x]*.0013,'vy':[y]*.0023,'vz':[z]*.0033,'ax':[x]*.00053,'ay':[y]*.00063,'az':[z]*.00073}
    else:
        data = {'x':x,'y':y,'z':z,'vx':x*.0013,'vy':y*.0023,'vz':z*.0033,'ax':x*.00053,'ay':y*.00063,'az':z*.00073}
    df = pd.DataFrame(data)
    lla = CC.ecef2geodetic(df['x'],df['y'],df['z'],df['vx'],df['vy'],df['vz'],df['ax'],df['ay'],df['az']).values
    #go back
    test_pts = df.values
    ecef = CC.geodetic2ecef(lla[:,0],lla[:,1],lla[:,2],lla[:,3],lla[:,4],lla[:,5],lla[:,6],lla[:,7],lla[:,8]).values
    print(f'LLA <-> ECEF Acc Match: {np.isclose(test_pts,ecef).all()}')
    assert np.isclose(test_pts,ecef).all()

def test_enu2rae_acc():
    x,y,z = ivory_coast
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'vx':[x*.1],'vy':[y*.1],'vz':[z*.1],'ax':[x*.3],'ay':[y*.3],'az':[z*.3]}
    else:
        data = {'x':x,'y':y,'z':z,'vx':x*.1,'vy':y*.1,'vz':z*.1,'ax':x*.3,'ay':y*.3,'az':z*.3}
    df = pd.DataFrame(data)
    rae = CC.enu2rae(df['x'],df['y'],df['z'],df['vx'],df['vy'],df['vz'],df['ax'],df['ay'],df['az']).values
    #go back
    test_pts = df.values
    enu = CC.rae2enu(rae[:,0],rae[:,1],rae[:,2],rae[:,3],rae[:,4],rae[:,5],rae[:,6],rae[:,7],rae[:,8]).values
    print(f'ENU <-> RAE Acc Match: {np.isclose(test_pts,enu).all()}')
    assert np.isclose(test_pts,enu).all()

def test_ruv2rfc_acc():
    x,y,z = ivory_coast
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'vx':[x+10],'vy':[y+10],'vz':[z+10],'ax':[x+20],'ay':[y+20],'az':[z+20]}
    else:
        data = {'x':x,'y':y,'z':z,'vx':x+10,'vy':y+10,'vz':z+10,'ax':x+20,'ay':y+20,'az':z+20}
    df = pd.DataFrame(data)
    ruv = CC.rfc2ruv(df['x'],df['y'],df['z'],df['vx'],df['vy'],df['vz'],df['ax'],df['ay'],df['az']).values
    #go back
    test_pts = df.values
    rfc = CC.ruv2rfc(ruv[:,0],ruv[:,1],ruv[:,2],ruv[:,3],ruv[:,4],ruv[:,5],ruv[:,6],ruv[:,7],ruv[:,8]).values
    print(f'RUV <-> RFC Acc Match: {np.isclose(test_pts,rfc).all()}')
    assert np.isclose(test_pts,rfc).all()

def test_rfc2enu_acc():
    x,y,z = ivory_coast
    radar_az,radar_el,radar_clocking = random_boresight
    if isinstance(x,(float,int)):
        data = {'x':[x],'y':[y],'z':[z],'vx':[x+10],'vy':[y+10],'vz':[z+10],'ax':[x+20],'ay':[y+20],'az':[z+20]}
    else:
        data = {'x':x,'y':y,'z':z,'vx':x+10,'vy':y+10,'vz':z+10,'ax':x+20,'ay':y+20,'az':z+20}
    df = pd.DataFrame(data)
    enu = CC.rfc2enu(df['x'],df['y'],df['z'],radar_az,radar_el,radar_clocking,df['vx'],df['vy'],df['vz'],df['ax'],df['ay'],df['az']).values
    #go back
    test_pts = df.values
    rfc = CC.enu2rfc(enu[:,0],enu[:,1],enu[:,2],radar_az,radar_el,radar_clocking,enu[:,3],enu[:,4],enu[:,5],enu[:,6],enu[:,7],enu[:,8]).values
    print(f'RFC <-> ENU Acc Match: {np.isclose(test_pts,rfc).all()}')
    assert np.isclose(test_pts,rfc).all()


if __name__ == '__main__':
    pass
    test_position_whole_enchilada()
    test_ecef2eci()
    test_ecef2enu()
    test_geodetic2ecef()
    test_enu2rae()
    test_ruv2rfc()
    test_rfc2enu()
    test_ecef2eci_vel()
    test_ecef2enu_vel()
    test_geodetic2ecef_vel()
    test_enu2rae_vel()
    test_ruv2rfc_vel()
    test_rfc2enu_vel()
    test_ecef2eci_acc()
    test_ecef2enu_acc()
    test_ecef2geodetic_acc()
    test_geodetic2ecef_acc()
    test_enu2rae_acc()
    test_ruv2rfc_acc()
    test_rfc2enu_acc()


