# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 22:28:33 2016

@author: andrew
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.ticker import LogLocator 
import matplotlib as mpl
from scipy.stats import gaussian_kde
import os 

plt.close('all')

def cart2sph(xyz_matrix):
    
    '''
    convert the Cartesian coordinates to spherical.
    
    Take as input [x1,y1,z1]
                  [x2,y2,z2]
                  ..........
                  [xN,yN,zN]
                  
     Return       [r1,azimuth1,elevation1]
                  [r2,azimuth2,elevation2]
                  ..........
                  [rN,azimuthN,elevation2]  
    '''
    
    if len(np.shape(xyz_matrix)) == 1:
        
        x = xyz_matrix[0]
        y = xyz_matrix[1]
        z = xyz_matrix[2]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y,x)
        elevation = np.arccos(z/r)
    
        sph_results = np.empty((3))
        sph_results[:] = np.NAN
        
        sph_results[0] = r
        sph_results[1] = azimuth
        sph_results[2] = elevation
        
    else:        
        
        x = xyz_matrix[:,0]
        y = xyz_matrix[:,1]
        z = xyz_matrix[:,2]
    
        sph_results = np.zeros((np.shape(xyz_matrix)[0],np.shape(xyz_matrix)[1]))
       
    
        r = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y,x)
        elevation = np.arccos(z/r)
        
        
        sph_results[:,0] = r
        sph_results[:,1] = azimuth
        sph_results[:,2] = elevation   
    
    return sph_results

def LoadSavedPositions(outputdirectory,runnum):
    
    #runfolder = 'AStudyInScarlet/old/run%d'%(runnum)
    #runfolder = 'AStudyInScarlet/optical_thick_off/run%d'%(runnum)
    runfolder = '%s/run%d'%(outputdirectory,runnum)
    
    x = np.load('%s/FinalTimestep_xpositions.npy'%(runfolder))
    y = np.load('%s/FinalTimestep_ypositions.npy'%(runfolder))
    z = np.load('%s/FinalTimestep_zpositions.npy'%(runfolder))
    r = np.load('%s/FinalTimestep_radius.npy'%(runfolder))
    #od = np.load('%s/OpticalDepthArray.npy'%(runfolder))
    od = np.load('%s/OpticalDepthArrayAzimuth.npy'%(runfolder))
    
    
    
    return x/R_sun_m,y/R_sun_m,z/R_sun_m,r,od
    #return x/R_sun_m,y/R_sun_m,z/R_sun_m,r,od
    
def LoadSaveState(outputdirectory,runnum,timestep):
    
    #runfolder = 'AStudyInScarlet/old/run%d'%(runnum)
    #runfolder = 'AStudyInScarlet/optical_thick_off/run%d'%(runnum)
    runfolder = '%s/run%d/SaveStates'%(outputdirectory,runnum)
    
    x = np.load('%s/Timestep%d_xpositions.npy'%(runfolder,timestep))
    y = np.load('%s/Timestep%d_ypositions.npy'%(runfolder,timestep))
    z = np.load('%s/Timestep%d_zpositions.npy'%(runfolder,timestep))
    r = np.load('%s/Timestep%d_radius.npy'%(runfolder,timestep))
    od = np.load('%s/Timestep%d_OpticalDepths.npy'%(runfolder,timestep))
    
    
    
    return x/R_sun_m,y/R_sun_m,z/R_sun_m,r,od
    #return x/R_sun_m,y/R_sun_m,z/R_sun_m,r,od
    
def PointsOnACircle(r,angles,originx=0,originy=0):
    '''
    Return x and y points for the positions
    on a circle given an origin and a vector of angles.
    
    Angles in radians.
    '''
    
    x = originx + r*np.cos(angles)
    y = originy + r*np.sin(angles)

    return x,y  
    
def PlotGridLines(xlist,ylist,radius_list):       
    
    '''
    Plot the last timestep topdown          
    '''    
    xlist = xlist*R_sun_m
    ylist = ylist*R_sun_m        
    
    
    plt.figure(figsize=((9,9)))       
    

    
    #plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
    plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(r)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.Normalize(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 

    cbar = plt.colorbar(ticks = LogLocator(subs=range(10)),fraction=0.0465, pad=0.02) # Draw log spaced tick marks on the colour bar 
    cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
    
    
    circle1=plt.Circle((0,0),radius=R_star,color='y')  # 0.7*R_sun as a rough estimate from wikipedia 
    plt.gcf().gca().add_artist(circle1)
    
    circle2=plt.Circle((0,-ap),radius=rp,color='k')  # 1.15 R_earth 1sigma limit from Brogi 2012
    plt.gcf().gca().add_artist(circle2)
    
    plt.ylabel('co-rotating y coordinate (m)')
    plt.xlabel('co-rotating x coordinate (m)')
    plt.title('top down view (zoom in)')     
    
   # plt.grid(True)
    
    
    plt.xlim(zoomtopdownxlim)
    plt.ylim(zoomtopdownylim)   
    

    plt.gca().set_aspect('equal')  
    
    #plt.savefig(os.path.join(OutputDirectory,'zoom_in_topdown.png'))
    
    for i in RadialBins:
        
        plt.plot(PointsOnACircle(i,np.linspace(0,2*np.pi,1000))[0],PointsOnACircle(i,np.linspace(0,2*np.pi,1000))[1],'k')
        
    for j in PhiBins:
        x2,y2 = PointsOnACircle(Rout,j*np.pi/180.)
        plt.plot([0,x2],[0,y2],'k')
        
def SaveSequenceParticeSize(OutputDirectory,runnum,timesteplist):
    
    
    
    topdownparticleRadiusSavePath = '%s/run%d/figures/TopDownParticleSize'%(OutputDirectory,runnum)
    topdownzoomparticleRadiusSavePath = '%s/run%d/figures/TopDownZoomParticleSize'%(OutputDirectory,runnum)

    sideviewxzparticleRadiusSavePath = '%s/run%d/figures/SideViewXZParticleSize'%(OutputDirectory,runnum)
    
    if not os.path.exists(topdownparticleRadiusSavePath):
        os.makedirs(topdownparticleRadiusSavePath)
        
    if not os.path.exists(topdownzoomparticleRadiusSavePath):
        os.makedirs(topdownzoomparticleRadiusSavePath)
    
    if not os.path.exists(sideviewxzparticleRadiusSavePath):
        os.makedirs(sideviewxzparticleRadiusSavePath)
        
    
    
    
    
    for i in timesteplist:
        
        x,y,z,r,od = LoadSaveState(OutputDirectory,runnum,i)
        
        #x,y,z,r,od = LoadSavedPositions(OutputDirectory,runnum)        
        
        print('Doing timestep %d of %d'%(i,len(timesteplist)))
        
        #plt.subplot(1,2,1)
        plt.figure(figsize=(10,10))
        plt.title('%s run %d timestep %d, No. part. = %.3e' % (OutputDirectory,runnum,i,len(x)))
        plt.scatter(x,y,edgecolors='none',s=5,c=np.array(r)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.Normalize(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
        plt.ylabel('Y (solar radii)')
        plt.xlabel('X (solar radii)')
        #plt.plot([-0.03,0.010],[-ap/R_sun_m,-ap/R_sun_m],'k-')
            ##plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=(np.min(radius_list)*1E6),vmax=(np.max(radius_list)*1E6))) ## Radius in um 
        
        #cbar = plt.colorbar(ticks = LogLocator(subs=range(10))) # Draw log spaced tick marks on the colour bar 
        cbar = plt.colorbar(fraction=0.04)
        cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
        plt.minorticks_on()
        plt.gca().set_aspect('equal')  
        
        #plt.savefig('%s/%s_run%d_timestep%d_TopDownParticleRadius.png'%(topdownparticleRadiusSavePath,OutputDirectory,runnum,i))
        
        plt.xlim(TopDownPlotLimsXLims)
        plt.ylim(TopDownPlotLimsYLims)
        
        plt.savefig('%s/%03d.png'%(topdownparticleRadiusSavePath,i))
        
        
#        #plt.title('%s run %d timestep %d zoom top down' % (OutputDirectory,runnum,i))
#        plt.scatter(x,y,edgecolors='none',s=5,c=np.array(r)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.Normalize(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
#        plt.ylabel('Y (solar radii)')
#        plt.xlabel('X (solar radii)')
#        #plt.plot([-0.03,0.010],[-ap/R_sun_m,-ap/R_sun_m],'k-')
#            ##plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=(np.min(radius_list)*1E6),vmax=(np.max(radius_list)*1E6))) ## Radius in um 
#        
#        #cbar = plt.colorbar(ticks = LogLocator(subs=range(10))) # Draw log spaced tick marks on the colour bar 
#        cbar = plt.colorbar(fraction=0.04)
#        cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
#        plt.minorticks_on()
#        plt.gca().set_aspect('equal')  
#        
        plt.xlim(TopDownZoomPlotLimsXLims)
        plt.ylim(TopDownZoomPlotLimsYLims)
        
        plt.savefig('%s/%03d.png'%(topdownzoomparticleRadiusSavePath,i))


        
        plt.close()

        #plt.subplot(1,2,2)
        
        plt.figure(figsize=(10,10))
        plt.title('%s run %d timestep %d' % (OutputDirectory,runnum,i))

        plt.scatter(x,z,edgecolors='none',s=5,c=np.array(r)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.Normalize(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
        
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
        
        plt.gca().set_aspect('auto')  
        
        #plt.gca().set_aspect('equal')  
        plt.minorticks_on()
        plt.ylabel('Z (solar radii)')
        plt.xlabel('X (solar radii)')
        #plt.ylim((-0.0036,+0.0036))
        #plt.xlim((-2.5,0.25))
        plt.xlim(SideViewXZ_Xlims)
        plt.ylim(SideViewXY_YLims)
        #plt.savefig('%s/%s_run%d_timestep%d_SideViewXZParticleRadius.png'%(sideviewxzparticleRadiusSavePath,OutputDirectory,runnum,i))
        plt.savefig('%s/%03d.png'%(sideviewxzparticleRadiusSavePath,i))
        
        plt.close()
        
        
    #plt.savefig(os.path.join(OutputDirectory,'zoom_in_topdown_GridLines.png'))
    

    
rp = 1.3E5  ## Gives a good tail height with density 
ap = 1961489736.7166538

R_sun_m =  6.957E8 #in m from WolframAlpha 

G = 6.67408E-11 # m^3/kg/s^2; # gravitational constant
ms=1.4e30 # mass of star in kg (0.7 solar masses)

PlanetDensity = 5427 # kg/m^3 (Density of Mercury)
R_sun_m =  6.957E8 #in m from WolframAlpha 

topdownxlim = (-2.5E9,0.5E9)
topdownylim = (-2.5E9,-1E9)

zoomtopdownxlim = (-0.3E8,0.3E8)
zoomtopdownylim = (-1991489736.7166538,-1931489736.7166538)


####################################################################

############### The original higher resolution radial bins 

nRScalingFactor = 5

nR = 200*nRScalingFactor

dR = 1.5E6/nRScalingFactor
RBinsBeforePlanet = 10*nRScalingFactor
RBinsAfterPlanet = nR-RBinsBeforePlanet




RadialBins = np.arange(ap-RBinsBeforePlanet*dR,ap+(RBinsAfterPlanet+1)*dR,dR) ### These are the grid boundaries so are of dimensions nR+1 (ie the edges of the bins so need 11 points for 10 bins)   

Rin = np.min(RadialBins) # m, from the top down plots, needs to be in AU for MCMAX3D (0.006AU) 
Rout = np.max(RadialBins)

#####################################################################

############### The radial bins of the epic run 

nR = 17500


dR = 3000.0

#RBinsBeforePlanet = (nR/2)
RBinsBeforePlanet = nR
RBinsAfterPlanet = nR-RBinsBeforePlanet


RadialBins = np.arange(ap-RBinsBeforePlanet*dR,ap+(RBinsAfterPlanet+1)*dR,dR) ### These are the grid boundaries so are of dimensions nR+1 (ie the edges of the bins so need 11 points for 10 bins)   

Rin = np.min(RadialBins) # m, from the top down plots, needs to be in AU for MCMAX3D (0.006AU) 
Rout = np.max(RadialBins)

#########################################################################




nPhi = 720#120 #120 # 360
PhiBins = np.linspace(0,360,nPhi+1)



R_star = 0.660*R_sun_m ## Rsun +- 0.060


## Christoph's estimate 
#rp=3.0e6 # planet radius in m from which particles are launched
#rp = 1.15*6371008.8 ## 1.15 R_earth 1sigma limit from Brogi 2012.  Conversion of Earth radii to m taken from WolframAlpha 
#rp = 4600.0*1E3 # upper limit from lack of secondary eclipse in van Werkhoven 
#rp = 4600.0*1E3/4
#rp = 100e3 #m 
#rp = 1 
#rp = 1E6
#rp = 3e6

rp = 1.3E5  ## Gives a good tail height with density 



#mp=5.0e23 # mass of planet in kg (1.5 Mercury masses)

mp = (4.0/3.0)*np.pi*(rp**3)*PlanetDensity
##mp = 5.000E+23

#MoonMass = 7.34767309E22 #kg ## From Google 
#
#if mp > 2*MoonMass:
#    raise Exception ### The upper limit of mass from radiative modelling is 2*Moon masses from models (physical limit from spectroscopy is 3 M*Jups) so this will stop unrealistic runs 

p=56467.584 # orbital period in seconds

c = 299792458 # speed of light in m/s

ParticleRemovalRadius = 1E-9
InitialParticleRadius = 1E-6

#ParticleRemovalRadius = 9.6650589427299361e-07

ap=(((p/2.0/np.pi)**2)*G*ms)**(1.0/3.0) # planet orbit radius in m
es=np.sqrt(2.0*G*mp/rp) # planetary escape velocity in m/s

########################
#runnum = 1395
#runnum = 1398
#runnum = 1420
#runnum = 1673
#runnum = 1420
#runnum = 1422
#runnum = 1671
#runnum = 1692 
#runnum = 1679
#runnum = 1681
#runnum = 1676
#runnum = 1673
#runnum = 1674
#runnum = 1820
#runnum = 1780
#runnum = 1826



#runnum = 1898
#runnum = 1873

#runnum = 1915



#OutputDirectory = 'TailDynamicsRunsFromRekere'
#runnum = 3

#OutputDirectory = 'output'
#runnum = 11

#OutputDirectory = 'JobLibTests'
#runnum = 40

#OutputDirectory = 'TailDynamicsRunsFromRekere'
#runnum = 7

#OutputDirectory = 'output'
#runnum = 11

#OutputDirectory = 'output'
#runnum = 12

#OutputDirectory = 'TailDynamicsRunsFromRekere'
#runnum = 8

#OutputDirectory = 'output'
#runnum = 15


#OutputDirectory = 'TailDynamicsRunsFromRekere'
#runnum = 9

#OutputDirectory = 'output'
#runnum = 21

#OutputDirectory = 'output'
#runnum = 32

#OutputDirectory = 'TailDynamicsRunsFromRekere'
#runnum = 10

#OutputDirectory = 'TailDynamicsRunsFromRekere'
#runnum = 11

#OutputDirectory = 'output'
#runnum = 37

#OutputDirectory = 'output'
#runnum = 52

#OutputDirectory = 'TailDynamicsRunsFromRekere'
#runnum = 12


#OutputDirectory = 'output'
#runnum = 53

#OutputDirectory = 'TailDynamicsRunsFromRekere'
#runnum = 13

#OutputDirectory = 'output'
#runnum = 70

#OutputDirectory = 'TailDynamicsRunsFromPara'   ### outburst at start 
#runnum = 6


#OutputDirectory = 'output'   ### outburst in middle 
#runnum = 71


#OutputDirectory = 'TailDynamicsRunsFromRekere'   ### outburst at end 
#runnum = 14

#OutputDirectory = 'TailDynamicsRunsFromRekere'   ### Rekere run 15 more extreme ejection in middle 
#runnum = 15

#OutputDirectory = 'output' ### The attempt at epic hopeless run 
#runnum = 89

#OutputDirectory = 'output' 
#runnum = 52

#OutputDirectory = 'output' 
#runnum = 96

#OutputDirectory = 'output' 
#runnum = 97
#
#TopDownPlotLimsXLims = (-3.0,0.5)
#TopDownPlotLimsYLims = (-3.2,-1.6)
#
#TopDownZoomPlotLimsXLims = (-0.08,0.02)
#TopDownZoomPlotLimsYLims = (-2.88,-2.8)
#
#SideViewXZ_Xlims = (-0.012,+0.012)
#SideViewXY_YLims = (-3,0.25)

#OutputDirectory = 'output' 
#runnum = 98
#
#TopDownPlotLimsXLims = (-0.5,0.2)
#TopDownPlotLimsYLims = (-3.0,-2.75)
#
#TopDownZoomPlotLimsXLims = (-0.3,0.1)
#TopDownZoomPlotLimsYLims = (-2.78,-2.9)
#
#SideViewXZ_Xlims = (-0.01,+0.01)
#SideViewXY_YLims = (-0.5,0.2)

#OutputDirectory = 'output' 
#runnum = 99
#
#TopDownPlotLimsXLims = (-3.0,0.5)
#TopDownPlotLimsYLims = (-3.2,-1.6)
#
#TopDownZoomPlotLimsXLims = (-0.08,0.02)
#TopDownZoomPlotLimsYLims = (-2.88,-2.8)
#
#SideViewXZ_Xlims = (-0.012,+0.012)
#SideViewXY_YLims = (-3,0.25)

#OutputDirectory = 'TailDynamicsRunsFromRekere' 
#runnum = 16

#TopDownPlotLimsXLims = (-3.0,0.5)
#TopDownPlotLimsYLims = (-3.2,-1.6)
#
#TopDownZoomPlotLimsXLims = (-0.08,0.02)
#TopDownZoomPlotLimsYLims = (-2.88,-2.8)
#
#SideViewXZ_Xlims = (-0.012,+0.012)
#SideViewXY_YLims = (-3,0.25)

#OutputDirectory = 'TailDynamicsRunsFromRekere' 
#runnum = 17

#OutputDirectory = 'TailDynamicsRunsFromRekere' 
#runnum = 18

#OutputDirectory = 'output' 
#runnum = 101

#OutputDirectory = 'output' 
#runnum = 103

#OutputDirectory = 'output' 
#runnum = 104

#OutputDirectory = 'output' 
#runnum = 114

#OutputDirectory = 'output' 
#runnum = 115

#OutputDirectory = 'TailDynamicsRunsFromPara' 
#runnum = 7

#OutputDirectory = 'TailDynamicsRunsFromRekere' 
#runnum = 19

#OutputDirectory = 'output' 
#runnum = 116

#OutputDirectory = 'output' 
#runnum = 126
#
#TopDownPlotLimsXLims = (-0.2,0.20)
#TopDownPlotLimsYLims = (-2.83,-2.80)
#
#TopDownZoomPlotLimsXLims = (-0.1,0.1)
#TopDownZoomPlotLimsYLims = (-2.820,-2.815)
#
#SideViewXZ_Xlims = (-0.20,0.20)
#SideViewXY_YLims = (-0.0006,0.0006)

#OutputDirectory = 'output' 
#runnum = 127



OutputDirectory = 'output' 
runnum = 129

#TopDownPlotLimsXLims = (-0.6,0.2)
#TopDownPlotLimsYLims = (-3,-2.8)
#
#TopDownZoomPlotLimsXLims = (-0.3,0.30)
#TopDownZoomPlotLimsYLims = (-2.85,-2.5)
#
#SideViewXZ_Xlims = (-0.010,0.010)
#SideViewXY_YLims = (-2.5,0.5)

TopDownPlotLimsXLims = (-3,0.5)
TopDownPlotLimsYLims = (-3.2,-1.8)

TopDownZoomPlotLimsXLims = (-0.05,0.05)
TopDownZoomPlotLimsYLims = (-2.85,-2.805)

SideViewXZ_Xlims = (-0.6,0.1)
SideViewXY_YLims = (-0.015,0.015)


#x,y,z,r,od = LoadSavedPositions(OutputDirectory,runnum)
#x,y,z,r,od = LoadSaveState(OutputDirectory,runnum,2999)




#SaveSequenceParticeSize(OutputDirectory,runnum,range(0,465,1))
#SaveSequenceParticeSize(OutputDirectory,runnum,range(0,500,1))

#SaveSequenceParticeSize(OutputDirectory,runnum,[2999])
SaveSequenceParticeSize(OutputDirectory,runnum,range(0,3000,5))
#SaveSequenceParticeSize(OutputDirectory,runnum,range(0,500,2))



#x,y,z,r,od = LoadSavedPositions(runnum)

#print np.min(od)


#xyzarray = np.zeros((len(x),3))
#xyzarray[:,0] = x 
#xyzarray[:,1] = y
#xyzarray[:,2] = z
#
#spharray = cart2sph(xyzarray)




#plt.scatter(x,y,edgecolors='none',s=3)
#
#plt.gca().set_aspect('equal')  

#################################################################

##### For the nice double plot 


#fig = plt.figure(figsize=(16,16))
#ax1 = fig.add_axes([0.05, 0.05, 0.4, 0.2])
#
#plt.scatter(x,y,edgecolors='none',s=5,c=np.array(r)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
#plt.title('%s run %d' % (OutputDirectory,runnum))
#plt.scatter(x,y,edgecolors='none',s=5,c=np.array(r)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.Normalize(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
#plt.ylabel('Y (solar radii)')
#plt.xlabel('X (solar radii)')
#    ##plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=(np.min(radius_list)*1E6),vmax=(np.max(radius_list)*1E6))) ## Radius in um 
#
##cbar = plt.colorbar(ticks = LogLocator(subs=range(10))) # Draw log spaced tick marks on the colour bar 
#cbar = plt.colorbar()
#cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
#plt.minorticks_on()
#plt.gca().set_aspect('equal')  
#
#ax2 = fig.add_axes([0.52, 0.05, 0.4,0.2])
##plt.scatter(x,z,edgecolors='none',s=5,c=np.array(r)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
#plt.scatter(x,z,edgecolors='none',s=5,c=np.array(r)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.Normalize(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
#
#
##cbar = plt.colorbar(ticks = LogLocator(subs=range(10))) # Draw log spaced tick marks on the colour bar 
##cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
#
#cbar = plt.colorbar()
#cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
#
##plt.gca().set_aspect('equal')  
#plt.minorticks_on()
#plt.ylabel('Z (solar radii)')
#plt.xlabel('X (solar radii)')
##plt.ylim((-0.0036,+0.0036))
##plt.xlim((-2.5,0.25))
#plt.ylim((-0.012,+0.012))
#plt.xlim((-3,0.25))
#
##plt.ylim((-0.005,+0.005))
##plt.xlim((-0.25,0.25))
#plt.draw()
##plt.savefig('TopSideRun%d'%(runnum),dpi=600)

############################################################################

#fig = plt.figure(figsize=(16,16))


###plt.scatter(x,y,edgecolors='none',s=5,c=np.array(r)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
#plt.figure()
#plt.title('%s run %d' % (OutputDirectory,runnum))
#plt.scatter(x,y,edgecolors='none',s=5,c=np.array(r)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.Normalize(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
#plt.ylabel('Y (solar radii)')
#        
#plt.xlabel('X (solar radii)')
#plt.plot([-0.03,0.010],[-ap/R_sun_m,-ap/R_sun_m],'k-')
#    ##plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=(np.min(radius_list)*1E6),vmax=(np.max(radius_list)*1E6))) ## Radius in um 
#
##cbar = plt.colorbar(ticks = LogLocator(subs=range(10))) # Draw log spaced tick marks on the colour bar 
#cbar = plt.colorbar()
#cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
#plt.minorticks_on()
#plt.gca().set_aspect('equal')  
#
#plt.figure()
##plt.scatter(x,z,edgecolors='none',s=5,c=np.array(r)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
#plt.scatter(x,z,edgecolors='none',s=5,c=np.array(r)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.Normalize(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
#plt.title('%s run %d' % (OutputDirectory,runnum))
#
#cbar = plt.colorbar()
#cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
#
##plt.gca().set_aspect('equal')  
#plt.minorticks_on()
#plt.ylabel('Z (solar radii)')
#plt.xlabel('X (solar radii)')
##plt.ylim((-0.0036,+0.0036))
##plt.xlim((-2.5,0.25))
#plt.ylim((-0.012,+0.012))
#plt.xlim((-3,0.25))
#
##plt.ylim((-0.005,+0.005))
##plt.xlim((-0.25,0.25))
#plt.draw()
##plt.savefig('TopSideRun%d'%(runnum),dpi=600)
#
#plt.figure()
##plt.plot(od[:,0]*180./np.pi,od[:,1],'.')
#plt.title('%s run %d' % (OutputDirectory,runnum))
#
#
#plt.plot(spharray[:,1]*180/np.pi,od,'.')
#plt.ylabel('opitcal depth')
#plt.xlabel('azimuthal angle (degs)')
#
#plt.figure()
##plt.plot(od[:,0]*180./np.pi,od[:,1],'.')
#plt.title('%s run %d' % (OutputDirectory,runnum))
#
#
#plt.plot(spharray[:,1]*180/np.pi,np.exp(-od),'.')
#plt.ylabel('Tramsmission')
#plt.xlabel('azimuthal angle (degs)')

#PlotGridLines(x,y,r)

#xy = np.vstack([x,y])
#densols = gaussian_kde(xy)(xy)
#
#xz = np.vstack([x,y])
#densolsxz = gaussian_kde(xy)(xy)
#
#dencols2 = densols**(0.5)
#
#dencolsxz = densols**(0.5)
#
################ Figure 2 
#
#fig = plt.figure(figsize=(16,16))
#ax1 = fig.add_axes([0.05, 0.05, 0.4, 0.2])
#
#plt.scatter(x,y,c=dencols2,s=1,edgecolor='')
#plt.ylabel('Y (solar radii)')
#plt.xlabel('X (solar radii)')
#
#cbar = plt.colorbar()
#cbar.ax.set_ylabel('Square root of particle number \n density [arbitrary units]')
#    
#
#plt.minorticks_on()
#plt.gca().set_aspect('equal')  
#
#ax2 = fig.add_axes([0.52, 0.05, 0.4,0.2])
#plt.scatter(x,z,c=dencols2,s=1,edgecolor='')
#plt.minorticks_on()
#plt.ylabel('Z (solar radii)')
#plt.xlabel('X (solar radii)')
##plt.ylim((-0.0036,+0.0036))
##plt.xlim((-2.5,0.25))
#plt.ylim((-0.012,+0.012))
#plt.xlim((-3,0.25))
##plt.ylim((-0.005,+0.005))
##plt.xlim((-0.25,0.25))
#
#cbar = plt.colorbar()
#cbar.ax.set_ylabel('Square root of particle number \n density [arbitrary units]')
#
#plt.draw()

#plt.figure()
#plt.scatter(x,y,c=densols2,s=1,edgecolor='')
#plt.colorbar()
#plt.figure()
#plt.plot(spharray[:,1],od[:,1],'.')

