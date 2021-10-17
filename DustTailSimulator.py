# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 23:12:30 2016

@author: andrew
"""
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import time 
from matplotlib.ticker import LogLocator 
from scipy.integrate import tplquad
import astropy.io.fits as pyfits 
import pickle 
import gzip 
import os
import random
from scipy.interpolate import interp1d 
from scipy.integrate import odeint 
#from ShapelyTest import GeometricAreaTransitDepth
#from joblib import Parallel, delayed
import multiprocessing



starttime = time.time()

class Particle:
    
    '''
    A general class for a particle object.
    It stores all information for the particle.
    
    
    Note: the accleration_history is not the net acceleration at each time step.
    It is instead the record of results from whenever the 
    calculate_acceleration function is called, which is 3 times per 
    time step     
    
    '''
    
    def __init__(self,name,initial_position,initial_velocity,initial_radius,creation_timestep):  
        
        self.name = name ## an identifying number for the particle 
        
        self.position = initial_position
        self.velocity = initial_velocity
        self.radius = initial_radius # in metres
        
        self.od_1 = 0.0
        self.od = 0.0
        self.beta = 0.037790310000000090
        
        ## Qpr = radiation pressure coefficient.
        ## Qpr = Qabs + Qsca(1-<cos alpha>) (pg 6 Burns+ 1979)
        ## depends on the composition of the particle 
        ## beta/Qpr is of order 1 for radii on order 0.5um 
        
        self.Qpr = 0.3 
        
                
        self.position_history = []
        self.velocity_history = []
        self.radius_history = []
        
        self.timestep_history = []
        
        self.od_1_history = []
        self.transmittance_history = []
        self.beta_history = []     
        
        ### Note: this is not the net acceleration at each time step.
        ### It is instead the record of results from whenever the 
        ### calculate_acceleration function is called, which is 3 times per 
        ### time step 
        self.acceleration_history = [] 
        self.AccelerationDueToPlanet_history = [] 
        
        self.position_history.append(initial_position)
        self.velocity_history.append(initial_velocity)
        self.timestep_history.append(creation_timestep) 
        
        self.age = 0 # the age of the particle in timesteps, starts at 0
        
        self.od_1_history.append(self.od_1)
        self.transmittance_history.append(self.od)
  
        
    def __repr__(self):
        return 'position = '+str(self.position)+'\n velocity = '+str(self.velocity)
    def __str__(self):
        return 'position = '+str(self.position)+'\n velocity = '+str(self.velocity)
    
        
    def update_position(self,new_pos):                
        self.position = new_pos
        if self.name in ParticlesToTrack:
            self.position_history.append(new_pos)        
        
    def update_velocity(self,new_vel):
        
        if self.name in ParticlesToTrack:
            self.acceleration_history.append((new_vel - self.velocity)/dt) ## nett acceleration from change in velocity over timestep  
            self.velocity_history.append(new_vel)
            
        self.velocity = new_vel         
        
    def update_radius(self,new_radius):
        self.radius = new_radius
        if self.name in ParticlesToTrack:
            self.radius_history.append(new_radius)        
            
    def update_planet_acceleration_history(self,new_planet_acceleration):
        
        if self.name in ParticlesToTrack:
            self.AccelerationDueToPlanet_history.append(new_planet_acceleration)  
            

        
    def update_age(self):
        self.age = self.age + 1
        
    def update_timestep_history(self,timestep):  
        if self.name in ParticlesToTrack:
            self.timestep_history.append(timestep)
                
   
    def update_od(self,new_od_1):
        
        self.od_1 = new_od_1
        self.transmittance = np.exp(-self.od_1) # exp(-tau) so I think this is like the transmittance T.  This odm is currently a global which might not be ideal for it to be used in this object 
        
        if self.name in ParticlesToTrack:
            
            self.od_1_history.append(self.od_1)
            self.transmittance_history.append(self.transmittance)        
        
               
    def update_beta(self,new_beta):
        
        self.beta = new_beta
        
        if self.name in ParticlesToTrack:
        
            self.beta_history.append(new_beta)  
    
#def calculate_acceleration(particle_object,omega,last_pos=False):
#    
#    '''
#    calcuate the acceleration from the equation in pg 120 of Rik's thesis 
#    
#    r is the position vector 
#    v is the velocity vector 
#    
#    NOTE: THIS IS ASSUMING THAT THE OMEGA IS THE CONSTANT OMEGA OF THE PLANET'S 
#    ORBIT.  IF IT NEEDS TO VARY WITH THE PARTILE, IT NEEDS TO BE IN THE OBJECT
#    '''        
#    
#    beta = particle_object.beta
#    
#    if last_pos==True:
#        
#        #print 'taking r to be the previous position'
#        r = particle_object.position_history[-2] ## when the position is updated, the current value is appended to the position_history list so this needs to be -2 to get the second to last value 
#        
#    
#    else:
#        r = particle_object.position
#        
#        
#    v = particle_object.velocity     
#    
#    
#    ### Acceleration from radiation pressure and gravity of star (beta)        
#    acceleration = -(G*ms*(1.0-beta)/norm(r)**3)*r - np.cross(2*omega,v) - np.cross(omega,(np.cross(omega,r))) 
##    print 'acceleration'
##    print acceleration     
#    
#    ### Gravitational acceleration from planet 
#    
#    distance_to_planet = np.linalg.norm(r-planet_pos)
#    planet_accel_direction = (r-planet_pos)/distance_to_planet ## Unit vector in direction from planet to particle
#    
#    planet_acceleration = (-G*mp/distance_to_planet**2.)*planet_accel_direction
#    
#    particle_object.acceleration_history.append(acceleration) 
#    particle_object.AccelerationDueToPlanet_history.append(planet_acceleration) 
#
#    nett_acceleration = acceleration + planet_acceleration
#    
#    #return nett_acceleration  
#    return planet_acceleration 
    
def Calculate_gas_drag_accel(vdust,R_dust,M_dust,distance_from_planet,direction_towards_planet,n0,R0):    
    
    '''
    
    Taken from Kramer 2015.
    
    
    A basic 1/r**2 relation in the decrease of gas density (N_gas)
    based on a mention in Kramer 2015 that it should go as 1/r**2
    and using the first part of the neutral gas density (eqn 3) of Edberg+16
    (this equantion also includes a factor exp(r-R0)/L to also exponentially 
    decrease the neutral density as the gas gets ionised)
    
    n0 is the density at the surface, given as n0 = 1.4E16 m^{-3}
    
    R0 is the radius of the planet (comet in this case)
    '''
    
    
    
    N_gas = n0*(R0/distance_from_planet)**2
    
    c_gas_drag = k_B*N_gas*np.pi*R_dust**2/M_dust 
    
    vgas = np.sqrt(k_B*Tgas/mgas)*direction_towards_planet*(-1) # to have velocity away from the planet 
    
    agas = (c_gas_drag*(vdust-vgas)*norm(vdust-vgas))/(norm(vgas)**2)
    
    return agas 
    
def calculate_acceleration2(r,v,beta,omega,R_dust,M_dust):
    
    '''
    calcuate the acceleration from the equation in pg 120 of Rik's thesis 
    
    r is the position vector 
    v is the velocity vector 
    
    NOTE: THIS IS ASSUMING THAT THE OMEGA IS THE CONSTANT OMEGA OF THE PLANET'S 
    ORBIT.  IF IT NEEDS TO VARY WITH THE PARTILE, IT NEEDS TO BE IN THE OBJECT
    '''        
    
    
    ### Acceleration from radiation pressure and gravity of star (beta)        
    acceleration = -(G*ms*(1.0-beta)/norm(r)**3)*r - np.cross(2*omega,v) - np.cross(omega,(np.cross(omega,r))) 
    
    planet_acceleration = AccelerationDueToPlanet(r)
    #planet_acceleration = np.array([0,0,0])    
    
    if IncludeGasDragAcceleration:
    
        acceleration_from_gas_drag = Calculate_gas_drag_accel(v,R_dust,M_dust,distance_to_planet,planet_accel_direction,Ngas_surface,rp)
   
        nett_acceleration = acceleration + planet_acceleration + acceleration_from_gas_drag
        
    if not IncludeGasDragAcceleration:
        
        nett_acceleration = acceleration + planet_acceleration 
        
        
        
    
#    with open("%s/corotatingbeta_planetgrav_gas_netaccel.txt"%(OutputDirectory), "a") as myfile:
#        myfile.write('%s    %s    %s    %s\n\n'%(acceleration,planet_acceleration,acceleration_from_gas_drag,nett_acceleration))
    
    return nett_acceleration  
    #return planet_acceleration 
    #return acceleration
    
def AccelerationDueToPlanet(r):
    
    '''
    Return the acceleration of the particle towards the planet.
    Only input is the particle position, r however 
    it also depends on the constants, planet mass and planet position
    '''
    
    ### Gravitational acceleration from planet     
    distance_to_planet = np.linalg.norm(r-planet_pos)
    planet_accel_direction = (r-planet_pos)/distance_to_planet ## Unit vector in direction from planet to particle
    
    planet_acceleration = (-G*mp/distance_to_planet**2.)*planet_accel_direction
    
    return planet_acceleration  
     
    
    
#def accel_due_to_planet(position):
#    
#    r = position
#    
#    distance_to_planet = np.linalg.norm(r-planet_pos)
#     
#    planet_accel_direction = (r-planet_pos)/distance_to_planet ## Unit vector in direction from planet to particle
#     
#    planet_acceleration = (-G*mp/distance_to_planet**2.)*planet_accel_direction
#    #particle_object.acceleration_history.append(acceleration) 
#     
#    return planet_acceleration
    
#def Verlet(particle_object, dt, calculate_acceleration,omega):
#    """Return new position and velocity from current values, time step and acceleration.
#    Parameters:
#        r is a numpy array giving the current position vector
#        v is a numpy array giving the current velocity vector
#        dt is a float value giving the length of the integration time step
#        a is a function which takes r, v omega as parameters and returns the acceleration vector as an array
#
#    Works with arrays of any dimension as long as they're all the same.
#    """
#      
#    r = particle_object.position
#    v = particle_object.velocity 
# 
#    # Deceptively simple (read about Velocity Verlet on wikipedia)
#    r_new = r + v*dt + calculate_acceleration(particle_object,omega)
#    particle_object.update_position(r_new)    
#    
#    v_new = v + (calculate_acceleration(particle_object,omega,last_pos=True) + calculate_acceleration(particle_object,omega))/2 * dt    
#     
#    
#    particle_object.update_velocity(v_new)
#    
#    ### add one to the particle's age 
#    particle_object.update_age()    
#    
#    particle_object.update_timestep_history(timestep)
#        
#    return None 
    
def Verlet2(particle_object, dt,omega):
    """
    Gets all relevant info from the particle and 
    calls the function VerletHope2 with these inputs to actually calculate 
    the acceleration and new veleocity.
    
    Then takes these new values and updates the particle object 
    """
      
    r = particle_object.position
    v = particle_object.velocity 
    beta = particle_object.beta 
    particle_radius = particle_object.radius
    particle_mass_kg = ((4.0/3.0)*np.pi*particle_radius**3)*particle_density ## Assuming spherical particles 
    
    ### updating the acceleration due to planet history with the position value before it is moved so that it is the acceleration due to the planet at the position that led to this nett acceleration.  
    particle_object.update_planet_acceleration_history(AccelerationDueToPlanet(r))
    
    (r_new, v_new) = VerletHope2(r,v,beta,dt,particle_radius,particle_mass_kg)
 
    particle_object.update_position(r_new)   
    
    particle_object.update_velocity(v_new)
    
    ### add one to the particle's age 
    particle_object.update_age()    
    
    particle_object.update_timestep_history(timestep)
        
    return None 
    
#def VerletHope(r, v, dt, a):
#	"""Return new position and velocity from current values, time step and acceleration.
#
#	Parameters:
#	   r is a numpy array giving the current position vector
#	   v is a numpy array giving the current velocity vector
#	   dt is a float value giving the length of the integration time step
#	   a is a function which takes x as a parameter and returns the acceleration vector as an array
#
#	Works with arrays of any dimension as long as they're all the same.
#	"""
#	# Deceptively simple (read about Velocity Verlet on wikipedia)
#	r_new = r + v*dt + a(r)*dt**2/2
#	v_new = v + (a(r) + a(r_new))/2 * dt
#	return (r_new, v_new)
 
def VerletHope2(r, v, beta,dt,R_dust,M_dust):
    """Return new position and velocity from current values, time step and acceleration.

    Parameters:
        r is a numpy array giving the current position vector
        v is a numpy array giving the current velocity vector
        dt is a float value giving the length of the integration time step
        a is a function which takes x as a parameter and returns the acceleration vector as an array

    Works with arrays of any dimension as long as they're all the same.
    """
    # Deceptively simple (read about Velocity Verlet on wikipedia)
    r_new = r + v*dt + calculate_acceleration2(r,v,beta,omega,R_dust,M_dust)*dt**2/2
    v_new = v + (calculate_acceleration2(r,v,beta,omega,R_dust,M_dust) + calculate_acceleration2(r_new,v,beta,omega,R_dust,M_dust))/2 * dt
    
    return (r_new, v_new)

 
#def Verlet(particle_object, dt, calculate_acceleration,omega):
#    """Return new position and velocity from current values, time step and acceleration.
#    Parameters:
#        r is a numpy array giving the current position vector
#        v is a numpy array giving the current velocity vector
#        dt is a float value giving the length of the integration time step
#        a is a function which takes r, v omega as parameters and returns the acceleration vector as an array
#
#    Works with arrays of any dimension as long as they're all the same.
#    """
#      
#    r = particle_object.position
#    v = particle_object.velocity 
# 
#    # Deceptively simple (read about Velocity Verlet on wikipedia)
#    r_new = r + v*dt + calculate_acceleration(particle_object,omega)*dt**2/2
#    particle_object.update_position(r_new)    
#    
#    v_new = v + (calculate_acceleration(particle_object,omega) + calculate_acceleration(particle_object,omega))/2 * dt    
#    particle_object.update_velocity(v_new)
#    
#    ### add one to the particle's age 
#    particle_object.update_age()    
#    
#    particle_object.update_timestep_history(timestep)
#        
#    return None 
 


def plot_particles(particle_list):
    
    plt.figure()
    
    cmap = mpl.cm.spectral
        
    for i in range(len(particle_list)):        
        
        
    #### Plot in the xy plane (looking down)                
        
        r_matrix = np.array(particle_list[i].position_history)
        
        x = r_matrix[:,0]
        y = r_matrix[:,1]
        
        plt.plot(x, y,'.',color=cmap(float(i)/len(particle_list)))
      
        
    circle1=plt.Circle((0,0),radius=R_star,color='y')  # 0.7*R_sun as a rough estimate from wikipedia 
    plt.gcf().gca().add_artist(circle1)
    
    circle2=plt.Circle((0,-ap),radius=rp,color='r')  # 1.15 R_earth 1sigma limit from Brogi 2012
    plt.gcf().gca().add_artist(circle2)
    
    plt.ylabel('co-rotating y coordinate (m)')
    plt.xlabel('co-rotating x coordinate (m)')
    plt.title('top down view (equal axes)')
    
    plt.gca().set_aspect('equal', adjustable='box')    
    
    return None 
    
def plot_particles_side_view(particle_list):
    
    plt.figure()
    
    cmap = mpl.cm.spectral
        
    for i in range(len(particle_list)):        
        
        
    #### Plot in the xy plane (looking down)                
        
        r_matrix = np.array(particle_list[i].position_history)
        
        x = r_matrix[:,0]
        y = r_matrix[:,1]
        z = r_matrix[:,2]                
        
        plt.plot(y, z,'.',color=cmap(float(i)/len(particle_list)))      
        
    circle1=plt.Circle((0,0),radius=R_star,color='y')  # 0.7*R_sun as a rough estimate from wikipedia 
    plt.gcf().gca().add_artist(circle1)
    
    circle2=plt.Circle((0,0),radius=rp,color='r')  # 1.15 R_earth 1sigma limit from Brogi 2012
    plt.gcf().gca().add_artist(circle2)
    
    plt.ylabel('co-rotating z coordinate (m)')
    plt.xlabel('co-rotating y coordinate (m)')
    plt.title('Side view (unequal axes)')
    
    #plt.gca().set_aspect('equal', adjustable='box')    
    
    return None 
    
def plot_particles3D(particle_list):
    
    cmap = mpl.cm.spectral    
        
    fig = plt.figure()    
    ax = fig.add_subplot(111,projection='3d')   
    
    ax.set_xlabel('co-rotating x coordinate (m)')
    ax.set_ylabel('co-rotating y coordinate (m)')
    ax.set_zlabel('co-rotating z coordinate (m)')
    
    ax.set_aspect('equal')
    
    for i in range(len(particle_list)):
        
        r_matrix = np.array(particle_list[i].position_history)
        
        ax.scatter(r_matrix[:,0], r_matrix[:,1], r_matrix[:,2], '.',color=cmap(float(i)/len(particle_list)))
                
    
    fig, ax = plt.plot()
    
def plot_last_particles3D(particle_list,elevation=0.,azimuth=0.):    
    
    '''
    Plot the final the last position of particles in the particle list in 3D 
    with the grid that is used to create the density file for MCMax3D 
    '''

    timestep = ns*no-1
        
    xlist = []
    ylist = []           
    zlist = []
    radius_list = []
            
    print 'doing timestep %d' % (timestep)
    
    for particle in particle_list:            
                            
        x = particle.position[0]
        y = particle.position[1]
        z = particle.position[2]
        radius = particle.radius
        
        xlist.append(x)
        ylist.append(y)
        zlist.append(z)
        radius_list.append(radius)
            
    circlex, circley = PointsOnACircle(Rin,np.linspace(0,2*np.pi,1000))
                    
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xlist, ylist, zlist)
    
    for i in RadialBins:
        
        ax.plot(PointsOnACircle(i,np.linspace(0,2*np.pi,1000))[0],PointsOnACircle(i,np.linspace(0,2*np.pi,1000))[1])
        
#    for j in PhiBins:
#        x2,y2 = PointsOnACircle(Rout,j*np.pi/180.)
#        ax.plot([0,x2],[0,y2])
#        
#    for k in ThetaBins:
#        x2,y2 = PointsOnACircle(Rout,(k-90)*np.pi/180.)
#        ax.plot([0,-x2],[0,-y2],zdir='x')
    
    #ax.plot([0,-2E9],[-2.2E9,-2.2E9])    
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    #ax.set_zlim(((-3E9,3E9)))    
    
    ax.view_init(elev=elevation,azim=azimuth)
    
    plt.show()
    
    plt.savefig('particle_grid_elevation%.0f_azimuth%.0f'%(elevation,azimuth))
    
                
   

    
def GetAllxyzCoords(ParticleList):
    
    '''
    
    Make a matrix [x1,y1,z1]
                  [x2,y2,z2]
                  ..........
                  [xN,yN,zN]
                  
    of all x,y,z positions of all particles 
    
    '''
    
    
    N = len(ParticleList)
    
    position_matrix = np.empty((N,3))
    
    for i in range(len(ParticleList)):
        
        position_matrix[i,:] = ParticleList[i].position 
        
    return position_matrix
    
      
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

def sph2cart(sph_matrix):
    
    '''
    
    angles in radians.  
    
    Elevation ranges from 0 to pi (0 to 180)
    
        
    
    The first if block is for a vector and the second else block is for a matrix.
    If a vector is given when it expects a matrix, there are too many indices for the array
    '''
    
    if len(np.shape(sph_matrix)) == 1:   
        
        r = sph_matrix[0]
        azimuth = sph_matrix[1]
        elevation = sph_matrix[2]
        
        
        xyz_matrix = np.empty_like(sph_matrix)
        xyz_matrix[:] = np.NAN        
        
        ### Equations from http://mathworld.wolfram.com/SphericalCoordinates.html
        x = r*np.cos(azimuth)*np.sin(elevation)     
        y = r*np.sin(azimuth)*np.sin(elevation)    
        z = r*np.cos(elevation)
        
        xyz_matrix[0] = x
        xyz_matrix[1] = y
        xyz_matrix[2] = z 
                
    else:        
    
        r = sph_matrix[:,0]
        azimuth = sph_matrix[:,1]
        elevation = sph_matrix[:,2]
        
        
        xyz_matrix = np.zeros((np.shape(sph_matrix)[0],np.shape(sph_matrix)[1]))
     
        
        ### Equations from http://mathworld.wolfram.com/SphericalCoordinates.html
        x = r*np.cos(azimuth)*np.sin(elevation)     
        y = r*np.sin(azimuth)*np.sin(elevation)    
        z = r*np.cos(elevation)
        
        xyz_matrix[:,0] = x
        xyz_matrix[:,1] = y
        xyz_matrix[:,2] = z 
    
    return xyz_matrix
            
def animate_side_view_no_colour_coding(particle_list,view_direction=1):
       
    
    '''
    
    Kept this old version because the colour coding with a scatter plot
    won't allow the particles to be overlaid on the star in the background 
    
    Couldn't manage to get an actual animation working but instead it saves a 
    png from every time step                  
                    
    '''
    
    
    cmap = mpl.cm.spectral
    
    for timestep in range(ns*no):
    #for timestep in range(2):    
        
        
        ylist = []   
        zlist = []        
                
        print 'doing timestep %d' % (timestep)
        
        plt.figure()        
        
        for particle in particle_list:
            
            if timestep in particle.timestep_history:
                
                timestep_index = np.where(np.array(particle.timestep_history)==timestep)[0][0]                
                                
                if view_direction==1:
                    y = particle.position_history[timestep_index][1]
                    z = particle.position_history[timestep_index][2]
                if view_direction==2:
                    y = particle.position_history[timestep_index][0]
                    z = particle.position_history[timestep_index][2]
                
                ylist.append(y)
                zlist.append(z)
                
                
                print 'y = %f, z = %f' % (y,z)                
                
        
        plt.plot(ylist,zlist,'.')
        
        circle1=plt.Circle((0,0),radius=R_star,color='y')  # 0.7*R_sun as a rough estimate from wikipedia 
        plt.gcf().gca().add_artist(circle1)
        
        if view_direction==1:
            plt.ylabel('co-rotating z coordinate (m)')
            plt.xlabel('co-rotating y coordinate (m)')
            plt.title('top down view (equal axes)')     
            
            plt.xlim((-6E9,6E9))
            plt.ylim((-8E7,8E7))
            
        if view_direction==2:
            plt.ylabel('co-rotating z coordinate (m)')
            plt.xlabel('co-rotating x coordinate (m)')
            plt.title('top down view (equal axes)')     
            
            
    #        plt.gca().set_aspect('equal', adjustable='box')  
    #        
            plt.xlim((-6E9,6E9))
            plt.ylim((-8E7,8E7))
        
        
                
        plt.savefig(os.path.join('timestep_figs','sideview','timestep%d'%(timestep)))
        
        plt.close()              
    

def random_point_on_sphere():
    
    '''
    Return a uniformly distrbuted random point on the surface of a a sphere 
    accoding to Muller 1959 and Marsaglia 1972.
    
    The method is to generate three Guassian random variables x, y, z then 
    the uniform random distribution on the surface of the sphere is given by 
    
    1/sqrt(x**2 + y**2 + z**2)*[x,y,z]
        
    '''

    xyz_vect = np.random.normal(size=3)
    
    return 1.0/(np.sqrt(np.sum(xyz_vect**2.0)))*xyz_vect
    
    
def ReproducableRandomPointonSphere(n,particle_index):
    
    '''
    load a fixed random number distribution so 
    every run will be identical
    
    n = the total number of particles 
    particle_count = the current count of particles to load a particle from 
    '''
    
    path = os.path.join('RandomOnSphereList','RandomSpherexyzN%d.npy'%(n))
    
    if not os.path.exists(path):
        a = np.random.normal(size=(int(n),3))
        np.save(path,a)
        
    
    xyz_vect = np.load(path)[particle_index]
    
    return (1.0/(np.sqrt(np.sum(xyz_vect**2.0))))*xyz_vect
    
def ReproducableRandomPointonDaySide(n,particle_index):
    
    '''
    load a fixed random number distribution so 
    every run will be identical
    
    n = the total number of particles 
    particle_count = the current count of particles to load a particle from 
    
    Only particles on the day side (launched towards the star so with a y component that is greater than zero (negative would be away from star))
    '''
    
    path = os.path.join('RandomOnSphereList','DaySide','RandomDaySidexyzN%d.npy'%(n))
    
    if not os.path.exists(path):
        a = np.zeros((int(n),3))
        i = 0
        while i < n: 
            trialRandomVect = np.random.normal(size=(1,3))[0]
            normalisedVector = (1.0/(np.sqrt(np.sum(trialRandomVect**2.0))))*trialRandomVect

            if normalisedVector[1] > 0:  ## I guess this could be set to >= to include the terminator
            
                a[i,:] = normalisedVector
                i = i + 1 
           
        np.save(path,a)
        
    
    NormalisedVect = np.load(path)[particle_index]
    
    return NormalisedVect    
    
def GetAllRadii(particle_list):
    
    radii = np.empty((len(particle_list)))
    radii[:] = np.NAN
    
    for i in range(len(particle_list)):
        
        radii[i] = particle_list[i].radius 
        
    return radii  
    
    

def GetAllX(particle_list,x):
    
    '''
    A general to make a vector of a given parameter (x) for all 
    particles in the particle list.
    Only works for single values (not vectors like position).
    
    x is a string for the required attribute.  eg. x='position'
    '''
    
    xvect = np.empty((len(particle_list)))
    xvect[:] = np.NAN
    
    code_to_execute = 'xvect[i] = particle_list[i].%s'%(x)
    
    for i in range(len(particle_list)):        
        exec code_to_execute
        
    return xvect        
    
    
def opdep_static_grid(nR,nTheta,nPhi,RadialBins,ThetaBins,PhiBins,odm,particle_list):
    
    '''
    estimate the optical depth to different particles using the same grid as 
    is used to generate the density file for MCMax3D.  The calculation 
    is based on Christoph's opdep4
    '''
    
    p = GetAllxyzCoords(particle_list) ## positions in m in Cartesiain coordinates 
    d = GetAllRadii(particle_list)*2.0 ## particle diameters in m
    
    n = len(d)  ## number of particles 
    
    sph = cart2sph(p) #positions in specical coordinate 
    
    kappa = np.zeros((nR,nPhi,nTheta)) # absorption in spherical coordinates
 
    kint = np.zeros_like(kappa) # optical thickness in azimuth and elevation
    
    od = np.empty((n))
    od[:] = np.NAN
    
    xr = np.digitize(sph[:,0],RadialBins)
    
    zeroindices = np.where((xr==0))
    toohighindices = np.where(xr==len(RadialBins))        
    
    xr[zeroindices] = 1                   #put values that are below the lower bound into the first bin
    xr[toohighindices] = len(RadialBins)-1   #put values that are above the upper bound into the last bin     
    
    xa = np.digitize(sph[:,1]*180.0/np.pi+180.0,PhiBins) #the position around the star (azimuth, 0 to 360 degrees)
       
    xe = np.digitize(sph[:,2]*180.0/np.pi,ThetaBins) ## The equator in the spherical coords is 0 degrees but it is theta = 90 in MCMAX ThetaBins
   
    ### Calculate the optical absorption in spherical coordinates     
     
    for i in range(n):
        
        #print 'making kappa for particle ', i
        
        ### A spherical grid (3D array) with every element corresponding to the total absorption cross section in that grid position (more particles add)
        kappa[xr[i]-1,xa[i]-1,xe[i]-1]=kappa[xr[i]-1,xa[i]-1,xe[i]-1]+np.pi*(d[i]/2.0)**2  
        
        
    
    kint = np.cumsum(kappa,0) # integrate over radius
    
    for i in range(n):
        
        
        od[i]=kint[xr[i]-1,xa[i]-1,xe[i]-1]
        
        ### Store this value of od_1,od and beta in the particle object         
        particle_list[i].update_od(od[i])
    
    return particle_list,od
    #return None 
    
def opdep_density(nR,nTheta,nPhi,RadialBins,ThetaBins,PhiBins,odm,particle_list):
    
    '''
    estimate the optical depth to different particles using the same grid as 
    is used to generate the density file for MCMax3D.  The calculation 
    is based on Christoph's opdep4
   
   
   For 600nm (0.6um)
     .6000E+00   .1560E+01   .1400E-01 
     
     from the .lnk file 
    
    '''
    
    p = GetAllxyzCoords(particle_list) ## positions in m in Cartesiain coordinates 
    d = GetAllRadii(particle_list)*2.0 ## particle diameters in m
    
    n = len(d)  ## number of particles 
    
    sph = cart2sph(p) #positions in specical coordinate 
    
    od = np.empty((n))
    od[:] = np.NAN
    
    xr = np.digitize(sph[:,0],RadialBins)
    
    zeroindices = np.where((xr==0))
    toohighindices = np.where(xr==len(RadialBins))        
    
    xr[zeroindices] = 1                   #put values that are below the lower bound into the first bin
    xr[toohighindices] = len(RadialBins)-1   #put values that are above the upper bound into the last bin     
    
    xa = np.digitize(sph[:,1]*180.0/np.pi+180.0,PhiBins) #the position around the star (azimuth, 0 to 360 degrees)
       
    xe = np.digitize(sph[:,2]*180.0/np.pi,ThetaBins) ## The equator in the spherical coords is 0 degrees but it is theta = 90 in MCMAX ThetaBins
   
    mass_density = np.zeros((nPhi,nTheta,nR))
    optical_depth_per_cell = np.zeros((nPhi,nTheta,nR))
    
    TotalMass = 0.0   
   
   
    ### Calculate the optical absorption in spherical coordinates     
     
    for i in range(n):


        particle_radius = particle_list[i].radius         
               
        particle_mass_kg = ((4.0/3.0)*np.pi*particle_radius**3)*particle_density ## Assuming spherical particles 
        
        #particle_mass_g = particle_mass_kg*1000.0
        
        TotalMass += particle_mass_kg*DensityScalingFactor        
        
        r1 = RadialBins[xr[i]-1]
        r2 = RadialBins[xr[i]]
        
        cell_thickness = r2-r1 
        
        t1 = PhiBins[xa[i]-1]*np.pi/180.0 ## NOTE!! theta (t) for this function is phi in rest of code
        t2 = PhiBins[xa[i]]*np.pi/180.0 ## NOTE!! theta (t) for this function is phi in rest of code
        
        p1 = ThetaBins[xe[i]-1]*np.pi/180.0 ## NOTE!! phi (p) for this function is theta in the rest of the code 
        p2 = ThetaBins[xe[i]]*np.pi/180.0 ## NOTE!! phi (p) for this function is theta in the rest of the code 
        
        volume = tplquad(diff_volume, r1, r2, lambda r:   t1, lambda r:   t2, lambda r,t: p1, lambda r,t: p2)[0] #in cubic metres 
        
        #volume_cm = 1000000.0*volume        
                                              
    #        volume_matrix[xr[i]-1,xa[i]-1,xe[i]-1] = volume                                
    #
    #        number_density[xr[i]-1,xa[i]-1,xe[i]-1]=number_density[xr[i]-1,xa[i]-1,xe[i]-1] + 1.0/volume ## counting the number density 
        
        mass_density[xa[i]-1,xe[i]-1,xr[i]-1] = mass_density[xa[i]-1,xe[i]-1,xr[i]-1] + DensityScalingFactor*particle_mass_kg/volume #volume in m**3
        optical_depth_per_cell[xa[i]-1,xe[i]-1,xr[i]-1] = mass_density[xa[i]-1,xe[i]-1,xr[i]-1]*cell_thickness*Kappa_ref_index
        
               
    #### this should be summing over dimension=2, can be seen  by doing sum and checking the resulting dimensions 
    ##kint = np.cumsum(optical_depth_per_cell,0) # integrate over radius
               
    kint = np.cumsum(optical_depth_per_cell,2) # integrate over radius.  Note the axes are different to opdep_static_grid
    
    for i in range(n):
        
        
        #od[i]=kint[xr[i]-1,xa[i]-1,xe[i]-1]
        od[i]=kint[xa[i]-1,xe[i]-1,xr[i]-1]
        
        ### Store this value of od_1,od and beta in the particle object         
        particle_list[i].update_od(od[i])
    
    return particle_list,od
    #return None 
    
def opdep_density2(nR,nTheta,nPhi,RadialBins,ThetaBins,PhiBins,odm,particle_list):
    
    '''
    estimate the optical depth to different particles using the same grid as 
    is used to generate the density file for MCMax3D.  The calculation 
    is based on Christoph's opdep4
   
   
   For 600nm (0.6um)
     .6000E+00   .1560E+01   .1400E-01 
     
     from the .lnk file 
    
    '''
    
    p = GetAllxyzCoords(particle_list) ## positions in m in Cartesiain coordinates 
    d = GetAllRadii(particle_list)*2.0 ## particle diameters in m
    
    n = len(d)  ## number of particles 
    
    sph = cart2sph(p) #positions in specical coordinate 
    
    od = np.empty((n))
    od[:] = np.NAN
    
    xr = np.digitize(sph[:,0],RadialBins)
    
    zeroindices = np.where((xr==0))
    toohighindices = np.where(xr==len(RadialBins))        
    
    xr[zeroindices] = 1                   #put values that are below the lower bound into the first bin
    xr[toohighindices] = len(RadialBins)-1   #put values that are above the upper bound into the last bin     
    
    xa = np.digitize(sph[:,1]*180.0/np.pi+180.0,PhiBins) #the position around the star (azimuth, 0 to 360 degrees)
       
    xe = np.digitize(sph[:,2]*180.0/np.pi,ThetaBins) ## The equator in the spherical coords is 0 degrees but it is theta = 90 in MCMAX ThetaBins
   
    mass = np.zeros((nPhi,nTheta,nR))
    volumes = np.zeros((nPhi,nTheta,nR))
    radial_extent = np.zeros((nPhi,nTheta,nR))
    optical_depth_per_cell = np.zeros((nPhi,nTheta,nR))
    
    TotalMass = 0.0   
   
   
    ### Calculate the optical absorption in spherical coordinates     
     
    for i in range(n):


        particle_radius = particle_list[i].radius         
               
        particle_mass_kg = ((4.0/3.0)*np.pi*particle_radius**3)*particle_density ## Assuming spherical particles 
        
        #particle_mass_g = particle_mass_kg*1000.0
        
        TotalMass += particle_mass_kg*DensityScalingFactor        
        

        
        #volume = tplquad(diff_volume, r1, r2, lambda r:   t1, lambda r:   t2, lambda r,t: p1, lambda r,t: p2)[0] #in cubic metres 
        
        #volume_cm = 1000000.0*volume        
                                              
    #        volume_matrix[xr[i]-1,xa[i]-1,xe[i]-1] = volume                                
    #
    #        number_density[xr[i]-1,xa[i]-1,xe[i]-1]=number_density[xr[i]-1,xa[i]-1,xe[i]-1] + 1.0/volume ## counting the number density 
        
        mass[xa[i]-1,xe[i]-1,xr[i]-1] = mass[xa[i]-1,xe[i]-1,xr[i]-1] + DensityScalingFactor*particle_mass_kg#/volume #volume in m**3
        
        
    for i in range(nPhi):
        for j in range(nTheta):
            for k in range(nR):
                
                if mass[i,j,k] != 0:
                    
                    r1 = RadialBins[k]
                    r2 = RadialBins[k+1]
                    
                    radial_extent[i,j,k] = r2-r1 
                    
                    t1 = PhiBins[i]*np.pi/180.0 ## NOTE!! theta (t) for this function is phi in rest of code
                    t2 = PhiBins[i+1]*np.pi/180.0 ## NOTE!! theta (t) for this function is phi in rest of code
                    
                    p1 = ThetaBins[j]*np.pi/180.0 ## NOTE!! phi (p) for this function is theta in the rest of the code 
                    p2 = ThetaBins[j+1]*np.pi/180.0 ## NOTE!! phi (p) for this function is theta in the rest of the code 
                    
                    volumes[i,j,k] = tplquad(diff_volume, r1, r2, lambda r:   t1, lambda r:   t2, lambda r,t: p1, lambda r,t: p2)[0] #in cubic metres 
                    optical_depth_per_cell[i,j,k] = (mass[i,j,k]/volumes[i,j,k])*radial_extent[i,j,k]*Kappa_ref_index  
                       
    
               
    #### this should be summing over dimension=2, can be seen  by doing sum and checking the resulting dimensions 
    ##kint = np.cumsum(optical_depth_per_cell,0) # integrate over radius
               
    kint = np.cumsum(optical_depth_per_cell,2) # integrate over radius.  Note the axes are different to opdep_static_grid
    
    for i in range(n):
        
        
        #od[i]=kint[xr[i]-1,xa[i]-1,xe[i]-1]
        od[i]=kint[xa[i]-1,xe[i]-1,xr[i]-1]
        
        ### Store this value of od_1,od and beta in the particle object         
        particle_list[i].update_od(od[i])
    
    return particle_list,od
    #return None 
    
def opdep_density3(nR,nTheta,nPhi,RadialBins,ThetaBins,PhiBins,particle_list):
    
    '''
    estimate the optical depth to different particles using the same grid as 
    is used to generate the density file for MCMax3D.  The calculation 
    is based on Christoph's opdep4
   
   
   For 600nm (0.6um)
     .6000E+00   .1560E+01   .1400E-01 
     
     from the .lnk file 
    
    '''
    
    p = GetAllxyzCoords(particle_list) ## positions in m in Cartesiain coordinates 
    d = GetAllRadii(particle_list)*2.0 ## particle diameters in m
    
    n = len(d)  ## number of particles 
    
    sph = cart2sph(p) #positions in specical coordinate 
    
    od = np.empty((n))
    od[:] = np.NAN
    
    xr = np.digitize(sph[:,0],RadialBins)
    
    zeroindices = np.where((xr==0))
    toohighindices = np.where(xr==len(RadialBins))        
    
    xr[zeroindices] = 1                   #put values that are below the lower bound into the first bin
    xr[toohighindices] = len(RadialBins)-1   #put values that are above the upper bound into the last bin     
    
    xa = np.digitize(sph[:,1]*180.0/np.pi+180.0,PhiBins) #the position around the star (azimuth, 0 to 360 degrees)
       
    xe = np.digitize(sph[:,2]*180.0/np.pi,ThetaBins) ## The equator in the spherical coords is 0 degrees but it is theta = 90 in MCMAX ThetaBins
   
    mass = np.zeros((nPhi,nTheta,nR))
    volumes = np.zeros((nPhi,nTheta,nR))
    radial_extent = np.zeros((nPhi,nTheta,nR))
    optical_depth_per_cell = np.zeros((nPhi,nTheta,nR))
    
    TotalMass = 0.0   
    
    ### Establish which grid cells have particles in them and therefore need to have a volume calculated 

    
    non_set_grid_cells = []

    for i in range(len(xr)):
        
        non_set_grid_cells.append((xa[i],xe[i],xr[i]))
        
    SetOfGridCellsWithMass = set(non_set_grid_cells)
   
    ### Calculate the optical absorption in spherical coordinates     
     
    for i in range(n):


        particle_radius = particle_list[i].radius         
               
        particle_mass_kg = ((4.0/3.0)*np.pi*particle_radius**3)*particle_density ## Assuming spherical particles 
        
        #particle_mass_g = particle_mass_kg*1000.0
        
        TotalMass += particle_mass_kg*DensityScalingFactor        
        

        
        #volume = tplquad(diff_volume, r1, r2, lambda r:   t1, lambda r:   t2, lambda r,t: p1, lambda r,t: p2)[0] #in cubic metres 
        
        #volume_cm = 1000000.0*volume        
                                              
    #        volume_matrix[xr[i]-1,xa[i]-1,xe[i]-1] = volume                                
    #
    #        number_density[xr[i]-1,xa[i]-1,xe[i]-1]=number_density[xr[i]-1,xa[i]-1,xe[i]-1] + 1.0/volume ## counting the number density 
        
        mass[xa[i]-1,xe[i]-1,xr[i]-1] = mass[xa[i]-1,xe[i]-1,xr[i]-1] + DensityScalingFactor*particle_mass_kg#/volume #volume in m**3
        
    
        
   ### Iterate over the cells that have mass (from SetOfGridCellsWithMass) calculating volume and optical depth     
        
        
    for xaxexr in SetOfGridCellsWithMass:
        
        tempxa = xaxexr[0]
        tempxe = xaxexr[1]
        tempxr = xaxexr[2]
                    
        r1 = RadialBins[tempxr-1]
        r2 = RadialBins[tempxr]
        
        radial_extent[tempxa-1,tempxe-1,tempxr-1] = r2-r1 
        
        t1 = PhiBins[tempxa-1]*np.pi/180.0 ## NOTE!! theta (t) for this function is phi in rest of code
        t2 = PhiBins[tempxa]*np.pi/180.0 ## NOTE!! theta (t) for this function is phi in rest of code
        
        p1 = ThetaBins[tempxe-1]*np.pi/180.0 ## NOTE!! phi (p) for this function is theta in the rest of the code 
        p2 = ThetaBins[tempxe]*np.pi/180.0 ## NOTE!! phi (p) for this function is theta in the rest of the code 
        
        volumes[tempxa-1,tempxe-1,tempxr-1] = tplquad(diff_volume, r1, r2, lambda r:   t1, lambda r:   t2, lambda r,t: p1, lambda r,t: p2)[0] #in cubic metres 
        optical_depth_per_cell[tempxa-1,tempxe-1,tempxr-1] = (mass[tempxa-1,tempxe-1,tempxr-1]/volumes[tempxa-1,tempxe-1,tempxr-1])*radial_extent[tempxa-1,tempxe-1,tempxr-1]*Kappa_ref_index  
                       
    
               
    #### this should be summing over dimension=2, can be seen  by doing sum and checking the resulting dimensions 
             
    kint = np.cumsum(optical_depth_per_cell,2) # integrate over radius.  Note the axes are different to opdep_static_grid
    
    for i in range(n):        
        
        #od[i]=kint[xr[i]-1,xa[i]-1,xe[i]-1]
        od[i]=kint[xa[i]-1,xe[i]-1,xr[i]-1]
        
        ### Store this value of od_1,od and beta in the particle object         
        particle_list[i].update_od(od[i])
    
    return particle_list,od
    
def opdep_density4(nR,nTheta,nPhi,RadialBins,ThetaBins,PhiBins,particle_list):
    
    '''
    estimate the optical depth to different particles using the same grid as 
    is used to generate the density file for MCMax3D.  The calculation 
    is based on Christoph's opdep4.
    
    This version loads the grid cell volume because it took too long to compute the grid cell volume every time
   
   
   For 600nm (0.6um)
     .6000E+00   .1560E+01   .1400E-01 
     
     from the .lnk file 
    
    '''
    
    p = GetAllxyzCoords(particle_list) ## positions in m in Cartesiain coordinates 
    d = GetAllRadii(particle_list)*2.0 ## particle diameters in m
    
    n = len(d)  ## number of particles 
    
    sph = cart2sph(p) #positions in specical coordinate 
    
    od = np.empty((n))
    od[:] = np.NAN
    
    xr = np.digitize(sph[:,0],RadialBins)
    
    zeroindices = np.where((xr==0))
    toohighindices = np.where(xr==len(RadialBins))        
    
    xr[zeroindices] = 1                   #put values that are below the lower bound into the first bin
    xr[toohighindices] = len(RadialBins)-1   #put values that are above the upper bound into the last bin     
    
    xa = np.digitize(sph[:,1]*180.0/np.pi+180.0,PhiBins) #the position around the star (azimuth, 0 to 360 degrees)
       
    xe = np.digitize(sph[:,2]*180.0/np.pi,ThetaBins) ## The equator in the spherical coords is 0 degrees but it is theta = 90 in MCMAX ThetaBins
   
    mass = np.zeros((nPhi,nTheta,nR))
    optical_depth_per_cell = np.zeros((nPhi,nTheta,nR))
    
    TotalMass = 0.0   
    
    ### Establish which grid cells have particles in them and therefore need to have a volume calculated 

    
    non_set_grid_cells = []

    for i in range(len(xr)):
        
        non_set_grid_cells.append((xa[i],xe[i],xr[i]))
        
    SetOfGridCellsWithMass = set(non_set_grid_cells)
   
    ### Calculate the optical absorption in spherical coordinates     
     
    for i in range(n):


        particle_radius = particle_list[i].radius         
               
        particle_mass_kg = ((4.0/3.0)*np.pi*particle_radius**3)*particle_density ## Assuming spherical particles 
        
        #particle_mass_g = particle_mass_kg*1000.0
        
        TotalMass += particle_mass_kg*DensityScalingFactor               
        
        mass[xa[i]-1,xe[i]-1,xr[i]-1] = mass[xa[i]-1,xe[i]-1,xr[i]-1] + DensityScalingFactor*particle_mass_kg#/volume #volume in m**3    
        
   ### Iterate over the cells that have mass (from SetOfGridCellsWithMass) calculating volume and optical depth         
        
    for xaxexr in SetOfGridCellsWithMass:
        
        tempxa = xaxexr[0]
        tempxe = xaxexr[1]
        tempxr = xaxexr[2]
                    
        optical_depth_per_cell[tempxa-1,tempxe-1,tempxr-1] = (mass[tempxa-1,tempxe-1,tempxr-1]/VolumeGrid[tempxa-1,tempxe-1,tempxr-1])*RadialExtentGrid[tempxa-1,tempxe-1,tempxr-1]*Kappa_ref_index  
                                      
    #### this should be summing over dimension=2, can be seen  by doing sum and checking the resulting dimensions 
             
    kint = np.cumsum(optical_depth_per_cell,2) # integrate over radius.  Note the axes are different to opdep_static_grid
    
    for i in range(n):        
        
 
        od[i]=kint[xa[i]-1,xe[i]-1,xr[i]-1]
        
        ### Store this value of od_1,od and beta in the particle object         
        particle_list[i].update_od(od[i])
    
    return particle_list,od
    
    
def OpDepInteriorFunctionOne(ParticleNumber):
    
    particle_radius = particle_list[i].radius        
    
    xsize = np.digitize(particle_radius,sizebins)
    if xsize == len(sizebins):
        xsize = len(sizebins)-1
           
    particle_mass_kg = ((4.0/3.0)*np.pi*particle_radius**3)*particle_density ## Assuming spherical particles 
    
    #particle_mass_g = particle_mass_kg*1000.0
    
    TotalMass += particle_mass_kg*DensityScalingFactor     

#        print 'timestep: %d, n: %d' % (timestep,n)        
#        
#        if ((timestep == 1)&(n==4)):
#        
#            mass[xsize-1,xa[i]-1,xe[i]-1,xr[i]-1] = mass[xsize-1,xa[i]-1,xe[i]-1,xr[i]-1] + DensityScalingFactor*particle_mass_kg#(volume done below in optical depth per cell) /volume #volume in m**3    
        
    mass[xsize-1,xa[i]-1,xe[i]-1,xr[i]-1] = mass[xsize-1,xa[i]-1,xe[i]-1,xr[i]-1] + DensityScalingFactor*particle_mass_kg#(volume done below in optical depth per cell) /volume #volume in m**3            

    


def opdepPlankMeanOpacityAttemptMultiprocess(nsize,nR,nTheta,nPhi,sizebins,RadialBins,ThetaBins,PhiBins,particle_list):
    
    '''
    estimate the optical depth to different particles using the same grid as 
    is used to generate the density file for MCMax3D.  The calculation 
    is based on Christoph's opdep4.
    
    This version loads the grid cell volume because it took too long to compute the grid cell volume every time
   
   
    
    '''
    
    p = GetAllxyzCoords(particle_list) ## positions in m in Cartesiain coordinates 
    d = GetAllRadii(particle_list)*2.0 ## particle diameters in m
    
    n = len(d)  ## number of particles 
    
    sph = cart2sph(p) #positions in specical coordinate 
    
#    od = np.empty((n))
#    od[:] = np.NAN
    
    od = np.zeros((n))
    
    xr = np.digitize(sph[:,0],RadialBins)
    
    zeroindices = np.where((xr==0))
    toohighindices = np.where(xr==len(RadialBins))        
    
    xr[zeroindices] = 1                   #put values that are below the lower bound into the first bin
    xr[toohighindices] = len(RadialBins)-1   #put values that are above the upper bound into the last bin     
    
    xa = np.digitize(sph[:,1]*180.0/np.pi+180.0,PhiBins) #the position around the star (azimuth, 0 to 360 degrees)
       
    xe = np.digitize(sph[:,2]*180.0/np.pi,ThetaBins) ## The equator in the spherical coords is 0 degrees but it is theta = 90 in MCMAX ThetaBins
   
    mass = np.zeros((nsize,nPhi,nTheta,nR))
    
    
    #### Attempted bug fix.  There are 15 size bins (not len(sizebins) = 16) so this should just be nsize like the mass      
    
    #optical_depth_per_cell = np.zeros((len(sizebins),nPhi,nTheta,nR))   ## Old 
    optical_depth_per_cell = np.zeros((nsize,nPhi,nTheta,nR))
    
    
    
    TotalMass = 0.0   
    
    #comp = np.zeros((nTemp,nsize,npart,nPhi,nTheta,nR))
    
    ### Establish which grid cells have particles in them and therefore need to have a volume calculated 

    
    non_set_grid_cells = []

    for i in range(len(xr)):
        
        non_set_grid_cells.append((xa[i],xe[i],xr[i]))
        
    SetOfGridCellsWithMass = set(non_set_grid_cells)
   
    ### Calculate the optical absorption in spherical coordinates     
     
    for i in range(n):
        
        pool = multiprocessing.Pool()


        
   ### Iterate over the cells that have mass (from SetOfGridCellsWithMass) calculating volume and optical depth         
        
    for xaxexr in SetOfGridCellsWithMass:
        
        tempxa = xaxexr[0]
        tempxe = xaxexr[1]
        tempxr = xaxexr[2]
        
        ### Attempting fix with optical_depth_per_cell having same dimensions as mass 
        #for sizeindex in range(len(sizebins)):  #old     
        for sizeindex in range(nsize):     
            
            ####optical_depth_per_cell[sizeindex,tempxa-1,tempxe-1,tempxr-1] = (mass[sizeindex-1,tempxa-1,tempxe-1,tempxr-1]/VolumeGrid[tempxa-1,tempxe-1,tempxr-1])*RadialExtentGrid[tempxa-1,tempxe-1,tempxr-1]#*Kappa_ref_index    ## Old
            
            optical_depth_per_cell[sizeindex,tempxa-1,tempxe-1,tempxr-1] = (mass[sizeindex,tempxa-1,tempxe-1,tempxr-1]/VolumeGrid[tempxa-1,tempxe-1,tempxr-1])*RadialExtentGrid[tempxa-1,tempxe-1,tempxr-1]#*Kappa_ref_index  


                 
    kint = np.cumsum(optical_depth_per_cell,3) # integrate over radius.  Note the axes are different to opdep_static_grid

    
    for i in range(n):       
        ## Adding up the contributions to the cumulative optical depth for all particle sizes 
        for sizeindex in range(nsize):   
 
            od[i] =  od[i] + kint[sizeindex,xa[i]-1,xe[i]-1,xr[i]-1]*ExtinctionInEachSizeBin[sizeindex]
        
        ### Store this value of od_1,od and beta in the particle object         
        particle_list[i].update_od(od[i])
    
    return particle_list,od
    
        
def opdepPlankMeanOpacity(nsize,nR,nTheta,nPhi,sizebins,RadialBins,ThetaBins,PhiBins,particle_list):
    
    '''
    estimate the optical depth to different particles using the same grid as 
    is used to generate the density file for MCMax3D.  The calculation 
    is based on Christoph's opdep4.
    
    This version loads the grid cell volume because it took too long to compute the grid cell volume every time
   
   
    
    '''
    
    p = GetAllxyzCoords(particle_list) ## positions in m in Cartesiain coordinates 
    d = GetAllRadii(particle_list)*2.0 ## particle diameters in m
    
    n = len(d)  ## number of particles 
    
    sph = cart2sph(p) #positions in specical coordinate 
    
#    od = np.empty((n))
#    od[:] = np.NAN
    
    od = np.zeros((n))
    
    xr = np.digitize(sph[:,0],RadialBins)
    
    zeroindices = np.where((xr==0))
    toohighindices = np.where(xr==len(RadialBins))        
    
    xr[zeroindices] = 1                   #put values that are below the lower bound into the first bin
    xr[toohighindices] = len(RadialBins)-1   #put values that are above the upper bound into the last bin     
    
    xa = np.digitize(sph[:,1]*180.0/np.pi+180.0,PhiBins) #the position around the star (azimuth, 0 to 360 degrees)
       
    xe = np.digitize(sph[:,2]*180.0/np.pi,ThetaBins) ## The equator in the spherical coords is 0 degrees but it is theta = 90 in MCMAX ThetaBins
   
    mass = np.zeros((nsize,nPhi,nTheta,nR))
    
    
    #### Attempted bug fix.  There are 15 size bins (not len(sizebins) = 16) so this should just be nsize like the mass      
    
    #optical_depth_per_cell = np.zeros((len(sizebins),nPhi,nTheta,nR))   ## Old 
    optical_depth_per_cell = np.zeros((nsize,nPhi,nTheta,nR))
    
    
    
    TotalMass = 0.0   
    
    #comp = np.zeros((nTemp,nsize,npart,nPhi,nTheta,nR))
    
    ### Establish which grid cells have particles in them and therefore need to have a volume calculated 

    
    non_set_grid_cells = []

    for i in range(len(xr)):
        
        non_set_grid_cells.append((xa[i],xe[i],xr[i]))
        
    SetOfGridCellsWithMass = set(non_set_grid_cells)
   
    ### Calculate the optical absorption in spherical coordinates     
     
    for i in range(n):


        particle_radius = particle_list[i].radius        
        
        xsize = np.digitize(particle_radius,sizebins)
        if xsize == len(sizebins):
            xsize = len(sizebins)-1
               
        particle_mass_kg = ((4.0/3.0)*np.pi*particle_radius**3)*particle_density ## Assuming spherical particles 
        
        #particle_mass_g = particle_mass_kg*1000.0
        
        TotalMass += particle_mass_kg*DensityScalingFactor     

#        print 'timestep: %d, n: %d' % (timestep,n)        
#        
#        if ((timestep == 1)&(n==4)):
#        
#            mass[xsize-1,xa[i]-1,xe[i]-1,xr[i]-1] = mass[xsize-1,xa[i]-1,xe[i]-1,xr[i]-1] + DensityScalingFactor*particle_mass_kg#(volume done below in optical depth per cell) /volume #volume in m**3    
            
        mass[xsize-1,xa[i]-1,xe[i]-1,xr[i]-1] = mass[xsize-1,xa[i]-1,xe[i]-1,xr[i]-1] + DensityScalingFactor*particle_mass_kg#(volume done below in optical depth per cell) /volume #volume in m**3            
        
   ### Iterate over the cells that have mass (from SetOfGridCellsWithMass) calculating volume and optical depth         
        
    for xaxexr in SetOfGridCellsWithMass:
        
        tempxa = xaxexr[0]
        tempxe = xaxexr[1]
        tempxr = xaxexr[2]
        
        ### Attempting fix with optical_depth_per_cell having same dimensions as mass 
        #for sizeindex in range(len(sizebins)):  #old     
        for sizeindex in range(nsize):     
            
            ####optical_depth_per_cell[sizeindex,tempxa-1,tempxe-1,tempxr-1] = (mass[sizeindex-1,tempxa-1,tempxe-1,tempxr-1]/VolumeGrid[tempxa-1,tempxe-1,tempxr-1])*RadialExtentGrid[tempxa-1,tempxe-1,tempxr-1]#*Kappa_ref_index    ## Old
            
            optical_depth_per_cell[sizeindex,tempxa-1,tempxe-1,tempxr-1] = (mass[sizeindex,tempxa-1,tempxe-1,tempxr-1]/VolumeGrid[tempxa-1,tempxe-1,tempxr-1])*RadialExtentGrid[tempxa-1,tempxe-1,tempxr-1]#*Kappa_ref_index  


                 
    kint = np.cumsum(optical_depth_per_cell,3) # integrate over radius.  Note the axes are different to opdep_static_grid

    
    for i in range(n):       
        ## Adding up the contributions to the cumulative optical depth for all particle sizes 
        for sizeindex in range(nsize):   
 
            od[i] =  od[i] + kint[sizeindex,xa[i]-1,xe[i]-1,xr[i]-1]*ExtinctionInEachSizeBin[sizeindex]
        
        ### Store this value of od_1,od and beta in the particle object         
        particle_list[i].update_od(od[i])
    
    return particle_list,od

def opdepPlanckMeanOpacityComputeVolumesOnFly(nsize,nR,nTheta,nPhi,sizebins,RadialBins,ThetaBins,PhiBins,particle_list):
    
    '''
    estimate the optical depth to different particles using the same grid as 
    is used to generate the density file for MCMax3D.  The calculation 
    is based on Christoph's opdep4.
    
    This version can compute the volumes of the grid cells if the volume of a 
    particular grid cell is not already available.  It first checks to see if a file 
    containing the volumes for this grid exists, and if it does it checks if 
    the volume of the required cell has already been computed and just uses that 
    volume if it's available.  Otherwise, it computes the volume and saves it 
    so that it can be loaded again if the same grid cell is used    
    
    '''
    
    p = GetAllxyzCoords(particle_list) ## positions in m in Cartesiain coordinates 
    d = GetAllRadii(particle_list)*2.0 ## particle diameters in m
    
    n = len(d)  ## number of particles 
    
    sph = cart2sph(p) #positions in specical coordinate     
  
    od = np.zeros((n))
    
    xr = np.digitize(sph[:,0],RadialBins)
    
    zeroindices = np.where((xr==0))
    toohighindices = np.where(xr==len(RadialBins))        
    
    xr[zeroindices] = 1                   #put values that are below the lower bound into the first bin
    xr[toohighindices] = len(RadialBins)-1   #put values that are above the upper bound into the last bin     
    
    xa = np.digitize(sph[:,1]*180.0/np.pi+180.0,PhiBins) #the position around the star (azimuth, 0 to 360 degrees)
       
    xe = np.digitize(sph[:,2]*180.0/np.pi,ThetaBins) ## The equator in the spherical coords is 0 degrees but it is theta = 90 in MCMAX ThetaBins
   
    mass = np.zeros((nsize,nPhi,nTheta,nR))
    
    optical_depth_per_cell = np.zeros((nsize,nPhi,nTheta,nR))
    
    
    
    TotalMass = 0.0   
    
   
    ### Establish which grid cells have particles in them and therefore need to have a volume calculated 

    
    non_set_grid_cells = []

    for i in range(len(xr)):
        
        non_set_grid_cells.append((xa[i],xe[i],xr[i]))
        
    SetOfGridCellsWithMass = set(non_set_grid_cells)
   
    ### Calculate the optical absorption in spherical coordinates     
     
    for i in range(n):


        particle_radius = particle_list[i].radius        
        
        xsize = np.digitize(particle_radius,sizebins)
        if xsize == len(sizebins):
            xsize = len(sizebins)-1
               
        particle_mass_kg = ((4.0/3.0)*np.pi*particle_radius**3)*particle_density ## Assuming spherical particles 
        
        
        TotalMass += particle_mass_kg*DensityScalingFactor     

            
        mass[xsize-1,xa[i]-1,xe[i]-1,xr[i]-1] = mass[xsize-1,xa[i]-1,xe[i]-1,xr[i]-1] + DensityScalingFactor*particle_mass_kg#(volume done below in optical depth per cell) /volume #volume in m**3            
        
   ### Iterate over the cells that have mass (from SetOfGridCellsWithMass) calculating volume and optical depth         
        
    for xaxexr in SetOfGridCellsWithMass:
        
        tempxa = xaxexr[0]
        tempxe = xaxexr[1]
        tempxr = xaxexr[2]
        
        if  radial_extent[tempxa-1,tempxe-1,tempxr-1] == 0:
        
            r1 = RadialBins[tempxr-1]
            r2 = RadialBins[tempxr]
    
                   
            
            radial_extent[tempxa-1,tempxe-1,tempxr-1] = r2-r1 
            
            t1 = PhiBins[tempxa-1]*np.pi/180.0 ## NOTE!! theta (t) for this function is phi in rest of code
            t2 = PhiBins[tempxa]*np.pi/180.0 ## NOTE!! theta (t) for this function is phi in rest of code
            
            p1 = ThetaBins[tempxe-1]*np.pi/180.0 ## NOTE!! phi (p) for this function is theta in the rest of the code 
            p2 = ThetaBins[tempxe]*np.pi/180.0 ## NOTE!! phi (p) for this function is theta in the rest of the code 
            
            
            volumes[tempxa-1,tempxe-1,tempxr-1] = tplquad(diff_volume, r1, r2, lambda r:   t1, lambda r:   t2, lambda r,t: p1, lambda r,t: p2)[0] #in cubic metres 
        

        for sizeindex in range(nsize):     
            
           
            optical_depth_per_cell[sizeindex,tempxa-1,tempxe-1,tempxr-1] = (mass[sizeindex,tempxa-1,tempxe-1,tempxr-1]/volumes[tempxa-1,tempxe-1,tempxr-1])*radial_extent[tempxa-1,tempxe-1,tempxr-1]#*Kappa_ref_index  


                 
    kint = np.cumsum(optical_depth_per_cell,3) # integrate over radius.  Note the axes are different to opdep_static_grid

    
    for i in range(n):       
        ## Adding up the contributions to the cumulative optical depth for all particle sizes 
        for sizeindex in range(nsize):   
 
            od[i] =  od[i] + kint[sizeindex,xa[i]-1,xe[i]-1,xr[i]-1]*ExtinctionInEachSizeBin[sizeindex]
        
        ### Store this value of od_1,od and beta in the particle object         
        particle_list[i].update_od(od[i]*odm)
    
    return particle_list,od
    
    
def PlotOpDepVsRadialDist(particle_list):
    
    '''
    plot the optical depth (specifically transmittance, or what is multiplied
    by beta) as a function of radial distance from the star 
    '''
    
    cartpos = GetAllxyzCoords(particle_list)
    sphericalPos = cart2sph(cartpos)
    
    RadialDistances = sphericalPos[:,0]
    
    opdeps = GetAllX(particle_list,'od_1')
    Transmittances = GetAllX(particle_list,'transmittance')
    
#    plt.figure(figsize=((12,12)))    
#    
#    plt.plot(RadialDistances,Transmittances,'.')
#    
#    plt.xlabel('radial distance from star (m)')
#    plt.ylabel('transmittance')
#    plt.title('transmittance to particle at the given distance')    
#    
#    plt.savefig(os.path.join(OutputDirectory,'transmittance.png'))
    
    transarr = np.zeros((len(RadialDistances),2))
    transarr[:,0] = RadialDistances
    transarr[:,1] = Transmittances
    
    np.save(os.path.join(OutputDirectory,'TransmittanceArray.npy'),transarr)
    
    
#    plt.figure(figsize=((12,12)))
#    
#    plt.plot(RadialDistances,opdeps,'.')
    
    opdeparr = np.zeros((len(RadialDistances),2))
    opdeparr[:,0] = RadialDistances
    opdeparr[:,1] = opdeps
    
    opdeparr_azimuth = np.zeros((len(RadialDistances),2))
    opdeparr_azimuth[:,0] = sphericalPos[:,1]
    opdeparr_azimuth[:,1] = opdeps
    
    np.save(os.path.join(OutputDirectory,'OpticalDepthArray.npy'),opdeparr)
    
#    plt.xlabel('radial distance from star (m)')
#    plt.ylabel('optical depth')
#    plt.title('optical depth to particle at a given distance')    
#    
#    plt.savefig(os.path.join(OutputDirectory,'optical_depth.png'))
#    
#    plt.figure(figsize=((12,12)))    
#    plt.plot(sphericalPos[:,1]*180./np.pi,opdeps,'.')
#    plt.ylabel('optical depth')
#    plt.xlabel('azimuth (degrees)')
#    
#    plt.savefig(os.path.join(OutputDirectory,'optical_depth_function_of_angle.png'))
    np.save(os.path.join(OutputDirectory,'OpticalDepthArrayAzimuth.npy'),opdeparr_azimuth)
    
    
    
    plt.close('all')
    
    
def light_curve_opdep(particle_list):
    
    '''
    start with 5 bands "horizontally" across the star  
    '''
    
    p = GetAllxyzCoords(particle_list) ## positions in m in Cartesiain coordinates 
    d = GetAllRadii(particle_list)*2.0 ## particle diameters in m
    
    n = len(d)  ## number of particles 
    
   

    
    
    
#def calculate_beta(particle_list):
#    
#    '''
#    Equataions taken from pg. 14 of Burns, Lamy and Soter 1979
#    '''
#    
#    for i in range(len(particle_list)):
#    
#        particle_radius = particle_list[i].radius
#        
#        particle_pos_cart = particle_list[i].position
#    
#        particle_pos_sph = cart2sph(particle_pos_cart)
#        
#               
#        Qpr = particle_list[i].Qpr # Radiation pressure coefficient 
#        
#        transmittance = particle_list[i].transmittance ## The transmittance from the optical depth 
#        
#        r = particle_pos_sph[0] #distance of particle from Sun  
#
#        ## Factor to account for resolved disk of host star.
#        ## From Rik's thesis pg 122, I think this can be muliplied by the flux          
#        Omega = 2*np.pi*(1-np.sqrt(1-(R_star/r)**2))
#        
#        Fg = (4.0/3.0)*np.pi*(particle_radius**3)*particle_density*G*M_star/r**2
#        
#        S = (L_star/(4.0*np.pi*r**2))*Omega
#        
#        A = np.pi*particle_radius**2
#        
#        Fr = (S*A/c)*Qpr*transmittance
#        
#        ##beta = Fr/Fg
#        
#        beta = 0.3  ## A constant beta for testing 
#        
#        particle_list[i].update_beta(beta)
#    
#    #return particle_object
#    #return beta,Fr,Fg    
#    return None 
    
    
#def p_v(A,B,T):
#    
#    
#    '''
#    Note that from Rik's thesis, A is in units of 10**4 K 
#    so a temperature of 2000K should be entered as 0.2.
#    '''
#    
#    
#    return np.exp(-A/T+B)
    
    
    
    
#def calculate_sublimation_rate(alpha,p_v,A,B,rho_d,T_d):
#    
#    '''
#    
#    The rate of change of radius with respect to time     
#    
#    Equation from Rik's thesis pg 121:    
#    
#    ds/dt = -(alpha*p_v(T_d)/(rho_d))*(mu*m_u/(2*np.pi*k_B*T_d))**(0.5)
#    
#    alpha = evaporation coefficient
#    
#    p_v = partial vapour pressure at phase equilibrium (depends on temperature)
#    
#    mu = molecular weight of dust molecules 
#    
#    T_d is the temperature of the dust
#    
#    '''
#    
#    m_u = 1.6605402E-27 # atomic mass unit in kg 
#    
#    k_B = 1.38064852E-23 # Boltzman constant in m**2 kg s**(-2) K**(-1)
#    
#    ## p_v should actually be a funtion that depends on T 
#    
#    ds_dot = -(alpha*p_v(A,B,T_d)/(rho_d))*(mu*m_u/(2*np.pi*k_B*T_d))**(0.5)
#    
#    ##return ds_dot
#    
#    ## Needs to be negative for decreasing radius 
#    #return -1E-11  ### 
#    return -0.5E-11  ### 

    
def sublimate_particles(particle_list,dt,subrate_proportional_factor):   
    
    ''' the subrate_proportional_factor is the base value that is multiplied 
    by the transmittance, exp(-tau*odm) to give the actual sublimation rate 
    '''
    
    
       
    for i in range(len(particle_list)):
        
        if not OpticalDepthOn:
                
            #subrate = calculate_sublimation_rate(1E-6,p_v,7,35,particle_density,0.2) # T in 10^4 K
            #subrate = -6E-13
            #subrate = -6E-12
            #subrate = -2E-11 
            #subrate = -1.5E-11
            
            subrate = subrate_proportional_factor
            
        if OpticalDepthOn:
            
            subrate = subrate_proportional_factor*(particle_list[i].transmittance)  
            
        new_radius = particle_list[i].radius+(dt*subrate)
        
        if new_radius < 0:
            
            new_radius = 0
        
        particle_list[i].update_radius(new_radius)
        
    if len(particle_list) == 0:
        print 'big problem, empty particle_list'
        
    return particle_list,subrate
    
def sublimate_particleForMultiprocess(Particle):   
    
    ''' the subrate_proportional_factor is the base value that is multiplied 
    by the transmittance, exp(-tau*odm) to give the actual sublimation rate 
    '''   
        
    if not OpticalDepthOn:           
        
        subrate = subrate_proportional_factor
        
    if OpticalDepthOn:
        
        subrate = subrate_proportional_factor*(Particle.transmittance)  
        
    new_radius = Particle.radius+(dt*subrate)
    
    if new_radius < 0:
        
        new_radius = 0
    
    Particle.update_radius(new_radius)
        
       
    return Particle
        
        
        
def remove_sublimated_particle(particle_list,sublimation_diam_threshold):
    
    '''
    A simple but inefficient way to make sure that no particles are skipped
    if one is removed from indexing errors.  Could probably start by counting 
    from the last item of the list but this should work     
    '''

    i = 0    
    
    while i < len(particle_list):
        
        if particle_list[i].radius < sublimation_diam_threshold:
            
            if particle_list[i].name in ParticlesToTrack:
                removed_particle_list.append(particle_list.pop(i)) # put all removed particles in a list 
            else:    
                particle_list.pop(i)  # Just delete the removed particles (can't make the animation)
                #raise Exception            
            
            i = 0
            
        else:
            
            i = i + 1 
    
    return particle_list 
    
def remove_particles_inside_planet(particle_list):
    
    '''
    Remove particles that are inside the planet (have hit the surface)
    
    A simple but inefficient way to make sure that no particles are skipped
    if one is removed from indexing errors.  Could probably start by counting 
    from the last item of the list but this should work     
    '''

    i = 0    
    
    while i < len(particle_list):
        
        particle_pos = particle_list[i].position
        particle_age = particle_list[i].age
        
        distance_to_planet = np.linalg.norm(particle_pos - planet_pos)
        
        if ((distance_to_planet <= rp)&(particle_age > 0)):
            if particle_list[i].name in ParticlesToTrack:
                removed_particles_from_inside_planet.append(particle_list.pop(i))
                print 'removing from inside planet'
                #NumberOfParticlesRemovedFromPlanet += 1 
            else:
                particle_list.pop(i)
                print 'removing from inside planet'
                #NumberOfParticlesRemovedFromPlanet += 1 
            i = 0
            
        else:
            i = i + 1 
            
    return particle_list 
    
    
def RemoveSublimatedParticlesAndInsidePlanetForMultiprocess(Particle):
    
    ## Check first if it's inside the planet:
    
    particle_pos = Particle.position
    particle_age = Particle.age
       
    distance_to_planet = np.linalg.norm(particle_pos - planet_pos)
    
    if (((distance_to_planet <= rp)&(particle_age > 0)) or Particle.radius < ParticleRemovalRadius):
            pass
    
    else:
        return Particle 
    
    
    
    

#def Christoph_calculate_beta(diameters):
#    
#    '''
#    Christoph's code doesn't seem to be correct because it peaks at a radius 
#    of 0.1 mm, not um.  It is out by some orders of magnitude
#    '''
#    
#    betas = 0.15*np.exp(-((np.log(np.clip(diameters,1e-10,1.0))+6.6)**2/0.5)) # radiation/gravity force ratio
#    
#    return betas      
    
    
def MultiModeBeta(radius,BetaMode):
    
    '''
    depending on the mode, defined by the global variable BetaMode 
    do the lognormal beta or the loaded beta function
    
    
    Lognormal:     

    A log normal distribution of beta as a function of particle radius.  
    The parameters a and b were found by empirically by trialling many 
    differnet parameters and comparing to the digitized data from the 
    Kimura 2002 Fig. 2 
    
    takes input as radius in metres
    
    This could also take an optical depth to which can be directly multiplied 
    by beta since beta = Fr/Fg where Fr is affected by the optical depth 
    '''
    
    
    if BetaMode == 'lognormal':
        
        a = 15.44
        b = 1.3
        ## A = 0.78 # 0.78 is required to have the same amplitude as the plot in Kimura et al. 2002 but Christoph used 0.15
        A = 0.15 
        
        betas = A*np.exp(-((np.log(radius)+a)**2/b)) # radiation/gravity force ratio
        
    if BetaMode == 'LoadedBeta':
        
        betas = BetaFunc(radius)
        
    return betas
    
    
    
    
def ApplyBeta(particle_list):
    
    '''
    apply the beta calculated by the function MultiModeBeta 
    to the particle objects
    '''
    
    for i in range(len(particle_list)):
         
        radius = particle_list[i].radius 
         
        #beta = lognormal_beta(radius)*particle_list[i].transmittance_history[-1]###*particle_list[i].transmittance ## Optical thickness has no affect on beta           
        if OpticalDepthOn: 
            beta = MultiModeBeta(radius,BetaMode)*(particle_list[i].transmittance)      
             
        if not OpticalDepthOn:
            beta = MultiModeBeta(radius,BetaMode)
       
        particle_list[i].update_beta(beta)
        #particle_list[i].update_beta(100)

         
    return particle_list 
    
def ApplyBetaToParticleForMultiprocess(Particle):
    
    '''
    apply the beta calculated by the function MultiModeBeta 
    to the particle objects
    '''
         
    radius = Particle.radius 
     
    #beta = lognormal_beta(radius)*particle_list[i].transmittance_history[-1]###*particle_list[i].transmittance ## Optical thickness has no affect on beta           
    if OpticalDepthOn: 
        beta = MultiModeBeta(radius,BetaMode)*(Particle.transmittance)      
         
    if not OpticalDepthOn:
        beta = MultiModeBeta(radius,BetaMode)
   
    Particle.update_beta(beta)
         
    return Particle 
    
   
def animate_top_down(particle_list):       
    
    '''
    
    Couldn't manage to get an actual animation working but instead it saves a 
    png from every time step                  
                    
    '''    
    
    #cmap = mpl.cm.spectral
    

    for timestep in range(ns*no):
    #for timestep in range(2):    
        
        xlist = []
        ylist = []           
        radius_list = []
                
        print 'doing timestep %d' % (timestep)
        
        plt.figure(figsize=((12,12)))        
        
        for particle in particle_list:
            
            if timestep in particle.timestep_history:
                
                              
                timestep_index = np.where(np.array(particle.timestep_history)==timestep)[0][0]                
                                
                x = particle.position_history[timestep_index][0]
                y = particle.position_history[timestep_index][1]
                radius = particle.radius_history[timestep_index]
                
                xlist.append(x)
                ylist.append(y)
                radius_list.append(radius)
                
                #print 'x = %f, y = %f' % (x,y)              
                
        
        plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=1E-2,vmax=5E0)) ## Radius in um 
        cbar = plt.colorbar(ticks = LogLocator(subs=range(10))) # Draw log spaced tick marks on the colour bar 
        cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
        
        
        circle1=plt.Circle((0,0),radius=R_star,color='y')  # 0.7*R_sun as a rough estimate from wikipedia 
        plt.gcf().gca().add_artist(circle1)
        
        circle2=plt.Circle((0,-ap),radius=rp,color='r')  # 1.15 R_earth 1sigma limit from Brogi 2012
        plt.gcf().gca().add_artist(circle2)
        
        plt.ylabel('co-rotating y coordinate (m)')
        plt.xlabel('co-rotating x coordinate (m)')
        plt.title('top down view (equal axes)')     
        
        plt.gca().set_aspect('equal', adjustable='box')  
        
        plt.xlim((-3E9,1E9))
        plt.ylim((-3E9,1E9))        
        
                
        plt.savefig(os.path.join('timestep_figs','topdown','timestep%d'%(timestep)))
        
        plt.close()
        
def plot_last_top_down(particle_list):       
    
    '''
    Plot the last timestep topdown          
    '''    

          
    xlist = []
    ylist = []           
    radius_list = []
            
    print 'doing timestep %d' % (timestep)
    
    plt.figure(figsize=((9,9)))        
    
    for particle in particle_list:
                            
        x = particle.position[0]
        y = particle.position[1]
        radius = particle.radius
        
        xlist.append(x)
        ylist.append(y)
        radius_list.append(radius)   
    
    plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
#    plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=(np.min(radius_list)*1E6),vmax=(np.max(radius_list)*1E6))) ## Radius in um 
#
    cbar = plt.colorbar(ticks = LogLocator(subs=range(10)),fraction=0.014, pad=0.02) # Draw log spaced tick marks on the colour bar 
    cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
    

#    
#    plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral) ## Radius in um 
#    plt.colorbar()
    
    circle1=plt.Circle((0,0),radius=R_star,color='y')  # 0.7*R_sun as a rough estimate from wikipedia 
    plt.gcf().gca().add_artist(circle1)
    
    circle2=plt.Circle((0,-ap),radius=rp,color='k')  # 1.15 R_earth 1sigma limit from Brogi 2012
    plt.gcf().gca().add_artist(circle2)
    
    plt.ylabel('co-rotating y coordinate (m)')
    plt.xlabel('co-rotating x coordinate (m)')
    plt.title('top down view')     
    
   # plt.grid(True)
    
    
#    plt.xlim(topdownxlim)
#    plt.ylim(topdownylim)   
    
    #plt.gca().set_aspect('equal', adjustable='box')  
    plt.gca().set_aspect('equal')  
    
    plt.savefig(os.path.join(OutputDirectory,'topdown.png'))
    
    for i in RadialBins:
        
        plt.plot(PointsOnACircle(i,np.linspace(0,2*np.pi,1000))[0],PointsOnACircle(i,np.linspace(0,2*np.pi,1000))[1],'k')
        
    for j in PhiBins:
        x2,y2 = PointsOnACircle(Rout,j*np.pi/180.)
        plt.plot([0,x2],[0,y2],'k')
        
    plt.savefig(os.path.join(OutputDirectory,'topdownGridLines.png'))
    
    plt.close('all')
    
def SnapshotTopDown(timestep,particle_list):       
       
    xlist = []
    ylist = []           
    radius_list = []           
   
    plt.figure(figsize=((9,9)))        
    
    for particle in particle_list:                         
                            
        x = particle.position[0]
        y = particle.position[1]
        radius = particle.radius
        
        xlist.append(x)
        ylist.append(y)
        radius_list.append(radius)   
    
    plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
    cbar = plt.colorbar(ticks = LogLocator(subs=range(10)),fraction=0.014, pad=0.02) # Draw log spaced tick marks on the colour bar 
    cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
    
    
    circle1=plt.Circle((0,0),radius=R_star,color='y')  # 0.7*R_sun as a rough estimate from wikipedia 
    plt.gcf().gca().add_artist(circle1)
    
    circle2=plt.Circle((0,-ap),radius=rp,color='k')  # 1.15 R_earth 1sigma limit from Brogi 2012
    plt.gcf().gca().add_artist(circle2)
    
    plt.ylabel('co-rotating y coordinate (m)')
    plt.xlabel('co-rotating x coordinate (m)')
    plt.title('top down view')     
    
    
    plt.xlim(topdownxlim)
    plt.ylim(topdownylim)   
    
    #plt.gca().set_aspect('equal', adjustable='box')  
    plt.gca().set_aspect('equal')  
    
    plt.savefig(os.path.join(TopDownAnimationOutputDirectory,'topdown_timestep%d.png'%(timestep)))
    
    plt.close('all')
    
def TrackSingleParticle(particle_list,particleinlist):  
       
    xlist = []
    ylist = []           
    zlist = []
    
    radius_list = []       
    beta_list = []
    
    nett_acceleration_magnitude_list = []
    
    planet_acceleration_list = []
    planet_acceleration_magnitude_list = []
    
    velocity_magnitude_list = []
    
    beta_acceleration_list = []
    beta_acceleration_magnitude_list = []         
    
    ### add every 10th point to these lists and plot them as circles so every 10th point is circled 
    circledx = []
    circledy = []
    circledz = []

    
    SingleParticleOutputDir = os.path.join(OutputDirectory,'TrackSingleParticle','ParticleInList%d'%(particleinlist))    
    
    topdowndir = os.path.join(SingleParticleOutputDir,'TopDown')    
    if not os.path.exists(topdowndir):    
        os.makedirs(topdowndir)
    
    zoomtopdowndir = os.path.join(SingleParticleOutputDir,'ZoomtTopDown')    
    if not os.path.exists(zoomtopdowndir):
        os.makedirs(zoomtopdowndir)
    
    sideview1dir = os.path.join(SingleParticleOutputDir,'Sideview1')
    if not os.path.exists(sideview1dir):
        os.makedirs(sideview1dir)
        
    sideview1zoomdir = os.path.join(SingleParticleOutputDir,'Sideview1zoom')
    if not os.path.exists(sideview1zoomdir):
        os.makedirs(sideview1zoomdir)
    
    distortedsideview1dir = os.path.join(SingleParticleOutputDir,'DistortedSideview1')    
    if not os.path.exists(distortedsideview1dir):
        os.makedirs(distortedsideview1dir)
    
    sideview2dir = os.path.join(SingleParticleOutputDir,'Sideview2') 
    if not os.path.exists(sideview2dir):
        os.makedirs(sideview2dir)
        
    sideview2zoomdir = '%s/Sideview2zoom'%(SingleParticleOutputDir) 
    if not os.path.exists(sideview2zoomdir):
        os.makedirs(sideview2zoomdir)
    
    distortedsideview2dir = os.path.join(SingleParticleOutputDir,'DistortedSideview2') 
    if not os.path.exists(distortedsideview2dir):
        os.makedirs(distortedsideview2dir)    
   
    particle = particle_list[particleinlist]
    
    particle_lifetime_timesteps = len(particle.position_history)
    particle_lifetme_orbits = particle_lifetime_timesteps*dt/p
    
    for i in range(len(particle.position_history)):

        print 'doing age %d of %d'%(i,len(particle.position_history))        
        
        plt.figure(figsize=(figuresize))     
                            
        x = particle.position_history[i][0]
        y = particle.position_history[i][1]
        z = particle.position_history[i][2]
        radius = particle.radius_history[i]
        beta_list.append(particle.beta_history[i])
        
        xlist.append(x)
        ylist.append(y)
        zlist.append(z)
        radius_list.append(radius)   
        
        if i%10==0: #it's the 10th point so add it to the circle drawing vector 
            
            circledx.append(x)
            circledy.append(y)
            circledz.append(z)   
       
        #### Plot topdown 
    
        plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
        cbar = plt.colorbar(ticks = LogLocator(subs=range(10)),fraction=0.014, pad=0.02) # Draw log spaced tick marks on the colour bar 
        cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
        
        plt.scatter(circledx, circledy, s=25, facecolors='none', edgecolors='k')
        
        
        circle1=plt.Circle((0,0),radius=R_star,color='y')  # 0.7*R_sun as a rough estimate from wikipedia 
        plt.gcf().gca().add_artist(circle1)
        
        circle2=plt.Circle((0,-ap),radius=rp,color='k')  # 1.15 R_earth 1sigma limit from Brogi 2012
        plt.gcf().gca().add_artist(circle2)   
        
        plt.ylabel('co-rotating y coordinate (m)')
        plt.xlabel('co-rotating x coordinate (m)')
        plt.title('Total particle lifetime = %d timesteps (%f orbits)'%(particle_lifetime_timesteps,particle_lifetme_orbits))             
        
        plt.xlim(topdownxlim)
        plt.ylim(topdownylim)   
        
        #plt.gca().set_aspect('equal', adjustable='box')  
        plt.gca().set_aspect('equal')  
        
        plt.savefig(os.path.join(topdowndir,'TopDownTrackSingleParticleAge%d.png'%(i)))        
        plt.close('all')
        
        ### Plot zoom topdown 
        
        plt.figure(figsize=(figuresize))     
        
        plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
        cbar = plt.colorbar(ticks = LogLocator(subs=range(10)),fraction=0.014, pad=0.02) # Draw log spaced tick marks on the colour bar 
        cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
        
        plt.scatter(circledx, circledy, s=25, facecolors='none', edgecolors='k')        
        
        circle1=plt.Circle((0,0),radius=R_star,color='y')  # 0.7*R_sun as a rough estimate from wikipedia 
        plt.gcf().gca().add_artist(circle1)
        
        circle2=plt.Circle((0,-ap),radius=rp,color='k')  # 1.15 R_earth 1sigma limit from Brogi 2012
        plt.gcf().gca().add_artist(circle2)
        
        plt.ylabel('co-rotating y coordinate (m)')
        plt.xlabel('co-rotating x coordinate (m)')
        plt.title('Total particle lifetime = %d timesteps (%f orbits)'%(particle_lifetime_timesteps,particle_lifetme_orbits))          
        
        plt.xlim(zoomtopdownxlim)
        plt.ylim(zoomtopdownylim)     
        
        #plt.gca().set_aspect('equal', adjustable='box')  
        plt.gca().set_aspect('equal')  
        
        plt.savefig(os.path.join(zoomtopdowndir,'TopDownZoomTrackSingleParticleAge%d.png'%(i)))  
        
        plt.close('all')
        
        ### Plot sideview1 
        plt.figure(figsize=(figuresize))  
        
        circle2=plt.Circle((-ap,0),radius=rp,color='k')  # 1.15 R_earth 1sigma limit from Brogi 2012
        plt.gcf().gca().add_artist(circle2)
        
#        circle1=plt.Circle((0,0),radius=0.7*696342000,color='y',zorder=0)  # 0.7*R_sun as a rough estimate from wikipedia. zorder puts it on the bottom layer 
#        plt.gcf().gca().add_artist(circle1)       
       
        plt.scatter(ylist,zlist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
        cbar = plt.colorbar(ticks = LogLocator(subs=range(10))) # Draw log spaced tick marks on the colour bar 
        cbar.ax.set_ylabel(r'particle radius [$\mu m$]')  
        
        plt.scatter(circledy, circledz, s=25, facecolors='none', edgecolors='k')
    
        plt.ylabel('co-rotating z coordinate (m)')
        plt.xlabel('co-rotating y coordinate (m)')
        plt.title('side view direction 1: Total particle lifetime = %d timesteps (%f orbits)'%(particle_lifetime_timesteps,particle_lifetme_orbits))     
        
        plt.xlim(sideview1xlim)
        plt.ylim(sideview1ylim)    
        
        plt.savefig(os.path.join(distortedsideview1dir,'TrackSingleDistortedSideview1ParticleAge%d.png'%(i)))
                
        plt.gca().set_aspect('equal', adjustable='box')  
        
        plt.xlim(sideview1xlim)
        plt.ylim(sideview1ylim)
        
        plt.savefig(os.path.join(sideview1dir,'TrackSingleSideview1ParticleAge%d.png'%(i)))         
        
        plt.xlim(sideview1zoomxlim)
        plt.ylim(sideview1zoomylim)
        
        plt.savefig(os.path.join(sideview1zoomdir,'TrackSingleSideview1ZoomParticleAge%d.png'%(i)))      
        
        plt.close('all')
        
        #################################################################
        
     
        
        ### Plot sideview2 
        plt.figure(figsize=(figuresize))  
        
        circle2=plt.Circle((-0,0),radius=rp,color='k')  # 1.15 R_earth 1sigma limit from Brogi 2012
        plt.gcf().gca().add_artist(circle2)
        
#        circle1=plt.Circle((0,0),radius=0.7*696342000,color='y',zorder=0)  # 0.7*R_sun as a rough estimate from wikipedia. zorder puts it on the bottom layer 
#        plt.gcf().gca().add_artist(circle1)
        
        plt.scatter(xlist,zlist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
        cbar = plt.colorbar(ticks = LogLocator(subs=range(10))) # Draw log spaced tick marks on the colour bar 
        cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
        
        plt.scatter(circledx, circledz, s=25, facecolors='none', edgecolors='k')
        
 
        plt.ylabel('co-rotating z coordinate (m)')
        plt.xlabel('co-rotating y coordinate (m)')
        plt.title('side view direction 2: Total particle lifetime = %d timesteps (%f orbits)'%(particle_lifetime_timesteps,particle_lifetme_orbits))     
        
        plt.xlim(sideview2xlim)
        plt.ylim(sideview2ylim)    
        
        plt.savefig(os.path.join(distortedsideview2dir,'TrackSingleDistortedSideview2ParticleAge%d.png'%(i)))
                
        plt.gca().set_aspect('equal', adjustable='box')  
        
        plt.xlim(sideview2xlim)
        plt.ylim(sideview2ylim)
        
        plt.savefig(os.path.join(sideview2dir,'TrackSingleSideview2ParticleAge%d.png'%(i)))     
        
        plt.xlim(sideview2zoomxlim)
        plt.ylim(sideview2zoomylim)
        
        plt.savefig(os.path.join(sideview2zoomdir,'TrackSingleSideview2ZoomParticleAge%d.png'%(i)))   
        
        plt.close('all')
        
    tau_list = particle.od_1_history
    transmittance_list = particle.transmittance_history   
    
    planet_acceleration_list = particle.AccelerationDueToPlanet_history
    
        
    for i in range(len(particle.acceleration_history)):
        

        beta_acceleration_list.append(particle.acceleration_history[i]-particle.AccelerationDueToPlanet_history[i])
        
        planet_acceleration_magnitude_list.append(np.linalg.norm(particle.AccelerationDueToPlanet_history[i]))
        nett_acceleration_magnitude_list.append(np.linalg.norm(particle.acceleration_history[i]))
        beta_acceleration_magnitude_list.append(np.linalg.norm(particle.acceleration_history[i]-particle.AccelerationDueToPlanet_history[i]))
        velocity_magnitude_list.append(np.linalg.norm(particle.velocity_history[i]))  ## TO TAKE THE VELOCITY THAT LED TO THIS ACCELERATION 
        
    plt.figure(figsize=(figuresize)) 
    
   
    plt.plot(beta_list[1:],beta_acceleration_magnitude_list,'.')
    plt.ylabel('beta acceleration (m/s)')
    plt.xlabel('beta')
    plt.close()
    
    plt.figure(figsize=(figuresize)) 
    
    plt.plot(beta_list[1:],nett_acceleration_magnitude_list,'.')
    plt.ylabel('nett acceleration (beta+planet) (m/s)')
    plt.xlabel('beta')
    plt.savefig(os.path.join(SingleParticleOutputDir,'beta_nett_acceleration.png'))
    plt.close()
    
    plt.figure(figsize=(figuresize))     
    plt.plot(beta_list[1:],planet_acceleration_magnitude_list,'.')
    plt.ylabel('planet acceleration (m/s)')
    plt.xlabel('beta')
    plt.savefig(os.path.join(SingleParticleOutputDir,'beta_planet_acceleration.png'))        
    plt.close()
    
    #### Plots with time 
    
    plt.figure(figsize=(figuresize))     
    plt.plot(beta_list[1:])
    plt.ylabel('beta')
    plt.xlabel('time')
    plt.savefig(os.path.join(SingleParticleOutputDir,'beta_time.png'))       
    plt.close()
    
    plt.figure(figsize=(figuresize))     
    plt.plot(velocity_magnitude_list)
    plt.ylabel('velocity (m/s)')
    plt.xlabel('time')
    plt.savefig(os.path.join(SingleParticleOutputDir,'velocity_time.png'))        
    plt.close()
    
    plt.figure(figsize=(figuresize))     
    plt.plot(tau_list[1:])
    plt.ylabel('tau')
    plt.xlabel('time')
    plt.savefig(os.path.join(SingleParticleOutputDir,'tau_time.png'))        
    plt.close()
    
    plt.figure(figsize=(figuresize))     
    plt.plot(transmittance_list[1:])
    plt.ylabel('transmittance')
    plt.xlabel('time')
    plt.savefig(os.path.join(SingleParticleOutputDir,'transmittance_time.png'))        
    plt.close()
    
    
    
    plt.figure(figsize=(figuresize))     
    plt.plot(beta_acceleration_magnitude_list)
    plt.ylabel('beta acceleration (m/s)')
    plt.xlabel('time')
    plt.savefig(os.path.join(SingleParticleOutputDir,'beta_acc_time.png'))        
    plt.close()
    
    plt.figure(figsize=(figuresize))     
    plt.plot(nett_acceleration_magnitude_list)
    plt.ylabel('nett acceleration (m/s)')
    plt.xlabel('time')
    plt.savefig(os.path.join(SingleParticleOutputDir,'nett_acc_time.png'))        
    plt.close()
    
    
    plt.figure(figsize=(figuresize))     
    plt.plot(planet_acceleration_magnitude_list)
    plt.ylabel('planet acceleration (m/s)')
    plt.xlabel('time')
    plt.savefig(os.path.join(SingleParticleOutputDir,'planet_acc_time.png'))        
    plt.close()
    
    ##### Write the output file 
    
    beta_list_omitted_fist = beta_list[1:]  ## omitting the first so that the creation values with no matching acceleration values are ignored 
    position_history_omitted_first = particle.position_history[1:]
    radius_list_omitted_first = particle.radius_history[1:]
    
    tau_list_omitted_first = particle.od_1_history[1:]
    transmittance_list_omitted_First = particle.transmittance_history[1:]
    
    AccelerationSummayFile = open(os.path.join(SingleParticleOutputDir,'AccelerationSummaryFile.txt'),'a')
    AccelerationSummayFile.write('''All positions in metres, all accelerations in m/s**2
-----------------------------------------------------------------------------
''')
    
    for i in range(len(beta_list_omitted_fist)):
        
        AccelerationString = '''Particle age %d

Position 
%s

Beta 
%s

Radius 
%s

optical depth (tau) 
%s

transmittance exp(-tau) 
%s

velocity that leads to this acceleration 
%s

Nett acceleration 
%s

acceleration due to beta acceleraiton 
%s 

acceleration due to planet 
%s

------------------------------------------------------------------------
''' % (i,position_history_omitted_first[i],beta_list_omitted_fist[i],radius_list_omitted_first[i],
        tau_list_omitted_first[i], transmittance_list_omitted_First[i],particle.velocity_history[i],
        particle.acceleration_history[i],beta_acceleration_list[i],planet_acceleration_list[i]) 
        
        
        AccelerationSummayFile.write(AccelerationString)
    
    AccelerationSummayFile.close()
    
    
    
    
    
def SnapshotZoomTopDown(timestep,particle_list):       
       
    xlist = []
    ylist = []           
    radius_list = []           
   
    plt.figure(figsize=((9,9)))        
    
    for particle in particle_list:
                            
        x = particle.position[0]
        y = particle.position[1]
        radius = particle.radius
        
        xlist.append(x)
        ylist.append(y)
        radius_list.append(radius)   
    
    plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
    cbar = plt.colorbar(ticks = LogLocator(subs=range(10)),fraction=0.014, pad=0.02) # Draw log spaced tick marks on the colour bar 
    cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
    
    
    circle1=plt.Circle((0,0),radius=R_star,color='y')  # 0.7*R_sun as a rough estimate from wikipedia 
    plt.gcf().gca().add_artist(circle1)
    
    circle2=plt.Circle((0,-ap),radius=rp,color='k')  # 1.15 R_earth 1sigma limit from Brogi 2012
    plt.gcf().gca().add_artist(circle2)
    
    plt.ylabel('co-rotating y coordinate (m)')
    plt.xlabel('co-rotating x coordinate (m)')
    plt.title('top down view')     
    
    
    plt.xlim(zoomtopdownxlim)
    plt.ylim(zoomtopdownylim)   
    
    #plt.gca().set_aspect('equal', adjustable='box')  
    plt.gca().set_aspect('equal')  
    
    plt.savefig(os.path.join(TopDownZoomAnimationOutputDirectory,'topdownzoom_timestep%d.png'%(timestep)))
  
    plt.close('all')  
    
def plot_last_top_down_zoomin(particle_list):       
    
    '''
    Plot the last timestep topdown          
    '''    

            
    xlist = []
    ylist = []   
    radius_list = []

            
    print 'doing timestep %d' % (timestep)
    
    plt.figure(figsize=((9,9)))        
    
    for particle in particle_list:                
                        
        x = particle.position[0]
        y = particle.position[1]
        radius = particle.radius
        
        xlist.append(x)
        ylist.append(y)
        radius_list.append(radius)   
    
    plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
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
    
    plt.savefig(os.path.join(OutputDirectory,'zoom_in_topdown.png'))
    
    for i in RadialBins:
        
        plt.plot(PointsOnACircle(i,np.linspace(0,2*np.pi,1000))[0],PointsOnACircle(i,np.linspace(0,2*np.pi,1000))[1],'k')
        
    for j in PhiBins:
        x2,y2 = PointsOnACircle(Rout,j*np.pi/180.)
        plt.plot([0,x2],[0,y2],'k')
        
    plt.savefig(os.path.join(OutputDirectory,'zoom_in_topdown_GridLines.png'))
    
    plt.close('all')



def SaveFinaTimestepArraysForPlotting(particle_list):
    
    xlist = []
    ylist = []   
    zlist = []        
    radius_list = []
    odlist = []
    particle_name_list = []
    particle_final_particle_list_index = []
    
    for i in range(len(particle_list)):
        
        particle = particle_list[i]
        
        x = particle.position[0]
        y = particle.position[1]
        z = particle.position[2]
        radius = particle.radius
        particle_name = particle.name 
        
        xlist.append(x)
        ylist.append(y)
        zlist.append(z)
        radius_list.append(radius)
        odlist.append(particle.od_1)
        particle_name_list.append(particle_name)
        particle_final_particle_list_index.append(i)
        
    np.save(os.path.join(OutputDirectory,'FinalTimestep_xpositions.npy'),np.array(xlist))
    np.save(os.path.join(OutputDirectory,'FinalTimestep_ypositions.npy'),np.array(ylist))
    np.save(os.path.join(OutputDirectory,'FinalTimestep_zpositions.npy'),np.array(zlist))
    np.save(os.path.join(OutputDirectory,'FinalTimestep_radius.npy'),np.array(radius_list))
    np.save(os.path.join(OutputDirectory,'FinalTimestep_OpticalDepths.npy'),np.array(odlist))
    np.save(os.path.join(OutputDirectory,'FinalTimestep_particlenames.npy'),np.array(particle_name_list))
    np.save(os.path.join(OutputDirectory,'FinalTimestep_particle_list_index.npy'),np.array(particle_final_particle_list_index))
    
def SaveState(particle_list,timestep):
    
    xlist = []
    ylist = []   
    zlist = []        
    radius_list = []
    odlist = []
    particle_name_list = []
    particle_final_particle_list_index = []
    
    for i in range(len(particle_list)):
        
        particle = particle_list[i]
        
        x = particle.position[0]
        y = particle.position[1]
        z = particle.position[2]
        radius = particle.radius
        particle_name = particle.name 
        
        xlist.append(x)
        ylist.append(y)
        zlist.append(z)
        radius_list.append(radius)
        odlist.append(particle.od_1)
        particle_name_list.append(particle_name)
        particle_final_particle_list_index.append(i)
        
    np.save(os.path.join(SaveStatesDir,'Timestep%d_xpositions.npy'%(timestep)),np.array(xlist))
    np.save(os.path.join(SaveStatesDir,'Timestep%d_ypositions.npy'%(timestep)),np.array(ylist))
    np.save(os.path.join(SaveStatesDir,'Timestep%d_zpositions.npy'%(timestep)),np.array(zlist))
    np.save(os.path.join(SaveStatesDir,'Timestep%d_radius.npy'%(timestep)),np.array(radius_list))
    np.save(os.path.join(SaveStatesDir,'Timestep%d_OpticalDepths.npy'%(timestep)),np.array(odlist))
    np.save(os.path.join(SaveStatesDir,'Timestep%d_particlenames.npy'%(timestep)),np.array(particle_name_list))
    #np.save(os.path.join(SaveStatesDir,'Timestep%d_particle_list_index.npy'%(timestep)),np.array(particle_final_particle_list_index))

def animate_side_view(particle_list,view_direction=1):
           
    '''
    
    Couldn't manage to get an actual animation working but instead it saves a 
    png from every time step                  
                    
    '''
    
    for timestep in range(ns*no):
    #for timestep in [100]:    
        
        
        ylist = []   
        zlist = []   
        radius_list = []
                
        print 'doing timestep %d' % (timestep)
        
        plt.figure(figsize=((9,9)))        
        
        for particle in particle_list:
            
            if timestep in particle.timestep_history:
                
                timestep_index = np.where(np.array(particle.timestep_history)==timestep)[0][0]                
                                
                if view_direction==1:
                    y = particle.position_history[timestep_index][1]
                    z = particle.position_history[timestep_index][2]
                if view_direction==2:
                    y = particle.position_history[timestep_index][0]
                    z = particle.position_history[timestep_index][2]
                    
                radius = particle.radius_history[timestep_index]
                
                ylist.append(y)
                zlist.append(z)
                radius_list.append(radius)
                
                
                #print 'y = %f, z = %f' % (y,z)    
      
        circle1=plt.Circle((0,0),radius=R_star,color='y',zorder=0)  # 0.7*R_sun as a rough estimate from wikipedia. zorder puts it on the bottom layer 
        plt.gcf().gca().add_artist(circle1)
        
        plt.scatter(ylist,zlist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=1E-2,vmax=5E0)) ## Radius in um 
        cbar = plt.colorbar(ticks = LogLocator(subs=range(10))) # Draw log spaced tick marks on the colour bar 
        cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
        
        if view_direction==1:
            plt.ylabel('co-rotating z coordinate (m)')
            plt.xlabel('co-rotating y coordinate (m)')
            plt.title('side view direction %d (unequal axes)'%(view_direction))     
            
            plt.xlim((-3E9,1E9))
            plt.ylim((-4E7,4E7))    
            
#            plt.xlim((-3E9,0))
#            plt.ylim((-0.5E7,0.5E7))
            
        if view_direction==2:
            plt.ylabel('co-rotating z coordinate (m)')
            plt.xlabel('co-rotating x coordinate (m)')
            plt.title('side view direction %d (unequal axes)'%(view_direction))     
            
            
    #        plt.gca().set_aspect('equal', adjustable='box')  
    #        
            plt.xlim((-3E9,1E9))
            plt.ylim((-4E7,4E7))    
        
                
        plt.savefig(os.path.join('timestep_figs','sideview','timestep%d'%(timestep)))
        
        plt.close()         
        
def plot_last_side_view(particle_list,view_direction=1):
           
    '''
    Plot the last timestep sideview 
    '''
    
    
    cmap = mpl.cm.spectral
    
    timestep = ns*no-1        
        
    ylist = []   
    zlist = []   
    radius_list = []
            
    print 'doing timestep %d' % (timestep)
    
    plt.figure(figsize=((12,12)))        
    
    for particle in particle_list:
                            
        if view_direction==1:
            y = particle.position[1]
            z = particle.position[2]
        if view_direction==2:
            y = particle.position[0]
            z = particle.position[2]
            
        radius = particle.radius
        
        ylist.append(y)
        zlist.append(z)
        radius_list.append(radius)

    circle1=plt.Circle((0,0),radius=R_star,color='y',zorder=0)  # 0.7*R_sun as a rough estimate from wikipedia. zorder puts it on the bottom layer 
    plt.gcf().gca().add_artist(circle1)
    
    plt.scatter(ylist,zlist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
    cbar = plt.colorbar(ticks = LogLocator(subs=range(10))) # Draw log spaced tick marks on the colour bar 
    cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
    
    if view_direction==1:
        plt.ylabel('co-rotating z coordinate (m)')
        plt.xlabel('co-rotating y coordinate (m)')
        plt.title('side view direction %d'%(view_direction))     
        
        plt.savefig(os.path.join(OutputDirectory,'distorted_sideview%d.png'%(view_direction)))
        
        plt.gca().set_aspect('equal', adjustable='box')  
        
        plt.savefig(os.path.join(OutputDirectory,'sideview%d.png'%(view_direction)))
        
        plt.xlim(sideview1xlim)
        plt.ylim(sideview1ylim)
        
        
        
#            plt.xlim((-3E9,0))
#            plt.ylim((-0.5E7,0.5E7))
        
    if view_direction==2:
        plt.ylabel('co-rotating z coordinate (m)')
        plt.xlabel('co-rotating x coordinate (m)')
        plt.title('side view direction %d'%(view_direction))     
        
        plt.savefig(os.path.join(OutputDirectory,'distorted_sideview%d.png'%(view_direction)))
        
        plt.gca().set_aspect('equal', adjustable='box')  
        

        
        plt.xlim(sideview2xlim)
        plt.ylim(sideview2ylim)    
       
        plt.savefig(os.path.join(OutputDirectory,'sideview%d.png'%(view_direction)))   
    
    plt.close('all')         
    
def SnapshotSideView(timestep,particle_list,view_direction=1):
           
    '''
    Plot the last timestep sideview 
    '''
    
           
    ylist = []   
    zlist = []   
    radius_list = []
                  
    plt.figure(figsize=((12,12)))        
    
    for particle in particle_list:
                            
        if view_direction==1:
            y = particle.position[1]
            z = particle.position[2]
        if view_direction==2:
            y = particle.position[0]
            z = particle.position[2]
            
        radius = particle.radius
        
        ylist.append(y)
        zlist.append(z)
        radius_list.append(radius)
            
            
            #print 'y = %f, z = %f' % (y,z)    
  
    circle1=plt.Circle((0,0),radius=R_star,color='y',zorder=0)  # 0.7*R_sun as a rough estimate from wikipedia. zorder puts it on the bottom layer 
    plt.gcf().gca().add_artist(circle1)
    
    plt.scatter(ylist,zlist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
    cbar = plt.colorbar(ticks = LogLocator(subs=range(10))) # Draw log spaced tick marks on the colour bar 
    cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
    
    if view_direction==1:
        plt.ylabel('co-rotating z coordinate (m)')
        plt.xlabel('co-rotating y coordinate (m)')
        plt.title('side view direction %d'%(view_direction))     
        
        plt.xlim(sideview1xlim)
        plt.ylim(sideview1ylim)    
        
        plt.savefig(os.path.join(Sideview1DistortedAnimationOutputDirectory,'distorted_sideview%d_timestep%d.png'%(view_direction,timestep)))
                
        plt.gca().set_aspect('equal', adjustable='box')  
        
        plt.xlim(sideview1xlim)
        plt.ylim(sideview1ylim)
        
        plt.savefig(os.path.join(Sideview1AnimationOutputDirectory,'sideview%d_timestep%d.png'%(view_direction,timestep)))              

        
    if view_direction==2:
        plt.ylabel('co-rotating z coordinate (m)')
        plt.xlabel('co-rotating x coordinate (m)')
        plt.title('side view direction %d'%(view_direction))     
        
        plt.xlim(sideview2xlim)
        plt.ylim(sideview2ylim)
        
        plt.savefig(os.path.join(Sideview2DistortedAnimationOutputDirectory,'distorted_sideview%d_timestep%d.png'%(view_direction,timestep)))
                
        plt.gca().set_aspect('equal', adjustable='box')          

        plt.xlim(sideview2xlim)
        plt.ylim(sideview2ylim)  
       
        plt.savefig(os.path.join(Sideview2AnimationOutputDirectory,'sideview%d_timestep%d.png'%(view_direction,timestep)))
    
    plt.close('all')         
        
def RotateVector(vector,theta):
    
    '''
    
    theta input in degrees 

    
    
    Using the rotation matrix 
    
    [x',y',z] = R(theta)*[x,y]
    
    where R(theta) for a rotation about the z axis is:
    
    [cos(t) -sin(t) 0]
    [sin(t) cos(t) 0]
    [0      0      1]
    
    '''  
    
    t = -np.pi*theta/180. #converting theta to radians and changing sign so on top down view it rototates clockwise 
    
    vect = np.matrix(vector.reshape(3,1))
    
    rot_matrix = np.matrix([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
    
    newVect = rot_matrix*vect
    
    results = np.empty((3))
    
    results[0] = newVect[0]
    results[1] = newVect[1]
    results[2] = newVect[2]
    
    
    return results
    
def FindDistanceFromPointToLine(point,line_direction):
    
    
    '''
    From Mathworld:
    
    x2,x1 are two points to define a line 
    x0 is the point to which the distance to the line is to be calculated 
    
    d = np.norm(np.cross((x2-x1),(x1-x0)))/np.norm(x2-x1)
    '''
    
    x2 = line_direction
    x1 = np.array([0,0,0]) #a point at the origin to define the radial line from the centre of the star 
    x0 = point

    return np.linalg.norm(np.cross((x2-x1),(x1-x0)))/np.linalg.norm(x2-x1)
    
def HG(theta,g):
    
    '''
    The Henyey Greenstein 1941 function
    '''
    
    results = (1.0-g**2)/(4.0*np.pi*(1.0 - 2.0*g*np.cos(theta)+g**2)**(3.0/2.0))
        
    return results
    
def FindAngle(point):
    
    '''
    This was made for returning a difference between 0 and 360 anticlockwise 
    so it doesn't give the true angular difference between two vectors     
    
    calculates the angle between the reference direction (taken to be "up" in the 
    top down plots [0,1,0] and a vector from the 
    centre of the star to the given point in the clockwise direction
    '''
    
    v = np.array([0,1,0]) #"up" on the topdown plot.  Opposite side of star from planet position 
    #v = reference_direction
    u = point # Vector is from origin to point so point - [0,0,0]
    
    angle = (180/np.pi)*np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))
    
    ### This returns the shortest anglular seperatation so need the points on the left to have 180 added to the angle 

    if u[0] >= 0: ## If the x value is positive 
    
        return angle
        
    if u[0] <= 0:
        
        return 360. - angle 
        
def FindAngleBetweenVectors(point,reference_direction):

    '''
    calculates the angle between the reference direction and a vector from the 
    centre of the star to the given point in the clockwise direction
    '''
    
    #v = np.array([0,1,0]) #"up" on the topdown plot.  Opposite side of star from planet position 
    v = reference_direction
    u = point # Vector is from origin to point so point - [0,0,0]
    
    angle = (180/np.pi)*np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))
    
    return angle
        
def CalculateHG_Contribution(position,lightcurve_beam_direction,delta_theta,HG_g_parameter):
    '''
    delta_theta is the number of timesteps per orbit
    '''
    
    distance_star_to_particle = cart2sph(position)[0]
    
    angular_size_star = np.arctan(R_star/distance_star_to_particle) 
    
    
    AngleToParticle = FindAngle(position)
    
    AngleToBeamDirection = FindAngle(lightcurve_beam_direction)
    
    ContributingScatteringAngle = AngleToParticle - AngleToBeamDirection 
    
    ## To get the scattering over a cone instead of at a single angle 
    HG_scattering_angles_deg = np.linspace(ContributingScatteringAngle-delta_theta/2,ContributingScatteringAngle+delta_theta/2,100)
        
    scattered_flux = HG(HG_scattering_angles_deg*(np.pi/180.),HG_g_parameter)
    #scattered_flux = scattered_flux - min(scattered_flux)    
    
    
     ### Note this actually doesn't really work.  Only a small region around 0 are reduced discontinuously.  Better just to leave it out 
     ### Since the scattering angle away from the star is always 0, so this just sets any scattering towards the star to be exactly zero.
     ### Not really neccessary since the scattering in directions close to 180 in the direction of the star is normally zero anyway 
    
#    towards_star_indices = np.where((HG_scattering_angles_deg>180.0-angular_size_star)&(HG_scattering_angles_deg<180.0+angular_size_star))
#    
#    if len(towards_star_indices[0]) > 0:
#        
#        'print scattering in the direction of the star was set to zero'
#        print towards_star_indices
#        
#        
#        #raise Exception 
#        
#        scattered_flux[towards_star_indices] = 0.0
    
    return np.sum(scattered_flux)
    
    
def diff_volume(p,t,r):
    '''
    IMPORTANT NOTE:
    theta (t) and phi (p) in this function and FindParticleDensities are the 
    opposite to how they are defined in the rest of the code.  
    Here theta is the azimuth (0 to 2pi) and phi is the elevation (0 to pi)
    
    Used in FindParticleDensities in the calculation of the volume element.
    
    '''
    return r**2*np.sin(p)
    
def WriteDensityFile(Rin,Rout,nR,nTheta,nPhi,npart,nsize,nTemp,RadialBins,ThetaBins,PhiBins,mass_densities,comp,filename):
    
    '''
    Rin and Rout are input in m and are converted to AU in this function
    phi and theta bins are input in degrees and are converted to radians in this function   
    RadialBins is input in m and converted to cm in this function    
    '''
    
    print 'writing density file'

    Rin_AU = Rin*6.68459E-12
    Rout_AU = Rout*6.68459E-12

    PrimaryHeader = pyfits.Header([('Rin',Rin_AU),('Rout',Rout_AU),('nR',nR),('nTheta',nTheta),
                                   ('nPhi',nPhi),('npart',npart),('nsize',nsize),('nTemp',nTemp),
                                 ('nHDU',5),('HIERARCH HDU 1','RGRID'),('HIERARCH HDU 2','TGRID'),
                                 ('HIERARCH HDU 3','PGRID'),('HIERARCH HDU 4','DENS'),('HIERARCH HDU 5','COMP')])

    hdu0 = pyfits.PrimaryHDU([],header=PrimaryHeader)
    
    hdulist = pyfits.HDUList([hdu0])
    
    hdu1 = pyfits.ImageHDU(RadialBins*100.0)    ### Converting m to cm 
    hdulist.append(hdu1)
    
    hdu2 = pyfits.ImageHDU(ThetaBins*np.pi/180.) ##Converting to radians and writing 
    hdulist.append(hdu2)
    
    hdu3 = pyfits.ImageHDU(PhiBins*np.pi/180.) ##Converting to radians and writing 
    hdulist.append(hdu3)
    
    hdu4 = pyfits.ImageHDU(mass_densities)
    hdulist.append(hdu4)
    
    hdu5 = pyfits.ImageHDU(comp)
    hdulist.append(hdu5)
    
    hdulist.writeto(filename,clobber=True)
    
def SortParticlesIntoGrids(particle_list,Rout2,planet_pos):
    
    '''
    Makes two new lists, one from the particles that should be in 
    grid 1 (the larger, low resolution grid centered on the star)
    and another for grid 2 (the higher resoultion grid centered on the planet)
    '''
    
    grid1_particles = []
    grid2_particles = []
    
    for i in range(len(particle_list)):
        
        particle_pos = particle_list[i].position 
        
        distance_to_planet = np.linalg.norm(particle_pos - planet_pos)
        
        if distance_to_planet > Rout2: #if it's greater than the outer radius of the high res grid centered on the planet 
        
            grid1_particles.append(particle_list[i]) #append it to grid 1
            
        if distance_to_planet <= Rout2: #if it's inside the outer radius of the high res grid centered on the planet
            
            grid2_particles.append(particle_list[i]) #append it to grid 1
        
    ## Check that every particle was assigned to either grid 1 or 2             
    if (len(grid1_particles) + len(grid2_particles)) != len(particle_list):
        raise Exception 
            
    return grid1_particles, grid2_particles
    

    
def FindParticleDensitiesShiftedOriginComp(nR,nTheta,nPhi,RadialBins,ThetaBins,PhiBins,particle_list,DensityScalingFactor,sizebins,nTemp,nsize,npart,diff_volume, shift = np.array([0,0,0])):
    
    '''
    
    IMPORTANT NOTE:
    theta (t) and phi (p) in this function and diff_volume are the 
    opposite to how they are defined in the rest of the code.
    Here theta is the azimuth (0 to 2pi) and phi is the elevation (0 to pi).
    
    Volumes are in cubic metres so these will need to be converted to cubic 
    astronomical units for input into MCMAX3D.  The grid should be consistent 
    though because it is defined in terms of the maximum and mimumim radius 
    and so many evenly distributed bins between those radii.  
    
    
    Find the particle densities for input into MCMAX3D
    
    Gives length of tail in R_star 
    
    '''
    
    print 'finding mass densities'
        
    p = GetAllxyzCoords(particle_list) ## positions in m in Cartesiain coordinates 
        
    shifted_p = p - shift
        
    n = len(particle_list)  ## number of particles 
    
    sph = cart2sph(shifted_p) #positions in specical coordinate 
    
    mass_density = np.zeros((nPhi,nTheta,nR)) 
    TotalMass = 0.0
    
    comp = np.zeros((nTemp,nsize,npart,nPhi,nTheta,nR))
    
    
    
    ## Here np.digitize is used to determine the bin that a value belongs in.
    ## The index given by digitize is the index of the right limit of the bin given in 
    ## the bin vector  
        
    xr = np.digitize(sph[:,0],RadialBins)
    
    zeroindices = np.where((xr==0))
    toohighindices = np.where(xr==len(RadialBins))        
    
    xr[zeroindices] = 1                   #put values that are below the lower bound into the first bin
    xr[toohighindices] = len(RadialBins)-1   #put values that are above the upper bound into the last bin     
    
    xa = np.digitize(sph[:,1]*180.0/np.pi+180.0,PhiBins) #the position around the star (azimuth, 0 to 360 degrees)
    print 'tail from phi = %f to %f' % (np.min(PhiBins[xa]),np.max(PhiBins[xa]))
    
    xe = np.digitize(sph[:,2]*180.0/np.pi,ThetaBins) ## The equator in the spherical coords is 0 degrees but it is theta = 90 in MCMAX ThetaBins
    
    for i in range(n):           
        
        particle_radius = particle_list[i].radius 
        
        xsize = np.digitize(particle_radius,sizebins)
        if xsize == len(sizebins): # for high optical depths, the last particle doesn't sublimate so stays at 1um (or close enough) and since the right bin boundary is not included, it puts it outside the bins, so need to subtract 1 to make it go in the right most bin 
            xsize = xsize - 1 
            
            
        
        particle_mass_kg = ((4.0/3.0)*np.pi*particle_radius**3)*particle_density ## Assuming spherical particles 
        
        particle_mass_g = particle_mass_kg*1000.0
        
        TotalMass += particle_mass_kg*DensityScalingFactor     
        
        #### on 23 August 2017 changed this to just use the precomputed volume instead of computing it again         
        
#        r1 = RadialBins[xr[i]-1]
#        r2 = RadialBins[xr[i]]
#        
#        t1 = PhiBins[xa[i]-1]*np.pi/180.0 ## NOTE!! theta (t) for this function is phi in rest of code
#        t2 = PhiBins[xa[i]]*np.pi/180.0 ## NOTE!! theta (t) for this function is phi in rest of code
#        
#        p1 = ThetaBins[xe[i]-1]*np.pi/180.0 ## NOTE!! phi (p) for this function is theta in the rest of the code 
#        p2 = ThetaBins[xe[i]]*np.pi/180.0 ## NOTE!! phi (p) for this function is theta in the rest of the code 
        
        #volume = tplquad(diff_volume, r1, r2, lambda r:   t1, lambda r:   t2, lambda r,t: p1, lambda r,t: p2)[0] #in cubic metres 
        
        volume = volumes[xa[i]-1,xe[i]-1,xr[i]-1]
        
        volume_cm = 1000000.0*volume        
                                              
    #        volume_matrix[xr[i]-1,xa[i]-1,xe[i]-1] = volume                                
    #
    #        number_density[xr[i]-1,xa[i]-1,xe[i]-1]=number_density[xr[i]-1,xa[i]-1,xe[i]-1] + 1.0/volume ## counting the number density 
        
        mass_density[xa[i]-1,xe[i]-1,xr[i]-1] = mass_density[xa[i]-1,xe[i]-1,xr[i]-1] + DensityScalingFactor*particle_mass_g/volume_cm
        comp[0,xsize-1,0,xa[i]-1,xe[i]-1,xr[i]-1] = comp[0,xsize-1,0,xa[i]-1,xe[i]-1,xr[i]-1] + DensityScalingFactor*particle_mass_g         
        
    ###return number_density,volume_matrix 
        
    LengthOfTail = (((np.max(PhiBins[xa])-np.min(PhiBins[xa]))/360.)*2*np.pi*ap)/R_star 
        
    
    if str(shift)==str(np.array([0, 0, 0])):
        logfilename = os.path.join(OutputDirectory,'MassDensityGrid1Log.txt')
        
        logcontents = '''Number of particles in tail = %d
Total mass of tail (kg) = %3E
Density Scaling Factor = %3E

Tail position info:

phi: ranges from (degrees) phi = %f to %f with mean %f (length of %f Rstar)

theta: ranges from (degrees) theta = %f to %f with mean %f

radius: ranges from (m) r = %3E to %3E with mean %3E'''% (n,TotalMass,DensityScalingFactor,
                                                       np.min(PhiBins[xa]),np.max(PhiBins[xa]),np.mean(PhiBins[xa]),LengthOfTail,
                                                       np.min(ThetaBins[xe]),np.max(ThetaBins[xe]),np.mean(ThetaBins[xe]),
                                                       np.min(RadialBins[xr]),np.max(RadialBins[xr]),np.mean(RadialBins[xr]))           
        
        
    else:
        logfilename = os.path.join(OutputDirectory,'MassDensityGrid2log.txt')
        
        logcontents = '''Number of particles in tail = %d
Total mass of tail (kg) = %3E
Density Scaling Factor = %3E

Tail position info:
    
phi: ranges from (degrees) phi = %f to %f with mean %f
    
theta: ranges from (degrees) theta = %f to %f with mean %f
    
radius: ranges from (m) r = %3E to %3E with mean %3E'''% (n,TotalMass,DensityScalingFactor,
                                                       np.min(cart2sph(p)[:,1])*180./np.pi+180,np.max(cart2sph(p)[:,1])*180./np.pi+180,np.mean(cart2sph(p)[:,1])*180./np.pi+180,  #The angles of the unshifted particles 
                                                       np.min(cart2sph(p)[:,2])*180./np.pi+90,np.max(cart2sph(p)[:,2])*180./np.pi+90,np.mean(cart2sph(p)[:,2])*180./np.pi+90,
                                                       np.min(cart2sph(p)[:,0]),np.max(cart2sph(p)[:,0]),np.mean(cart2sph(p)[:,0]))  
        
        
    logfile = open(logfilename,'w')
 
    
    logfile.write(logcontents)
    
    logfile.close()
    
    
    return mass_density,comp, LengthOfTail

    
def PointsOnACircle(r,angles,originx=0,originy=0):
    '''
    Return x and y points for the positions
    on a circle given an origin and a vector of angles.
    
    Angles in radians.
    '''
    
    x = originx + r*np.cos(angles)
    y = originy + r*np.sin(angles)

    return x,y    
    
def WriteInputSummary():
    
    '''
    Write a file which summarises all inputs
    '''
    
    towrite='''Input Summary:
BetaFile = %s
OpticalDepthOn = %s
ConstantParticleRadius = %s
LaunchMethod = %s
ConeDirection = %s
cone_opening_angle (degrees) = %f
--------------------
Particle Density (kg/m^3) = %f
Particle Denisty Multiplier = %0.2E 
Optical Density Multiplier (odm) = %.2E
Total number of particles in simulation = %.1E  
Number of orbits = %d
Numper of time steps per orbit (ns) = %d 
Particles added per time step (nps) = %d
Outburst timesteps = %s
Outburst nps multiplicitive factor = %f
No ejection timesteps = %s
Sublimation prop. factor (sub rate with no sheilding) (radius change (m)/s) = %.3E
Example sublimation rate for random particle (radius change (m)/s) = %.3E 
Tail length (stellar radii) = %f 
-----------------------------------
K_B (SI units) = %.4E
mass of gas particle (kg) = %.4E
temperature of gas (K) = %.4E
Gas surface density (number per m^3) = %.4E
-----------------------------------
Planet mass (kg) = %.3E
Planet radius (m) = %.3E
Planet orbital period (s) = %f
Time step (s) = %f 
Planet orbital radius (m) %.3E 
Planet escape velocity (m/s) = %.3E
Planet orbital velocity (m/s) = %.3E
Planet position (co-rotating coordinates, m) = %s 
----------------------------------------
Particle Launch speed (escape velocity units) = %.3f
Particle Launch speed (m/s) = %.3E 

Initial particle radius (m) = %.3E
Particle Removal radius (m) = %.3E

If particle size distribution is used:

size that 1/e particles have = %0.3E 
Minimum size from distribution (amin) = %0.3E 
Maximum  size from distribution (amax) = %0.3E 
-----------------------------------------------

Simple light curve beam radius = %.5E 
Number of steps in simple light curve = %d 
Simple light curve scattering scaling factor = %.5E 
Simple light curve HG parameter = %f 

-------------------------------------
Grid 1: 
Rin = %.3E
Rout = %.3E
nR = %d
nTheta = %d
nPhi = %d
npart = %d
nTemp = %d
nsize = %d

Grid 2 (if used): 

Rin = %.3E
Rout = %.3E
nR = %d
nTheta = %d
nPhi = %d
npart = %d
nTemp = %d
nsize = %df

-------------------------

Bins:

Size bins: 
%s

RadialBins
%s

PhiBins
%s

ThetaBins
%s

RadialBins2
%s

ThetaBins2
%s

PhiBins2
%s
   
'''%(BetaArrayFileName,OpticalDepthOn,ConstantParticleRadius,LaunchMethod,ConeDirection,cone_opening_angle,particle_density,
     DensityScalingFactor,odm,n,no,ns,nps,OutburstTimestep,OutburstScalingFactor,NoEjectionTimesteps,subrate_proportional_factor,subrate,TailLength/R_star,k_B,mgas,Tgas,Ngas_surface,mp,rp,p,dt,ap,es,vorb,planet_pos,N_es,LaunchSpeed,InitialParticleRadius,ParticleRemovalRadius,
     log1overe,amin,amax,SimpleLightCurveBeamRadius,SimpleLightCurveNumberOfSteps,SimpleLightCurveScatteringScalingFactor,SimpleLightCurveHGparameter,Rin,Rout,nR,nTheta,nPhi,npart,nTemp,nsize,
     Rin2,Rout2,nR2,nTheta2,nPhi2,npart2,nTemp2,nsize2,sizebins,RadialBins,PhiBins,ThetaBins,RadialBins2,PhiBins2,ThetaBins2)    

    InputSummaryFile = open(os.path.join(OutputDirectory,'InputSummaryFile.txt'),'w')
    InputSummaryFile.write(towrite)
    InputSummaryFile.close()
    
    return None 
    
def WriteParticleListToDisk():
    
    '''
    write the particle list as a compressed pkl file 
    
    To load:
    f = gzip.open('testPickleFile.pklz','rb')
    myNewObject = pickle.load(f)
    f.close()

    
    '''    
    particleListFile = gzip.open(os.path.join(OutputDirectory,'particle_list.pkl.gz'),'wb')
    pickle.dump(particle_list,particleListFile)
    particleListFile.close()
    
def FindRandomPointInCone(reference_direction,cone_opening_angle):
    
        
    '''
    Generate Random points on the sphere until it finds one that is within
    the specified angle from a reference direction to launch particles 
    randomly within a cone 
    
    Cone opening angle in degrees
    '''
    
    angular_difference = 180
            
    while (angular_difference > cone_opening_angle/2):
        
                
        TrialDirection = random_point_on_sphere()
        #print 'trial direction', TrialDirection
        angular_difference = FindAngleBetweenVectors(TrialDirection,reference_direction)
        #print 'angular_difference', angular_difference
        
    return TrialDirection
    
def RandomDirectionInRing():
    
    '''
    For launching from a ring around the terminator.
    '''
    
    RandomAngle = random.uniform(0,2*np.pi) # radians 
    
    xz = PointsOnACircle(1,RandomAngle,originx=0,originy=0)
    
    ## For a ring around the terminator, y=fixed (zero for a direction)
    return np.array([xz[0],0,xz[1]]) 
    
def IdentifyParticlesFromSmallRegionOnPlot(particle_list,xlims,ylims):
    
    particle_indices_in_region = []
    
    for particleindex in range(len(particle_list)):
        x,y,z = particle_list[particleindex].position_history[-1]
        
        if ((xlims[0]<=x<=xlims[1])&(ylims[0]<=y<=ylims[1])):
            particle_indices_in_region.append(particleindex)
            
    return particle_indices_in_region
    
def RandomPointInDisk(r):
    
    '''
    random points on a disk
    '''
    
    ToReturn = False 
    
    while ToReturn==False:
    
        x = random.uniform(-r,r)
        y = random.uniform(-r,r)
        
        if (x**2+y**2) < (r**2):
            
            ToReturn = True 
            
        if ToReturn == True:        
            return (x,y)
        
def PlotRandomDiskPoints(r=1,N=10):
    '''
    plot N random disk points 
    '''
    counter = 0
    
    xyvects = np.zeros((N,2))
    
    while counter < N:
        randomxy = RandomPointInDisk(r)
        xyvects[counter,:] = randomxy
        
        plt.plot(randomxy[0],randomxy[1],'o')
        counter += 1 
        
    xyvects[0,:] = (0.0,0.0)
    
    np.save('randomxydisk.npy',xyvects)
    
    
def AddParticlesForPlanet(particle_list):
    
    '''
    add 100 particles within the planet's radius 
    with huge radii so that they absorb all the flux 
    mimicing the planet blocking all the light that hits it.
    
    Add 100 particles with the positions given in randomxydisk.npy
    '''
    
    diskpositions = np.load('randomxydisk.npy')
    
       
    for i in range(NparticlesForPlanet):
        
        position = np.array([diskpositions[i,0]*rp,-ap,diskpositions[i,1]*rp])
        
        planetparticle = Particle('planet',position,np.array([0,0,0]),rp,timestep)
                
        particle_list.append(planetparticle)
        
def RemoveAddedParticlesForPlanet(particle_list):
    
    '''
    after the particles that were added to mimic the planet have 
    contributed to the optical depth calculation, remove them all before 
    they get moved, sublimated etc etc..
    '''
    
    particle_list = particle_list[:-NparticlesForPlanet]
        
    return particle_list
    
def CalculateSublimationRateForFixedTailLength(tail_length,initial_radius,final_radius,planet_orbital_period,planet_semi_major_axis):
    
    '''
    For a given tail length, initial size and final size, calculate the required 
    sublimation rate for it to sublimate at the correct length.
    
    Approximating the angular velocity using 4*pi*beta/(planet period)  (Rik’s thesis pg 96)
    which is only valid for a non-sublimating particle (because beta will change as it sublimates)
    but it will hopefully be good enough.
    
    For beta, it takes the average beta over the range of radii from initial 
    to final.
    '''
    
    radii_range = np.linspace(initial_radius,final_radius,10000)

    avg_beta = np.mean(MultiModeBeta(radii_range,BetaMode))
    
    dust_omega_syn = 4*np.pi*avg_beta/planet_orbital_period 
    
    TailAngularLength_rads = 2*np.pi*tail_length/(2*np.pi*planet_semi_major_axis) ## 2pi*fraction of circle that tail takes up assuming circular orbits 
    
    TimeToSublimate = TailAngularLength_rads/dust_omega_syn
    
    RequiredRate = (final_radius - initial_radius)/TimeToSublimate
    
    return RequiredRate
    
def SetAllParticleZtoZero(particle_list):
    
    '''
    Needed for the simple light curve maker to ensure that the beam in the plane
    will catch all the particles 
    '''
    
    for i in range(len(particle_list)):
        
        particle_position = particle_list[i].position
        x = particle_position[0]
        y = particle_position[1]
        z = particle_position[2]
        
        particle_list[i].update_position(np.array([x,y,0.0]))
        
        return particle_list
        
        
    
def MakeSimpleLightCurve(particle_list,ScatteringScalingFactor,HG_g_parameter,ns):
    
    '''
    Make a fast and simple light curve 
    based on the number of absorbing particles in a rotating cylinder
    with a Henyey-Greenstein function for scattering 
    ''' 
    
    particle_list = SetAllParticleZtoZero(particle_list)  ## Set all the particles' z positions to zero so they will all be caught by the beam which is a cylinder that lies in the plane of z = 0 
    
    tphase,tflux = LoadTimsBinnedLightCruve('20161125_KIC1255b_II_192741Z')
    #ttflux = (tflux/100)+1

    delta_theta = 360./ns ## The angle to rotate the radial vector by each timestep in degrees 
    
    
    initial_vect_direction = np.array([0,1,0]) ## Towards the top of the screen in the top down view     
       
    absorption_light_curve = np.ones((ns))
    
    scattering_light_curve = np.ones((ns))
    
    azimuthal_mass = np.ones((ns))
    
    azimuthal_number_of_particle = np.ones((ns))
    
    for timestep in range(ns):
        
        
        
        if timestep%10 == 0: print 'timestep %d of %d'%(timestep,ns)
        
        #print 'making lightcurve timestep %d of %d' % (timestep,ns*no)
        
        line_direction = RotateVector(initial_vect_direction,timestep*delta_theta)
        
        if timestep in np.arange(int(ns/2),int(0.75*ns),1):
            SnapshotTopDownBeamDirection(particle_list,timestep,line_direction)
            
    #    print 'angle', timestep*delta_theta
    #    print 'line direction',line_direction            
           
        for particle in particle_list:                    
           
            position = particle.position
            
            #light_curve[timestep] += CalculateHG_Contribution(position,line_direction,delta_theta,HG_g_parameter)*ScatteringScalingFactor
            scattering_light_curve[timestep] += CalculateHG_Contribution(position,line_direction,delta_theta,HG_g_parameter)*ScatteringScalingFactor
            
            #### See if it's on the correct side of the plane that line_direction is the normal of by taking the dot product between the normal vector and the point of interest 
            #### will be positive if it's on the side in the direction of the normal to the plane (line_direction) and negative if on the other side
            
            if np.dot(position,line_direction) > 0:                
            
                distance_to_line_of_sight = FindDistanceFromPointToLine(position,RotateVector(initial_vect_direction,timestep*delta_theta))
                
                if distance_to_line_of_sight <= SimpleLightCurveBeamRadius:  
                               
                    absorption_light_curve[timestep] *= np.exp(-np.pi*(particle.radius**2)*odm)
                    
                    azimuthal_mass[timestep] += (4.0/3.0)*np.pi*(particle.radius**3.0)*particle_density
                    
                    azimuthal_number_of_particle[timestep] += 1 
                    
                    
    
    percent_abs_lc = (absorption_light_curve-1.0)*100   
        
    shifited_sct_lc = scattering_light_curve - min(scattering_light_curve) ## Using the minimum instead of 1 
    ##because the minimum is slightly greater than 1 because there is always some scattering in all directions.  
    ##For scaling convienience, want to set it to zero.  Could just add that minimum back to it   
    
    scaled_shifited_sct_lc = shifited_sct_lc*ScatteringScalingFactor
    
    AbsorptionScalingFactor = np.min(tflux)/np.min(percent_abs_lc)    
    
    scaled_absorption_light_curve = percent_abs_lc*AbsorptionScalingFactor
    
    light_curve = scaled_shifited_sct_lc + scaled_absorption_light_curve
    
    #xplot = -0.5+np.arange(ns)*dt/p  ## xplot is in units of phase 
    xplot = np.linspace(0,1,len(absorption_light_curve))-0.5  ## For an arbitrary number of steps in the light curve 
    
    plt.figure(figsize=figuresize)
    plt.step(xplot,scaled_absorption_light_curve,'b',label='absorption',where='mid')
    plt.title('absorption (scaling factor = %.3E'%(AbsorptionScalingFactor))
    plt.xlabel('phase')
    plt.ylabel('(flux-1)*100 (%)')
    plt.xlim(-0.2,0.2)
    plt.savefig(os.path.join(LightCurvesDir,'absorption.png')) 
    np.save(os.path.join(LightCurvesDir,'absorption.npy'),absorption_light_curve)
    
    plt.figure(figsize=figuresize)
    plt.step(xplot,scaled_shifited_sct_lc,'r',label='scattering',where='mid')
    plt.xlim(-0.2,0.2)
    
    plt.title('scattering (Scaling factor = %.3E, HG parameter = %.3E' % (ScatteringScalingFactor,HG_g_parameter))
    plt.savefig(os.path.join(LightCurvesDir,'scattering.png')) 
    np.save(os.path.join(LightCurvesDir,'scattering.npy'),scaled_shifited_sct_lc)
    
             
    plt.figure(figsize=figuresize)
    plt.title('Model components N=%d (odm taken as 1)'%(n)) 
           
    plt.step(xplot,scaled_absorption_light_curve,'b--',label='absorption',where='mid')    
    
    plt.step(xplot,scaled_shifited_sct_lc,'r--',label='scattering',where='mid')
    
    plt.step(xplot,light_curve,'k',label='sum',where='mid')
    
    plt.step(tphase,tflux,'g',label='Kepler LC',where='mid')
    
    plt.plot([min(xplot),max(xplot)],[0,0],'k:')
    
    plt.xlabel('phase')
    
    plt.xlim(-0.2,0.2)
    
    #plt.xlim((1.8,2.3))
    #plt.xlim((0.8,1.3))
    
    plt.xlabel('phase')
    plt.ylabel('(flux-1)*100 (%)')
    
    plt.legend(loc=4)
    plt.savefig(os.path.join(LightCurvesDir,'model_components.png')) 
    
    plt.figure(figsize=figuresize)
    plt.step(xplot,scaled_absorption_light_curve,'b',label='absorption',where='mid')
    plt.step(tphase,tflux,'k',label='Kepler LC',where='mid')
    plt.xlim(-0.2,0.2)
    plt.legend(loc=4) 
    plt.xlabel('phase')
    plt.ylabel('(flux-1)*100 (%)')
    
    plt.figure(figsize=figuresize)
    plt.step(xplot,scaled_shifited_sct_lc,'r',label='scattering',where='mid')
    plt.step(tphase,tflux,'k',label='Kepler LC',where='mid')
    plt.xlim(-0.2,0.2)
    plt.legend(loc=4)
    plt.xlabel('phase')
    plt.ylabel('(flux-1)*100 (%)')
    
    plt.figure(figsize=figuresize)
    plt.step(xplot,light_curve,'b',label='combined LC',where='mid')
    plt.step(tphase,tflux,'k',label='Kepler LC',where='mid')
    plt.xlim(-0.2,0.2)
    plt.legend(loc=4)
    plt.xlabel('phase')
    plt.ylabel('(flux-1)*100 (%)')
    plt.savefig(os.path.join(LightCurvesDir,'model_KeplerLC_overplot.png')) 
    np.save(os.path.join(LightCurvesDir,'combined_model_lightcurve.npy'),light_curve)
        
    plt.figure(figsize=figuresize)
    plt.step(xplot,azimuthal_mass,where='mid')
    plt.title('mass as a function of azimuthal angle')    
    plt.xlabel('azimth or orbital phase (degrees)')
    plt.xlim(-0.2,0.2)
    
    plt.savefig(os.path.join(LightCurvesDir,'azimutal_mass.png')) 
    np.save(os.path.join(LightCurvesDir,'azimuthal_mass.npy'),azimuthal_mass)
    
    plt.figure(figsize=figuresize)
    plt.step(xplot,azimuthal_number_of_particle,where='mid')
    plt.title('number of particles as a function of azimuthal angle')    
    plt.xlabel('azimth or orbital phase (degrees)')
    plt.xlim(-0.5,0.5)
    
    plt.savefig(os.path.join(LightCurvesDir,'azimutal_NumberOfParticles.png')) 
    np.save(os.path.join(LightCurvesDir,'azimuthal_NumberOfParticles.npy'),azimuthal_mass)
    
    
def LoadTimsBinnedLightCruve(folder):
    

    f = pyfits.open(os.path.join('TimsWork',folder,'pdcsap_binned_nbin192_d0.8_1.0.fits'))
    tbdata = f[1].data  # assume the first extension is a table
    #print tbdata[:1]  
    
    phase = tbdata['PHASE']
    flux = tbdata['FLUX']
    
    ### Change the flux -1 (%)
    
    flux = (flux-1.0)*100
    
    
    #sap_flux = tbdata['SAP_FLUX_FIT']
    #pdcsap_flux = tbdata['PDCSAP_FLUX_FIT']
    
    fmodel = pyfits.open(os.path.join('TimsWork',folder,'pdcsap_binned_nbin192_d0.8_1.0_model.fits'))
    modeltbdata = fmodel[1].data
    modelphase = modeltbdata['MODELPHASE']
    modelflux = modeltbdata['MODELFLUX']
    
    modelflux = (modelflux-1.0)*100
    
    return phase,flux
    
def acceleration_for_odeint(u,t,beta,omega,r0):    
    
    '''
    u = [rx,ry,rz,vx,vy,vz] (ie the [positions, velocities]).  
    Have to give the parameters individually since it can't take vectors.
    
    Should be called like:
    sol = odeint(acceleration, u0, t, args=(beta, omega,r0))
    where r0 is the planet position.
    
    and the output of sol (solution) is a 2D array of u vectors ([rx,ry,rz,vx,vy,vz])
    where every row is the position and velocity at that time step (so has number of rows = len(t))
    
    where t is the time vector 
    
    This function when called by odeint in the way above, returns the positions at every time step.
    
    vprime = dv/dt = d**2 r /dt**2.
    
    See how it takes as input u which has the current positions, and velocities, [rx,ry,rz,vx,vy,vz]
    then returns its derivate  [rx',ry',rz',vx',vy',vz'] = [vx (input),vy (input),vz (input),vx'= acceleration x, acceleration y, acceleration z']
    where the accelerations are calculated here     
    '''
    
    
    r = np.array([u[0],u[1],u[2]])    
    v = np.array([u[3],u[4],u[5]])
    
    #vprime = -(G*M_star/((np.linalg.norm(r))**3))-np.cross(2*omega,np.cross(omega,r)) #- (2*G*mp/((np.linalg.norm(r-r0))**3))*(r-r0)
    
    #planet_acceleration = np.array([0,0,0]) #AccelerationDueToPlanet(r)    
    planet_acceleration = AccelerationDueToPlanet(r)
    #planet_acceleration = np.array([0,0,0])
    
    vprime = -(G*ms*(1.0-beta)/np.linalg.norm(r)**3)*r - np.cross(2*omega,v) - np.cross(omega,(np.cross(omega,r))) + planet_acceleration
    #print 'vv',velocity_vector
    
    dudt = [u[3],u[4],u[5],vprime[0],vprime[1],vprime[2]]  ## du / dt where u is the input vector 
    
    return dudt 
    
def call_odeint(particle_object,dt,omega):
    
    """
    Gets all relevant info from the particle and 
    calls the function VerletHope2 with these inputs to actually calculate 
    the acceleration and new veleocity.
    
    Then takes these new values and updates the particle object 
    """
    
#    print 
#    print '##########################'
#    
    r = particle_object.position
    v = particle_object.velocity 
    beta = particle_object.beta 
#    
#    print 'Initial parameters'
#    print 'r', r
#    print 'v', v
#    print 'beta',beta
#    print 
#    beta = 0.15
    #particle_radius = particle_object.radius
        
#    ### updating the acceleration due to planet history with the position value before it is moved so that it is the acceleration due to the planet at the position that led to this nett acceleration.  
#    particle_object.update_planet_acceleration_history(AccelerationDueToPlanet(r))
    
    u0 = np.array([r[0],r[1],r[2],v[0],v[1],v[2]])
    
    
    sol = odeint(acceleration_for_odeint, u0, np.array([0,dt]), args=(beta, omega,planet_pos))  ## The solution in the from [rx,ry,rz,vx,vy,vz]
    
    r_new = np.array([sol[1,0],sol[1,1],sol[1,2]])
    v_new = np.array([sol[1,3],sol[1,4],sol[1,5]]) 
    
#    print 'New parameters'
#    print 'r', r_new
#    print 'v', v_new
#
#    print 
    
    #particle_object.update_position(r_new)   
    #particle_object.update_velocity(v_new)
    
    ### add one to the particle's age 
    particle_object.update_age()       
    particle_object.update_timestep_history(timestep)
    
    #last_uvect = sol[1,:]
    
    #return last_uvect
    
    #return 0,0
    return r_new,v_new
    
def SnapshotTopDownBeamDirection(particle_list,timestep,LineDirection):       
       
    xlist = []
    ylist = []           
    radius_list = []           
   
    plt.figure(figsize=((9,9)))        
    
    for particle in particle_list:                         
                            
        x = particle.position[0]
        y = particle.position[1]
        radius = particle.radius
        
        xlist.append(x)
        ylist.append(y)
        radius_list.append(radius)   
    
    plt.scatter(xlist,ylist,edgecolors='none',s=5,c=np.array(radius_list)*1E6,cmap=mpl.cm.spectral,norm=mpl.colors.LogNorm(vmin=ParticleRemovalRadius*1E6,vmax=InitialParticleRadius*1E6)) ## Radius in um 
    cbar = plt.colorbar(ticks = LogLocator(subs=range(10)),fraction=0.014, pad=0.02) # Draw log spaced tick marks on the colour bar 
    cbar.ax.set_ylabel(r'particle radius [$\mu m$]')
    
    
    circle1=plt.Circle((0,0),radius=R_star,color='y')  # 0.7*R_sun as a rough estimate from wikipedia 
    plt.gcf().gca().add_artist(circle1)
    
    circle2=plt.Circle((0,-ap),radius=rp,color='k')  # 1.15 R_earth 1sigma limit from Brogi 2012
    plt.gcf().gca().add_artist(circle2)
    
    plt.ylabel('co-rotating y coordinate (m)')
    plt.xlabel('co-rotating x coordinate (m)')
    plt.title('top down view')     
    
    
    plt.xlim(BeamDirectionTopDownxlim)
    plt.ylim(BeamDirectionTopDownylim)   
    
    #plt.gca().set_aspect('equal', adjustable='box')  
    plt.gca().set_aspect('equal')  
    
    v0 = [0,0,0]
    v1 = LineDirection*2*ap
    
    perpvect = Simple2DPerpVect(v1)
    ## normalise perpvect 
    
    perpvect = perpvect/np.linalg.norm(perpvect)
    
    v2 = v1+perpvect*R_star
    v20 = v0+perpvect*R_star
    
    v3 = v1-perpvect*R_star
    v30 = v0-perpvect*R_star
    
    
    #plt.plot([v0[0],v1[0]],[v0[1],v1[1]],'b')    
    
    plt.plot([v20[0],v2[0]],[v20[1],v2[1]],'r')    
    plt.plot([v30[0],v3[0]],[v30[1],v3[1]],'r')  
    
    
    plt.savefig(os.path.join(BeamDirections,'BeamDirections_timestep%d'%(timestep))) 
    
    plt.close('all')
    
def Simple2DPerpVect(v):
    
    '''
    A simple function to return a 2D vector perpendicular to the given vector.
    Returns a 3D vector but the z direction is fixed to zero 
    
    a = [ax,ay] so aperp = [-ay,ax]
    '''
    
    return np.array([-v[1],v[0],0])
    
def Add_n_ParticleCopies(n,ParticleObjectToCopy):
    
    pl = []    
    
    counter = 0
    
    while counter <= n:
        
        pl.append(ParticleObjectToCopy)
        
        counter += 1 
        
    return pl 
        
        
        
        
    
def SetExponentialTail(N,b):
    
    '''
    Angles are defined as (in radians)
    
           pi/2
       pi        0
          -pi/2
          
    N is the number of particles in the tail 
    '''
    
    pl = []
    number_at_each_position = [] 
    
    expx = np.linspace(-0.2,0.2,100) # in phase 
    
    y = np.exp(-expx*b)
    
    Nscalingfactor = N/np.sum(y)
    
    Ny = np.round(Nscalingfactor*y) ## total won't be exactly N because of the rounding 
    
    
    
    sph_coord = -(expx*2*np.pi + np.pi/2) # theta 
    
    for i in range(len(sph_coord)):
        
        sph_pos = np.array([ap,sph_coord[i],np.pi/2])
        
        cart_pos = sph2cart(sph_pos)       
    
        pl = pl+Add_n_ParticleCopies(Ny[i],Particle(0,cart_pos,np.array([0,0,0]),1E-6,0))
        number_at_each_position = number_at_each_position + Add_n_ParticleCopies(Ny[i],Ny[i])
        
    Allpos = GetAllxyzCoords(pl)
        
    plt.scatter(Allpos[:,0],Allpos[:,1],edgecolors='none',s=1,c=number_at_each_position,cmap=mpl.cm.spectral) ## Radius in um 
    cbar = plt.colorbar(ticks = LogLocator(subs=range(10)))
    plt.title('Number density particles at each position with a force logarithmic distribution')
    plt.savefig(os.path.join(OutputDirectory,'forced_exponential_number_density'))
    
    return pl 
    
def FindMaximumTailHeight(particle_list):
    
    xyz = GetAllxyzCoords(particle_list)
    
    z = xyz[:,2]
    
    spread = np.max(z) - np.min(z)
    
    towrite = '''spread (m), planet mass (kg), planet radius (m), planet density (kgm^(-1)), particle launch velocity (m/s), planet escape velocity, particle launch speed (escape velocities)
    %.5E, %.5E, %.5E, %.5E, %.5E, %.5E, %.5E'''%(spread,mp,rp,PlanetDensity,LaunchSpeed,es,N_es)
    
    TailHeightDataFile = open(os.path.join(OutputDirectory,'TailHeightDataFile.txt'),'w')
    TailHeightDataFile.write(towrite)
    TailHeightDataFile.close()
    
    return spread
    
    
def CalculateGridCellVolumes(nR,nTheta,nPhi,RadialBins,ThetaBins,PhiBins):
    
    volumes = np.zeros((nPhi,nTheta,nR))
    radial_extent = np.zeros((nPhi,nTheta,nR))
    
#    for i in range(nPhi):
#        for j in range(nTheta):
#            for k in range(nR):
    tstart = time.time()
    for i in range(10):
        for j in range(10):
            for k in range(10):
                    
                r1 = RadialBins[k]
                r2 = RadialBins[k+1]
                
                radial_extent[i,j,k] = r2-r1 
                
                t1 = PhiBins[i]*np.pi/180.0 ## NOTE!! theta (t) for this function is phi in rest of code
                t2 = PhiBins[i+1]*np.pi/180.0 ## NOTE!! theta (t) for this function is phi in rest of code
                
                p1 = ThetaBins[j]*np.pi/180.0 ## NOTE!! phi (p) for this function is theta in the rest of the code 
                p2 = ThetaBins[j+1]*np.pi/180.0 ## NOTE!! phi (p) for this function is theta in the rest of the code 
                
                volumes[i,j,k] = tplquad(diff_volume, r1, r2, lambda r:   t1, lambda r:   t2, lambda r,t: p1, lambda r,t: p2)[0] #in cubic metres 
                
    np.save('VolumeGrid_nPhi%d_nTheta%d_nR%d.npy'%(nPhi,nTheta,nR),volumes)    
    np.save('RadialExtents_nPhi%d_nTheta%d_nR%d.npy'%(nPhi,nTheta,nR),radial_extent)    
    
    endtime=time.time()
    print 'time taken',endtime-tstart
        
                       
def CheckIfParticlesAreOutsideGrid():
    
    '''
    check the maximum and minimum radial distances and check that they are not 
    outside the grid
    '''
    
    cartpos = GetAllxyzCoords(particle_list)
    sphericalPos = cart2sph(cartpos)
    
    RadialDistances = sphericalPos[:,0]
    
    if False in ((Rin<=RadialDistances)&(Rout>=RadialDistances)):
        
        outside_grid = np.where((((Rin<=RadialDistances)&(Rout>=RadialDistances)))==False)
        
        np.save(os.path.join(OutputDirectory,'ParticleIndicesOutsideGrid'),outside_grid)
        

    
def ApproximateTransitDepth(particle_list,nR,nTheta,nPhi,RadialBins,ThetaBins,PhiBins,TailHeight,SimulatedTailLength):
    
    VectToPlanet = np.array([0,-1,0])
    RotatedVectToSeeMostOfTail = RotateVector(VectToPlanet,13.0)
    
    p = GetAllxyzCoords(particle_list) ## positions in m in Cartesiain coordinates 
      
    #### This code determines which bin each particle is in 
    
    sph = cart2sph(p) #positions in specical coordinate     
    xr = np.digitize(sph[:,0],RadialBins)
    
    zeroindices = np.where((xr==0))
    toohighindices = np.where(xr==len(RadialBins))        
    
    xr[zeroindices] = 1                   #put values that are below the lower bound into the first bin
    xr[toohighindices] = len(RadialBins)-1   #put values that are above the upper bound into the last bin     
    
    xa = np.digitize(sph[:,1]*180.0/np.pi+180.0,PhiBins) #the position around the star (azimuth, 0 to 360 degrees)
       
    xe = np.digitize(sph[:,2]*180.0/np.pi,ThetaBins) ## The equator in the spherical coords is 0 degrees but it is theta = 90 in MCMAX ThetaBins
    ####
    
    odlist = []
    IncludedCellsList = []
    
    for particleindex in range(len(particle_list)):  

        particle = particle_list[particleindex]                  
    
        position = particle.position      
        
        
        if np.dot(position,RotatedVectToSeeMostOfTail) > 0:       
            
            #### See if it's on the correct side of the plane that line_direction is the normal of by taking the dot product between the normal vector and the point of interest 
            #### will be positive if it's on the side in the direction of the normal to the plane (line_direction) and negative if on the other side
        
            distance_to_line_of_sight = FindDistanceFromPointToLine(position,RotatedVectToSeeMostOfTail)
            
            if distance_to_line_of_sight <= SimpleLightCurveBeamRadius:  
                
                indextuple = (xa[particleindex],xe[particleindex],xr[particleindex])
                
                if indextuple not in IncludedCellsList:
                    
                    odlist.append(particle.od_1)
                    IncludedCellsList.append(indextuple)
                    
    Transmissions = np.exp(-np.array(odlist))
    
    Absorptions = 1.0 - Transmissions
    
    AverageAbsorption = np.mean(Absorptions)
    
    TransitDepth = GeometricAreaTransitDepth(TailHeight,SimulatedTailLength)*AverageAbsorption  ## with tail height in m and tail length in R_star  

    np.savetxt(os.path.join(ApproximateTransitDepthDir,'TransitDepth.txt'),np.array([TransitDepth]))
    np.savetxt(os.path.join(ApproximateTransitDepthDir,'AverageAbsorption.txt'),np.array([AverageAbsorption]))
    np.savetxt(os.path.join(ApproximateTransitDepthDir,'TailHeight.txt'),np.array([TailHeight]))
    np.savetxt(os.path.join(ApproximateTransitDepthDir,'TailLength.txt'),np.array([SimulatedTailLength]))
    
    
    np.save(os.path.join(ApproximateTransitDepthDir,'OpticalDepths.npy'),np.array(odlist))
    np.save(os.path.join(ApproximateTransitDepthDir,'Transmissions.npy'),Transmissions)
    np.save(os.path.join(ApproximateTransitDepthDir,'Absorptions.npy'),Absorptions)
    
    GridCellIndexFile = open(os.path.join(ApproximateTransitDepthDir,'GridCellIndexFile.txt'), 'w')
    
    for item in IncludedCellsList:
        GridCellIndexFile.write("%s\n" % str(item))
    
    GridCellIndexFile.close()


    return TransitDepth #odlist, IncludedCellsList
    
def task(particle_number):
    
    #print 'particle number %d' % (particle_number)
    #print type(particle_list[particle_number])
    #print particle_list[particle_number]
    #print 

    #r_new,v_new = call_odeint(particle_list[particle_number],dt,omega)
    r_new,v_new = call_odeint(particle_list[particle_number],dt,omega)

    
    #r_new,v_new = 0,0#call_odeint(particle_list[particle_number],dt,omega)

    #r_new, v_new = 0,0
    
    particle_list[particle_number].update_position(r_new)
    particle_list[particle_number].update_velocity(v_new)
    
def taskods(particle_number):
    
    particle_list2,od4 = opdepPlankMeanOpacity(nsize,nR,nTheta,nPhi,sizebins,RadialBins,ThetaBins,PhiBins,particle_list)      

    particle_list[:] = particle_list2
    
    
def tasksublimate(particle_number):
    
    particle_list2,subrate = sublimate_particles(particle_list,dt,subrate_proportional_factor)

    particle_list[:] = particle_list2
    
def taskRemoveSubbed(particle_number):
    
    particle_list2 = remove_sublimated_particle(particle_list,ParticleRemovalRadius)  ## If this is turned off, it crashes in the optical depth calculation with a NAN 

    particle_list[:] = particle_list2

def taskRemoveInPlanet(particle_number):
    
    particle_list2 = remove_particles_inside_planet(particle_list)

    particle_list[:] = particle_list2
    
def taskApplyBeta(particle_number):    

    particle_list2 = ApplyBeta(particle_list)
    
    particle_list[:] = particle_list2

testlist = []
    
def taskAll(particle_number):
#    
#    r_new,v_new = call_odeint(particle_list[particle_number],dt,omega)
#    
#    #r_new,v_new = 0,0#call_odeint(particle_list[particle_number],dt,omega)
#
#    
#    particle_list[particle_number].update_position(r_new)
#    particle_list[particle_number].update_velocity(v_new)
    
    particle_list2,od4 = opdepPlankMeanOpacity(nsize,nR,nTheta,nPhi,sizebins,RadialBins,ThetaBins,PhiBins,particle_list)      

    particle_list[:] = particle_list2
    
    particle_list2,subrate = sublimate_particles(particle_list,dt,subrate_proportional_factor)

    particle_list[:] = particle_list2
    
    particle_list2 = remove_sublimated_particle(particle_list,ParticleRemovalRadius)  ## If this is turned off, it crashes in the optical depth calculation with a NAN 

    particle_list[:] = particle_list2
    
#    particle_list2 = remove_particles_inside_planet(particle_list)
#
#    particle_list[:] = particle_list2
    
    particle_list2 = ApplyBeta(particle_list)
    
    particle_list[:] = particle_list2

def MoveParticleForMultiprocess(Particle):
    
#    print 'The particle in move function:'
#    print Particle
    
    r_new,v_new = call_odeint(Particle,dt,omega)
    Particle.update_position(r_new)
    Particle.update_velocity(v_new)
    
    return Particle 
    
    

        


def AllFunctions(particle_list):
    
    r_new,v_new = call_odeint(particle_list[particle_number],dt,omega)
    
    
    particle_list,od4 = opdepPlankMeanOpacity(nsize,nR,nTheta,nPhi,sizebins,RadialBins,ThetaBins,PhiBins,particle_list)      

    
    particle_list,subrate = sublimate_particles(particle_list,dt,subrate_proportional_factor)

    particle_list = remove_sublimated_particle(particle_list,ParticleRemovalRadius)  ## If this is turned off, it crashes in the optical depth calculation with a NAN 

    
    particle_list = remove_particles_inside_planet(particle_list)
    
    particle_list = ApplyBeta(particle_list)
    
    return particle_list 
    
def AllIndividualParticleUpdatesForMultiProcess(Particle):
    
    ### Move the particle 
    r_new,v_new = call_odeint(Particle,dt,omega)
    Particle.update_position(r_new)
    Particle.update_velocity(v_new)
    
    ### ** Sublimate the particle  **
    
    if not OpticalDepthOn:           
        
        subrate = subrate_proportional_factor
        
    if OpticalDepthOn:
        
        subrate = subrate_proportional_factor*(Particle.transmittance)  
        
    new_radius = Particle.radius+(dt*subrate)
    
    if new_radius < 0:
        
        new_radius = 0
    
    Particle.update_radius(new_radius)
    
    ### ** Update the Particle's new beta value **
    
    radius = Particle.radius 
     
    if OpticalDepthOn: 
        beta = MultiModeBeta(radius,BetaMode)*(Particle.transmittance)      
         
    if not OpticalDepthOn:
        beta = MultiModeBeta(radius,BetaMode)
   
    Particle.update_beta(beta)
    
    ### *** Replace particles to remove with None ***
    
    particle_pos = Particle.position
    particle_age = Particle.age
       
    distance_to_planet = np.linalg.norm(particle_pos - planet_pos)
    ###  remove particles inside planet#              remove particles smaller than the removal radius     
    if (((distance_to_planet <= rp)&(particle_age > 0)) or Particle.radius < ParticleRemovalRadius):
            pass
    
    else:
        return Particle 
    
    
    
         


   
    
# n       total number of particles to launch
# no      number of orbits
# odm     optical density multiplier
# outpos  output position
# outvel  output velocity
# outdia  output diameter

##******************
### Define whether or not optical depth should be considered when calculating particle dynamics 

OpticalDepthOn = False        


## Define if the particles will have a constant radius or take them from a distribution 

ConstantParticleRadius = True    

IncludeGasDragAcceleration = False 

#####BetaArrayFileName = 'CorundumBetaDigitizedFromRiksPaper.txt.csv'
#####BetaArray = np.a=np.loadtxt(BetaArrayFileName,delimiter=',',usecols=(0,1),skiprows=6)
#####BetaFunc = interp1d(BetaArray[:,0]*1E-6,BetaArray[:,1],bounds_error=False,fill_value=(BetaArray[0,1],BetaArray[-1,1])) #converting from loaded um to m

BetaMode = 'LoadedBeta' 
#BetaMode = 'lognormal'

if BetaMode == 'LoadedBeta':

    BetaArrayFileName = os.path.join('matfiles','corundum.dat')
    RikCorundumMatarray = np.genfromtxt(BetaArrayFileName,max_rows=2)    
    BetaFunc = interp1d(RikCorundumMatarray[0,:]*1E-6,RikCorundumMatarray[1,:],bounds_error=False,fill_value=(RikCorundumMatarray[1,0],RikCorundumMatarray[1,-1]),kind='cubic') #converting from loaded um to m
    
if BetaMode == 'lognormal':
    BetaArrayFileName = 'using log normal beta'





OutputLocation = 'output'

LaunchMethod = 'uniform_sphere'
#LaunchMethod = 'DaySide'
#LaunchMethod = 'cone'
ConeDirection = np.array([-1,0,0]) #only used if LaunchDirection == True
cone_opening_angle = 20 ## degrees 

previous_run = np.loadtxt(os.path.join(OutputLocation,'run_counter.txt'))
current_run = previous_run+1

OutputDirectory = os.path.join(OutputLocation,'run%d'%(int(current_run)))
if not os.path.exists(OutputDirectory):
    os.makedirs(OutputDirectory)
    
### optional output directories     
    
#LightCurvesDir= os.path.join(OutputDirectory,'SimpleLightCurves')
#if not os.path.exists(LightCurvesDir):
#    os.makedirs(LightCurvesDir)
#    
#BeamDirections = os.path.join(OutputDirectory,'SimpleLightCurves','BeamDirections')
#if not os.path.exists(BeamDirections):
#    os.makedirs(BeamDirections)
#    
#ApproximateTransitDepthDir = os.path.join(OutputDirectory,'ApproximateTransitDepth')
#if not os.path.exists(ApproximateTransitDepthDir):
#    os.makedirs(ApproximateTransitDepthDir)
#    
SaveStatesDir = os.path.join(OutputDirectory,'SaveStates')
if not os.path.exists(SaveStatesDir):
    os.makedirs(SaveStatesDir)
    
np.savetxt(os.path.join(OutputLocation,'run_counter.txt'),np.array([current_run]))

ParticlesToTrack = []
TrackedParticlesIndicesInFinalParticleList = []

# ****************

LoadParticleList = False   
particle_list_filename = os.path.join(OutputLocation,'run199','particle_list.pkl.gz')

NparticlesForPlanet = 100

R_sun_m =  6.957E8 #in m from WolframAlpha 

Msun = 1.989E30 # mass of Sun in kg from Wikipedia 

Rearth = 6.3674447E6 #m from wolfram alpha  

#### Stellar parameters of KIC12557548 from Huber+ 2014 #####

T_star = 4550  ## effective temperature in K +- 133K 
log_g = 4.630  ## +- 0.400

FeH = -0.173  ##+- 0.300

R_star = 0.660*R_sun_m ## Rsun +- 0.060

M_star = 0.666*Msun ## Msun +- 0.067

#####################################
sigma = 5.670367E-8 # W m**(-2) K **(-4)

L_star = 4*np.pi*((R_star)**2)*sigma*T_star**4

particle_density = 4020 # kg/m^3 or 4.02 g/cm^3.  Density of corundum 



## optical depth multiplier, needed to properly calibrate the optical deptsh 
odm = 3


no = 1 ## Number of orbits 

n = int(1000) ### *****  Total number of particles to be ejected from planet in simulation 




particle_list = []
removed_particle_list = []
removed_particles_from_inside_planet = []

#NumberOfParticlesRemovedFromPlanet = 0 


ns = 100  ## Number of timesteps per orbit 
SaveStateEveryThisManyTimeSteps = 1


OutburstTimestep = range(0,100,1)
OutburstScalingFactor = 1

NoEjectionTimesteps = [] 


subrate = 1 ### A dummy value since the multiprocess can't return it as well


#NoEjectionTimesteps = []#range(11,500)#range(50,500,1)




nps=n/ns/no # particles to add per integration step
Basenps = n/ns/no

##G=6.673e-11 # m^3/kg/s^2; # gravitational constant   (Christoph's value)

G = 6.67408E-11 # m^3/kg/s^2; # gravitational constant
ms=1.4e30 # mass of star in kg (0.7 solar masses)

PlanetDensity = 5427 # kg/m^3 (Density of Mercury)



rp = 1.3E5  ## Gives a good tail height with density 



#mp=5.0e23 # mass of planet in kg (1.5 Mercury masses)

mp = (4.0/3.0)*np.pi*(rp**3)*PlanetDensity
##mp = 5.000E+23

### Optional check on planet mass 
#MoonMass = 7.34767309E22 #kg ## From Google 
#
#if mp > 2*MoonMass:
#    raise Exception ### The upper limit of mass from radiative modelling is 2*Moon masses from models (physical limit from spectroscopy is 3 M*Jups) so this will stop unrealistic runs 

p=56467.584 # orbital period in seconds

c = 299792458 # speed of light in m/s

dt=p/ns; # time step (orbital period / number of steps per period)

ap=(((p/2.0/np.pi)**2)*G*ms)**(1.0/3.0) # planet orbit radius in m
es=np.sqrt(2.0*G*mp/rp) # planetary escape velocity in m/s
vorb=2.*np.pi*ap/p # orbital velocity in m/s

planet_pos = np.array([0,-ap, 0]) #in the co-rotating coordinates 

### 1780 (with the correct DensityScaling Factor)
N_es = 3.0
LaunchSpeed = N_es*es

InitialParticleRadius = 1E-6


ParticleRemovalRadius = 1E-9 # in m 
## This parameter isn't used but it is written somewhere in the output file so it will crash if it's not defined 
TailLength = 0 ## Fixing the subrate  

Kappa_ref_index = .1400E-01 ##at 600nm from the .lnk file, this is the Kappa from n = real +i*Kappa 



subrate_proportional_factor = -(InitialParticleRadius - ParticleRemovalRadius)/(ns*dt)   ## so that without optical depth affects, it will sublimate after exactly 1 orbit 


### Only needed if a distribution of particle sizes is used 

log1overe = 0.25E-6


####### Parameters the gas accleration 

k_B = 1.38064852E-23# m2 kg s-2 K-1 from Google 
mgas = 1.65676E-26 # mass of an oxygen atom. Assuming Al2O3 is making O (not molecular)
Tgas=2000
#Ngas_surface = 1.4E16 #From the Rosetta paper 
Ngas_surface = 1.4E20

###############  Parameters to set the grid for getting the density of particles that is used by MCMAX3D 

Rin = 1.7E9 # m, from the top down plots, needs to be in AU for MCMAX3D (0.006AU) 
Rout = 2.9E9 #m 
#nR = 80
#nTheta = 40 ## Need enough theta bins so that the bin size is about the size of the angular height of the disk
#nPhi = 720#120 #120 # 360
#npart = 1 
#nTemp = 1 



nTheta = 40 ## Need enough theta bins so that the bin size is about the size of the angular height of the disk
nPhi = 720
npart = 1 
nTemp = 1 

nR = 100
dR = 3E6
RBinsBeforePlanet = 2
RBinsAfterPlanet = nR-RBinsBeforePlanet

RadialBins = np.arange(ap-RBinsBeforePlanet*dR,ap+(RBinsAfterPlanet+1)*dR,dR)
#
#Rin = np.min(RadialBins) # m, from the top down plots, needs to be in AU for MCMAX3D (0.006AU) 
#Rout = np.max(RadialBins)
#

#nRScalingFactor = 5*20   ## 3 is the max that a density .fits file can be produced. 2 is the max that MCMax3D can use (with nPhi = 720 and nTheta = 40)
#nRScalingFactor = 5*10  ## crashed it 

#nRScalingFactor = 5
#
#nR = 200*nRScalingFactor
#
#dR = 1.5E6/nRScalingFactor
#RBinsBeforePlanet = 10*nRScalingFactor
#RBinsAfterPlanet = nR-RBinsBeforePlanet




RadialBins = np.arange(ap-RBinsBeforePlanet*dR,ap+(RBinsAfterPlanet+1)*dR,dR) ### These are the grid boundaries so are of dimensions nR+1 (ie the edges of the bins so need 11 points for 10 bins)   

Rin = np.min(RadialBins) # m, from the top down plots, needs to be in AU for MCMAX3D (0.006AU) 
Rout = np.max(RadialBins)

##Rout = 2.9E9  

nsize = 15 #number of size bins


amin = ParticleRemovalRadius
amax = InitialParticleRadius

sizebins = np.logspace(np.log10(amin),np.log10(amax),nsize+1)
#PlankMeanOpacity = [355.834402680797,
#                    355.883761307134,
#                    356.034639097711,
#                    356.763277235272,
#                    358.998724375076,
#                    366.871015738598,
#                    408.775268855157,
#                    530.432964250027,
#                    843.523118107390,
#                    843.523118107390,
#                    1435.19055609497,]

Extinction = np.genfromtxt('CorundumExtinctionFunctionOfParticleSize.dat')  #in cm**2/g
sizebins_central_size = (sizebins[:-1]+sizebins[1:])/2.0

ExtinctionFunc = interp1d(Extinction[:,0]*1E-6,Extinction[:,1]/10.0,bounds_error=False,kind='cubic') #converting radius from loaded um to m and converting loaded cm**2/g to m**2/kg, by dividing by 10

ExtinctionInEachSizeBin = ExtinctionFunc(sizebins_central_size)

Rin2 = rp
Rout2 = 5*Rearth 
nR2 = 50
nTheta2 = nTheta #10# 180
nPhi2 = nPhi#60 #360
npart2 = 1 
nTemp2 = 1 
nsize2 = 1 

## the bins, defined as the values in the array being the edges of the bins 

### These are the bin boundaries so there need to be n+1 bins 

PhiBins = np.linspace(0,360,nPhi+1)

CentralThetaBinBoundaries = [89.0,91]

CentralThetaBins = np.linspace(CentralThetaBinBoundaries[0],CentralThetaBinBoundaries[1],nTheta-1)  ### These are the grid boundaries so are of dimensions n+1 
deltaCentralThetaBins = CentralThetaBins[1] - CentralThetaBins[0]

ThetaBins = np.zeros((nTheta+1))
ThetaBins[0] = 0.0
ThetaBins[-1] = 180.0

ThetaBins[1:-1] = CentralThetaBins

#### This second grid is never used but it will try to write it to an output file so just leave it here for ease 
RadialBins2 = np.linspace(Rin2,Rout2,nR2+1)
ThetaBins2 = ThetaBins#np.linspace(0,180,nTheta2+1)
PhiBins2 = PhiBins#np.linspace(0,360,nPhi2+1)

#### *** A quick work around to not have to recaclulate the volume every time 

VolumeGridDirectory = 'NormalVolumeGridsComputedOnTheFly'

if not os.path.exists(VolumeGridDirectory):
    os.makedirs(VolumeGridDirectory)
    
VolumeGridFilename = 'VolumeGrid_nPhi%d_nTheta%d_CentralThetaLims%f_%f_deltaThetaCentralBins%f_nR%d_Rlims%.5E_%0.5E.npy' % (nPhi,nTheta,CentralThetaBins[0],CentralThetaBins[-1],deltaCentralThetaBins,nR,RadialBins[0],RadialBins[-1])
RadialExtentGridIFilename = 'RadialExtentGrid_nPhi%d_nTheta%d_CentralThetaLims%f_%f_deltaThetaCentralBins%f_nR%d_Rlims%.5E_%0.5E.npy' % (nPhi,nTheta,CentralThetaBins[0],CentralThetaBins[-1],deltaCentralThetaBins,nR,RadialBins[0],RadialBins[-1])


if os.path.exists('%s/%s' % (VolumeGridDirectory,VolumeGridFilename)):
    
    print('Loading volume and radial extent grid')
    
    volumes = np.load('%s/%s' % (VolumeGridDirectory,VolumeGridFilename))
    radial_extent = np.load('%s/%s' % (VolumeGridDirectory,RadialExtentGridIFilename))
    
else:

    volumes = np.zeros((nPhi,nTheta,nR))
    radial_extent = np.zeros((nPhi,nTheta,nR))
    
#### Now need to save thee updated volume_grid and radial_extent grid 

#### *** End a quick work around to not have to recaclulate the volume every time 



DensityScalingFactor = 8e22


SimpleLightCurveBeamRadius = R_star#6.42040E+07/2.0 #(see google doc for hos this relates to Tim's bin size) 
SimpleLightCurveNumberOfSteps = 192 ## as being 1/(the width (resolution) of Tim's light curve bins) of 0.005209 in phase 
SimpleLightCurveScatteringScalingFactor = 1.5e-2
SimpleLightCurveHGparameter = 0.5




BeamDirectionTopDownxlim = (-2*ap,+2*ap)
BeamDirectionTopDownylim = (-2*ap,+2*ap)


WriteInputSummary()


ParticleCounter = 0 


if not LoadParticleList:   
    
    print 'generating particle list'

    # Set starting values for position and velocity    
    
    omega = np.array([0,0,2*math.pi/p]) # CUK: omega = 2PI/period

    for timestep in range(ns*no):
        
        if timestep in OutburstTimestep:
            nps = Basenps*OutburstScalingFactor
        if timestep in NoEjectionTimesteps:
            nps = 0
    #for timestep in [0,1]:#range(ns*no):

        
        #### Add new particles then move all particles in the particle_list 
        
        print 'Doing timestep %d of %d' % (timestep,ns*no)
            
            
        ## Add particles 
        for i in range(nps):     
            
                              
            
            if LaunchMethod=='fixed_direction':                
                LaunchDirection = ConeDirection            
            
            if LaunchMethod =='cone':                
                LaunchDirection = FindRandomPointInCone(ConeDirection,cone_opening_angle)

            if LaunchMethod=='uniform_sphere':                
                
                #LaunchDirection = ReproducableRandomPointonSphere(n,ParticleCounter) # defines the launch direction
                LaunchDirection = random_point_on_sphere()
            
            if LaunchMethod=='ring':
                
                ConeDirectionOnRing = RandomDirectionInRing()
                
                LaunchDirection = FindRandomPointInCone(ConeDirectionOnRing,cone_opening_angle)                
                
            if LaunchMethod == 'DaySide':
                LaunchDirection = ReproducableRandomPointonDaySide(n,ParticleCounter)
                
                          
            r = planet_pos + LaunchDirection*rp #launch from 1 planetary radius 
            ## r = planet_pos + np.array([0,0,1])*rp #launch from 1 planetary radius directly above
            v =  LaunchSpeed*LaunchDirection
            
            ###v =  es*np.array([0,0,1]) ## To launch directly up from the planet 
            
            if not ConstantParticleRadius:
                radius =  np.clip(np.random.exponential()*log1overe,amin,amax) #in m so 1/e occurs at size of 1E-6, clipped to be between 1E-10 and 1 m
            if ConstantParticleRadius:
                radius =  InitialParticleRadius
            
            particle_list.append(Particle(ParticleCounter,r,v,radius,timestep))              
            
            ParticleCounter += 1 
            
            
        nps = Basenps
        
        particle_list,od4 = opdepPlanckMeanOpacityComputeVolumesOnFly(nsize,nR,nTheta,nPhi,sizebins,RadialBins,ThetaBins,PhiBins,particle_list)      
            
        if __name__ == '__main__':
            
            #pool = multiprocessing.Pool(processes=16)
            pool = multiprocessing.Pool()
            
            particle_list = pool.map(AllIndividualParticleUpdatesForMultiProcess, particle_list)           
            

            pool.close()
            pool.join()
        
        ### If a particle is removed the particle list is returned with None in its old place 
        ### to remove the 'nones' from the list, an order N operation 
           
            
        particle_list = [x for x in particle_list if x != None]    
        
        if (timestep%SaveStateEveryThisManyTimeSteps)==0:
            SaveState(particle_list,timestep)
            
            
            
#    

#    plot_last_top_down(particle_list)
#    plot_last_top_down_zoomin(particle_list)
#    plot_last_side_view(particle_list)
#    plot_last_side_view(particle_list,2)
    PlotOpDepVsRadialDist(particle_list)   ## Actually doedsn't plot anything 

    
    SaveFinaTimestepArraysForPlotting(particle_list)
    TailHeight = FindMaximumTailHeight(particle_list)

    print 'calculating mass densities on given grid'
    ## For one grid:
    mass_densities1,comp, SimulatedTailLength = FindParticleDensitiesShiftedOriginComp(nR,nTheta,nPhi,RadialBins,ThetaBins,PhiBins,particle_list,DensityScalingFactor,sizebins,nTemp,nsize,npart,diff_volume)

    print 'writing density file'
    filename1 = os.path.join(OutputDirectory,'densfile1_%d_particles_nR%d_nT%d_nP%d.fits.gz'%(len(particle_list),nR, nTheta,nPhi))

    #### Write the updated VolumeGrid and radial_extent arrays so that they can be loaded again if needed 
    np.save('%s/%s' % (VolumeGridDirectory,VolumeGridFilename),volumes)
    np.save('%s/%s' % (VolumeGridDirectory,RadialExtentGridIFilename),radial_extent)
        

    WriteDensityFile(Rin,Rout,nR,nTheta,nPhi,npart,nsize,nTemp,RadialBins,ThetaBins,PhiBins,mass_densities1,comp,filename1)

     

    CheckIfParticlesAreOutsideGrid()
    
endtime = time.time()

totaltime_s = endtime-starttime

print 'total time for n = %d particles was %f hours'%(n,totaltime_s/3600.0)
np.savetxt(os.path.join(OutputDirectory,'TotalTime_hours.txt'),np.array([totaltime_s/3600.0]))

