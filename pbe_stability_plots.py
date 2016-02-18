# -*- coding: utf-8 -*-
#Created on Feb 18, 2016
#@author: Inom Mirzaev

"""
    This program simulates the PBE for various initial conditions. 
    The model rates need to be specified in the 'pbe_model_rate.py' file.
"""


from __future__ import division
from scipy.optimize import fsolve 
from scipy.integrate import odeint 
from pbe_model_rates import *


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin
import time , os


start = time.time()

#Initialize the approximate operators 
An, Ain, Aout, nu, N, dx = initialization( 100 , a ,  b , c )

#Initialize the root_finding functions 
root_finding  = partial( approximate_IG , An=An, Aout=Aout, Ain=Ain )    
exact_jacobian = partial( jacobian_IG , An=An, Aout=Aout, Ain=Ain)


#Search the root of the system for 10 different initial seeds
for mm in range( 10 ):

    seed = 2**mm * np.ones(N)                    
    sol = fsolve( root_finding ,  seed , fprime = exact_jacobian , xtol = 1e-8 , full_output=1 )

    #Break the loop if a positive solution is found
    if sol[2]==1 and np.linalg.norm( sol[0] ) > 1 and np.all( sol[0] > 0 ):
        break


sol = sol[0]


# Ode simulations with arbitrary small initial condition
def myode(y , t, An=An, Aout=Aout, Ain=Ain):
    
    a = np.zeros_like(y)    
    a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    
    
    out = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , a).T  - ( Aout.T * y ).T + An , y )             
    
    return out    
 
#Time of the simulation
times = np.linspace(0 , 10 , 1000, endpoint=True)

#Initial condition 1 and ODE simulation

#Generate some normally distributed noise
noise = np.random.normal( 0 , 1 , len(sol) )
y1  =  sol + 0.1*noise
yout1 = odeint( myode, y1 , times )

#Initial condition 2 and ODE simulation
y2  =   2 + 0.4*np.sin( 3 * np.linspace( 0 , np.pi , N ) ) 
yout2 = odeint( myode, y2 , times )

#Initial condition 3 and ODE simulation
y3  =   sol + 0.5*np.sin( 6*np.linspace( 0 , np.pi , N ) ) 
yout3 = odeint( myode, y3 , times )

#Initial condition 4 and ODE simulation
noise = np.random.normal( 0 , 1 , len(sol) )
y4  =  2 + 0.4*noise
yout4 = odeint( myode, y4 , times )


plt.close('all')

"""
    Initial conditions
"""
plt.figure(0)
    
plt.plot( nu[ range(N)] ,  yout1[0,:] , linewidth=2 , color='r' )    
plt.plot( nu[ range(N)] ,  yout2[0,:] , linewidth=2 , color='g' )
plt.plot( nu[ range(N)] ,  yout3[0,:] , linewidth=2 , color='b' )
plt.plot( nu[ range(N)] ,  yout4[0,:] , linewidth=2 , color='k' )

plt.xlabel( '$x$' , fontsize = 16 )
plt.ylabel( '$u_0(x)$' , fontsize = 16 )
plt.savefig(os.path.join( 'images' , 'stable_equilibrium_initial.png' ) , dpi=400 , bbox_inches='tight' )


"""
    Evolution of total number of flocs
"""
plt.figure(1)

plt.plot(  times , dx * yout1.sum(axis=1)  , linewidth=2 , color='r' )    
plt.plot(  times , dx * yout2.sum(axis=1)  , linewidth=2 , color='g' )
plt.plot(  times , dx * yout3.sum(axis=1)  , linewidth=2 , color='b' )
plt.plot(  times , dx * yout4.sum(axis=1)  , linewidth=2 , color='k' )
plt.xlabel( '$t$' , fontsize = 16 )
plt.ylabel( '$M_0(t)$' , fontsize = 16 )

plt.savefig( os.path.join( 'images' , 'stable_equilibrium_number.png' ) , dpi=400 , bbox_inches='tight' )


"""
    Solution at final time
"""
plt.figure(2)

plt.plot(  nu[range(N)] , yout1[-1,:]  , linewidth=2 , color='b' )    
plt.xlabel( '$x$' , fontsize = 16 )
plt.ylabel( '$u_*(x)$' , fontsize = 16 )
plt.savefig( os.path.join( 'images' , 'stable_equilibrium.png' ) , dpi=400 , bbox_inches='tight' )


"""
    Evolution of total mass of flocs
"""
plt.figure(3)

#Total mass
aa = ( nu[range(1,N+1)] ** 2 - nu[range(N)] ** 2 ) / 2

plt.plot( times , np.dot( yout1 , aa) , linewidth=2 , color='r' )    
plt.plot( times , np.dot( yout2 , aa) , linewidth=2 , color='g' )
plt.plot( times , np.dot( yout3 , aa) , linewidth=2 , color='b' )
plt.plot( times , np.dot( yout4 , aa) , linewidth=2 , color='k' )

plt.xlabel( '$t$' , fontsize = 16 )
plt.ylabel( '$M_1(t)$' , fontsize = 16 )

plt.savefig( os.path.join( 'images' , 'stable_equilibrium_mass.png' ), dpi=400 , bbox_inches='tight' )

   
end = time.time()

print "Elapsed time " + str( round( (end - start) , 1)  ) + " seconds"
    