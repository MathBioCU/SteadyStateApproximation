# -*- coding: utf-8 -*-
"""
Created on Oct 14, 2015

@author: Inom Mirzaev

This code approximates the stationary solutions of the microbial flocculation model
by the stationary solutions of the system of nonlinear ODEs. 
The code uses scipy.optimize.fsolve to find roots of nonlinear function. 
This function uses Powell's hybrid method. For faster convergence rate, we provided 
the solver with the exact Jacobian defined in the paper. For the purpose of 
illustration, the model parameters have been chosen such that they satisfy existence conditions 
derived in Mirzaev and Bortz (2015).

"""


from __future__ import division
from scipy.optimize import fsolve 
from scipy.integrate import odeint 
from scipy.spatial.distance import cdist
from pbe_model_rates import *


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin
import time , os


start = time.time()
a = 1
b = 0.5
c = 1
An, Ain, Aout, nu, N, dx = initialization( 100 , a ,  b , c )




root_finding  = partial( approximate_IG , An=An, Aout=Aout, Ain=Ain )    
exact_jacobian = partial( jacobian_IG , An=An, Aout=Aout, Ain=Ain)


seed = 10*np.arange(N)

sol = fsolve( root_finding ,  seed , fprime = exact_jacobian , xtol = 1e-8 , full_output=1 )

sol = sol[0]


# Ode simulations with arbitrary small initial condition
def myode(y , t, An=An, Aout=Aout, Ain=Ain):
    
    a = np.zeros_like(y)    
    a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    
    
    out = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , a).T  - ( Aout.T * y ).T + An , y )             
    
    return out    
 
#Time of the simulation
times = np.linspace(0 , 10 , 1000, endpoint=True)

noise = np.random.normal( 0 , 1 , len(sol) )
#Initial condition
y1  =  sol + 0.1*noise
yout1 = odeint( myode, y1 , times )

y2  =   2 + 0.4*np.sin( 3 * np.linspace( 0 , np.pi , N ) ) 
yout2 = odeint( myode, y2 , times )

y3  =   sol + 0.5*np.sin( 6*np.linspace( 0 , np.pi , N ) ) 
yout3 = odeint( myode, y3 , times )


noise = np.random.normal( 0 , 1 , len(sol) )
y4  =  2 + 0.4*noise
yout4 = odeint( myode, y4 , times )

plt.close('all')

#Total mass
a = ( nu[range(1,N+1)] ** 2 - nu[range(N)] ** 2 ) / 2

#Total number
b = dx * yout1.sum(axis=1)

plt.figure(0)
    
plt.plot( nu[ range(N)] ,  yout1[0,:] , linewidth=2 , color='r' )    
plt.plot( nu[ range(N)] ,  yout2[0,:] , linewidth=2 , color='g' )
plt.plot( nu[ range(N)] ,  yout3[0,:] , linewidth=2 , color='b' )
plt.plot( nu[ range(N)] ,  yout4[0,:] , linewidth=2 , color='k' )

plt.xlabel( '$x$' , fontsize = 16 )
plt.ylabel( '$u_0(x)$' , fontsize = 16 )
plt.savefig(os.path.join( 'images' , 'stable_equilibrium_initial.png' ), dpi=400 , bbox_inches='tight' )


plt.figure(1)

plt.plot(  times , dx * yout1.sum(axis=1)  , linewidth=2 , color='r' )    
plt.plot(  times , dx * yout2.sum(axis=1)  , linewidth=2 , color='g' )
plt.plot(  times , dx * yout3.sum(axis=1)  , linewidth=2 , color='b' )
plt.plot(  times , dx * yout4.sum(axis=1)  , linewidth=2 , color='k' )
plt.xlabel( '$t$' , fontsize = 16 )
plt.ylabel( '$M_0(t)$' , fontsize = 16 )

plt.savefig( os.path.join( 'images' , 'stable_equilibrium_number.png' ) , dpi=400 , bbox_inches='tight' )


plt.figure(2)

plt.plot(  nu[range(N)] , yout1[-1,:]  , linewidth=2 , color='b' )    
plt.xlabel( '$x$' , fontsize = 16 )
plt.ylabel( '$u_*(x)$' , fontsize = 16 )
plt.savefig( os.path.join( 'images' , 'stable_equilibrium.png' ) , dpi=400 , bbox_inches='tight' )


plt.figure(3)

plt.plot( times , np.dot( yout1 , a) , linewidth=2 , color='r' )    
plt.plot( times , np.dot( yout2 , a) , linewidth=2 , color='g' )
plt.plot( times , np.dot( yout3 , a) , linewidth=2 , color='b' )
plt.plot( times , np.dot( yout4 , a) , linewidth=2 , color='k' )

plt.xlabel( '$t$' , fontsize = 16 )
plt.ylabel( '$M_1(t)$' , fontsize = 16 )

plt.savefig( os.path.join( 'images' , 'stable_equilibrium_mass.png' ), dpi=400 , bbox_inches='tight' )

   
end = time.time()

print "Elapsed time " + str( round( (end - start) , 1)  ) + " seconds"
    