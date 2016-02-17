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
from scipy.integrate import odeint , quad
from matplotlib import gridspec
from scipy.spatial.distance import cdist


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin
import numdifftools as ndt
import time


start = time.time()

# Minimum and maximum floc sizes
x0 = 0
x1 = 1

# Post-fragmentation density distribution
def gam( y , x , x0 = x0 ):
    
    out = 6*y * ( x - y )  / (x**3)
    
    if type(x) == np.ndarray or type(y) == np.ndarray:        
        out[y>x] = 0

    return out 


#Fragmentation rate

def kf(x, x0=x0):

    return 1 * x
    

#Aggregation rate
def ka(x,y, x1=x1):
    
    out = ( x ** ( 1/3 ) + y ** ( 1/3 ) ) **3      

    return out
    
# Initializes uniform partition of (x0, x1) and approximate operator F_n
def initialization(N , a, b, c, x1=x1 , x0=x0):
    
    #delta x
    dx = ( x1 - x0 ) / N
    
    #Uniform partition into smaller frames
    nu = x0 + np.arange(N+1) * dx
    
    #Aggregation in
    Ain = np.zeros( ( N , N ) )
    
    #Aggregation out
    Aout = np.zeros( ( N , N ) )
    
    #Fragmentation in
    Fin = np.zeros( ( N , N ) )
    
    #Fragmentation out
    Fout = np.zeros( N )
    
    def q(x , x0=x0 , a=a):
        
        return a*(x + 1)
        
    def g(x, x0=x0, x1=x1, b=b):

        return b*(x+1)
        
            
   #Removal rate    
    def rem(x, x0=x0, c=c):

         return c * x
         


    for mm in range( N ):
    
        for nn in range( N ):
            
            if mm>nn:
            
                Ain[mm,nn] = 0.5 * dx * ka( nu[mm] , nu[nn+1] )
            
            if mm + nn < N-1 :
                
                Aout[mm, nn] = dx * ka( nu[mm+1] , nu[nn+1] )
                    
            if nn > mm :
            
                Fin[mm, nn] = dx * gam( nu[mm+1], nu[nn+1] ) * kf( nu[nn+1] )


    Fout = 0.5 * kf( nu[range( 1 , N + 1 ) ] ) + rem( nu[range( 1 , N + 1 )] )

    #Growth operator
    Gn=np.zeros( ( N , N ) )

    for jj in range(N-1):
        Gn[jj,jj] = -g( nu[jj+1] ) / dx
        Gn[jj+1,jj] = g( nu[jj+1] ) / dx
        
    Gn[0,:] = Gn[0,:] + q( nu[range( 1 , N+1 ) ] )
    Gn[N-1, N-1] = -g( nu[N] ) / dx
    
    #Growth - Fragmentation out + Fragmentation in
    An = Gn - np.diag( Fout ) + Fin

    return (An , Ain , Aout , nu , N , dx)



An, Ain, Aout, nu, N, dx = initialization( 100 , 5 , 1 , 1 )




#The right hand side of the evolution equation
def root_finding( y ,  An=An , Aout=Aout , Ain=Ain):
    
    a = np.zeros_like(y)

    a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    

    out = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , a).T - ( Aout.T * y ).T + An , y ) 
      
    return out

#Exact Jacobian of the RHS 
def exact_jacobian(y, An=An, Aout=Aout, Ain=Ain):

    a = np.zeros_like(y)

    a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ] 

    out = An - ( Aout.T * y ).T - np.diag( np.dot(Aout , y) ) + 2*Ain * lin.toeplitz( np.zeros_like(y) , a).T

    return out    




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
times = np.linspace(0 , 2 , 1000, endpoint=True)

noise = np.random.normal( 0 , 1 , len(sol) )
#Initial condition
y1  =  sol + 1*noise
yout1 = odeint( myode, y1 , times )

y2  =   15 + 10*np.sin( 3*np.linspace( 0 , np.pi , N ) ) 
yout2 = odeint( myode, y2 , times )

y3  =   sol + 5*np.sin( 6*np.linspace( 0 , np.pi , N ) ) 
yout3 = odeint( myode, y3 , times )

#y4  =   30 + 30*np.sin( 5*np.linspace( 0 , np.pi , N ) ) 
noise = np.random.normal( 0 , 1 , len(sol) )
y4  =  20 + 8*noise
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
plt.savefig('stable_equilibrium_initial.png', dpi=400 , bbox_inches='tight' )


plt.figure(1)

plt.plot(  times , dx * yout1.sum(axis=1)  , linewidth=2 , color='r' )    
plt.plot(  times , dx * yout2.sum(axis=1)  , linewidth=2 , color='g' )
plt.plot(  times , dx * yout3.sum(axis=1)  , linewidth=2 , color='b' )
plt.plot(  times , dx * yout4.sum(axis=1)  , linewidth=2 , color='k' )
plt.xlabel( '$t$' , fontsize = 16 )
plt.ylabel( '$M_0(t)$' , fontsize = 16 )

plt.savefig('stable_equilibrium_number.png', dpi=400 , bbox_inches='tight' )


plt.figure(2)

plt.plot(  nu[range(N)] , yout1[-1,:]  , linewidth=2 , color='b' )    
plt.xlabel( '$x$' , fontsize = 16 )
plt.ylabel( '$u_*(x)$' , fontsize = 16 )
plt.savefig('stable_equilibrium.png', dpi=400 , bbox_inches='tight' )


plt.figure(3)

plt.plot( times , np.dot( yout1 , a) , linewidth=2 , color='r' )    
plt.plot( times , np.dot( yout2 , a) , linewidth=2 , color='g' )
plt.plot( times , np.dot( yout3 , a) , linewidth=2 , color='b' )
plt.plot( times , np.dot( yout4 , a) , linewidth=2 , color='k' )
plt.xlabel( '$t$' , fontsize = 16 )
plt.ylabel( '$M_1(t)$' , fontsize = 16 )


plt.savefig('stable_equilibrium_mass.png', dpi=400 , bbox_inches='tight' )
"""
aa = myode( yout1[-1,:] , 1 )



times = np.linspace(0 , 5 , 1000, endpoint=True)

a = ( nu[range(1,N+1)] ** 2 - nu[range(N)] ** 2 ) / 2

#Total mass is stored in column #1 and total number is stored in column #2  
totals = np.zeros(( 25, 2) )


for nn in range(25):
    
    y0 = 2**(nn -5)*np.random.normal( 0 , 1 , len(sol) )
    yout = odeint( myode, y0 , times  )
    totals[nn, 0] = np.sum( yout[-1, :] * a )
    totals[nn, 1] = np.sum( yout[-1, :] * dx )
    os.system('cls' if os.name == 'nt' else 'clear')



#Pairwise distance between totals are calculated 
max_distance = np.max( cdist(totals, totals ) )


if max_distance < 1e-5:
    print str( len(totals) ) + ' different initial conditions are given'
    print "For all given initial conditions solutions converge to the same steady state"
"""
    
end = time.time()

print "Elapsed time " + str( round( (end - start) , 1)  ) + " seconds"
    