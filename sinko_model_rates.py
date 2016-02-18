# -*- coding: utf-8 -*-
#Created on Oct 14, 2015
#@author: Inom Mirzaev


"""
    Model rates and parameters used for generation of existence regions of the
    Sinko-Streifer population model (see Sinko, J. W. and Streifer, W. (1967),
    Ecology, 48(6):910-918. ) 
"""


from __future__ import division
from functools import partial


import numpy as np



"""
    Number of CPUs used for computation of existence and stability regions.
    For faster computation number of CPUs should be set to the number of cores available on
    your machine.
"""

ncpus = 2


# Minimum and maximum floc sizes
x0 = 0
x1 = 1


#Renewal rate    
def renewal(x , a):
    #Should return a vector
    return a*(x + 1)
 
 
#Growth rate   
def growth(x ,  b ):
    #Should return a vector
    return b*(x+1)
    
        
#Removal rate    
def removal( x , c ):
    
     #Should return a vector
     return c * ( x + 1 - x )

     

"""
    Initialization of intervals for generation of existence and stability regions.
    For smoother plots more discretization points should be given.
"""

#Interval initialization renewal function

# a minimum 
amin = 0

# a maximum
amax = 1

#Number of discretization  points in a interval
apart = 40

# b minimum
bmin = 0

# b maximum
bmax = 1

#Number of discretization points in b interval
bpart = 40

# c minimum
cmin = 0

# c maximum
cmax = 1

# Number of discretization points in c interval
cpart = 40



a_ = np.linspace( amax , amin , apart , endpoint=False ).tolist()
a_.sort()

b_ = np.linspace( bmax , bmin , bpart , endpoint=False ).tolist()
b_.sort()


c_ = np.linspace( cmax , cmin , cpart , endpoint=False ).tolist()
c_.sort()


grid_x, grid_y, grid_z = np.meshgrid(a_, b_, c_, indexing='ij')

 
#Initializes uniform partition of (x0, x1) and approximate operator F_n
def sinko_initialization(N , a, b, c, x1=x1 , x0=x0):
    
    #delta x
    dx = ( x1 - x0 ) / N
    
    #Uniform partition into smaller frames
    nu = x0 + np.arange(N+1) * dx
    
    
    #Initialize renewal function with parameter a
    renew = partial( renewal , a=a)
    
    #Initialize growth function with paramter b
    grow = partial( growth , b=b )
    
    #Initialize removal function with paramter c        
    rem = partial( removal , c=c )    
    
    #Removal operator
    Remov_mat =  - np.diag( rem( nu[range( 1 , N + 1 )] ) )
    
    #Growth operator
    Growth_mat = np.zeros( ( N , N ) )
    
    #Renewal operator
    Renew_mat = np.zeros( ( N , N ) )
    Renew_mat[0,:] = Renew_mat[0,:] + renew( nu[range( 1 , N+1 ) ] )
    
    for jj in range(N-1):
        Growth_mat[jj,jj] = -grow( nu[jj+1] ) / dx
        Growth_mat[jj+1,jj] = grow( nu[jj+1] ) / dx
        
    Growth_mat[N-1, N-1] = -grow( nu[N] ) / dx

    return ( Renew_mat , Growth_mat , Remov_mat , nu , N , dx)
   

