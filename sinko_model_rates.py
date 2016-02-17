# -*- coding: utf-8 -*-
"""
Created on Oct 14, 2015

@author: Inom Mirzaev

"""


from __future__ import division
from functools import partial

import scipy.linalg as lin
import numpy as np


ncpus = 2


# Minimum and maximum floc sizes
x0 = 0
x1 = 1


#Renewal rate    
def renewal(x , a):
    
    return a*(x + 1)
 
 
#Growth rate   
def growth(x ,  b ):

    return b*(x+1)
    
        
#Removal rate    
def removal( x , c ):

     return c * ( x + 1 - x )

     
amin = 0
amax = 1
apart = 40

bmin = 0
bmax = 1
bpart = 40

cmin = 0
cmax = 1
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
    
    
    renew = partial( renewal , a=a)
    
    grow = partial( growth , b=b )
            
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
   

