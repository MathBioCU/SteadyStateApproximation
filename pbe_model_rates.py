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

# Post-fragmentation density distribution
def gam( y , x ):
    
    out = 6*y * ( x - y )  / (x**3)
    
    if type(x) == np.ndarray or type(y) == np.ndarray:        
        out[y>x] = 0

    return out 
    

#Aggregation rate
def aggregation( x , y ):
    
    out = ( x ** ( 1/3 ) + y ** ( 1/3 ) ) **3      

    return out


#Renewal rate    
def renewal(x , a):
    
    return a*(x + 1)
 
 
#Growth rate   
def growth(x ,  b ):

    return b*(x+1)
    
        
#Removal rate    
def removal( x , c ):

     return c * x

     
#Fragmentation rate
def fragmentation( x ):

    return  x


amin = 0
amax = 1
apart = 10

bmin = 0
bmax = 1
bpart = 10

cmin = 0
cmax = 1
cpart = 10


a_ = np.linspace( amax , amin , apart , endpoint=False ).tolist()
a_.sort()

b_ = np.linspace( bmax , bmin , bpart , endpoint=False ).tolist()
b_.sort()


c_ = np.linspace( cmax , cmin , cpart , endpoint=False ).tolist()
c_.sort()


grid_x, grid_y, grid_z = np.meshgrid(a_, b_, c_, indexing='ij')

    
#Initializes uniform partition of (x0, x1) and approximate operator F_n
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
    
    renew = partial( renewal , a=a)
    
    grow = partial( growth , b=b )
            
    rem = partial( removal , c=c )


    for mm in range( N ):
    
        for nn in range( N ):
            
            if mm>nn:
            
                Ain[mm,nn] = 0.5 * dx * aggregation( nu[mm] , nu[nn+1] )
            
            if mm + nn < N-1 :
                
                Aout[mm, nn] = dx * aggregation( nu[mm+1] , nu[nn+1] )
                    
            if nn > mm :
            
                Fin[mm, nn] = dx * gam( nu[mm+1], nu[nn+1] ) * fragmentation( nu[nn+1] )


    Fout = 0.5 * fragmentation( nu[range( 1 , N + 1 ) ] ) + rem( nu[range( 1 , N + 1 )] )

    #Growth operator
    Gn=np.zeros( ( N , N ) )

    for jj in range(N-1):
        Gn[jj,jj] = -grow( nu[jj+1] ) / dx
        Gn[jj+1,jj] = grow( nu[jj+1] ) / dx

    Gn[0,:] = Gn[0,:] + renew( nu[range( 1 , N+1 ) ] )
    Gn[N-1, N-1] = -grow( nu[N] ) / dx
    
    #Growth - Fragmentation out + Fragmentation in
    An = Gn - np.diag( Fout ) + Fin

    return (An , Ain , Aout , nu , N , dx)



#Approximate operator for the right hand side of the evolution equation
def approximate_IG( y ,  An , Aout , Ain):
    
    a = np.zeros_like(y)

    a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    

    out = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , a).T - ( Aout.T * y ).T + An , y ) 
      
    return out

#Exact Jacobian of the RHS 
def jacobian_IG(y, An , Aout , Ain):

    a = np.zeros_like(y)

    a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ] 

    out = An - ( Aout.T * y ).T - np.diag( np.dot(Aout , y) ) + 2*Ain * lin.toeplitz( np.zeros_like(y) , a).T

    return out    
    
   

