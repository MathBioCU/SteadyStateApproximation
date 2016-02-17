# -*- coding: utf-8 -*-
"""
Created on Oct 14, 2015

@author: Inom Mirzaev

"""


from __future__ import division

import numpy as np
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
    

agg=1e0

#Aggregation rate
def ka(x,y, x1=x1 , agg=agg):
    
    out = agg*( x ** ( 1/3 ) + y ** ( 1/3 ) ) **3      

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
         
    #Fragmentation rate

    def kf(x, x0=x0, c=c):

        return 1 * x


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



fnames = ['data_2016_01_19_00_09.npy','data_2016_01_19_16_36.npy' , 'data_2016_01_20_21_23.npy' ]

output=np.load( fnames[-1] )
output = output[ np.nonzero( output[: , 3 ] ) ]

eigs = np.zeros( ( len(output) , 2 ) )


for nn in range(len(output)):

    An, Ain, Aout, nu, N, dx = initialization( 100 , output[nn , 0] , output[nn , 1] , output[nn, 2] )
    
    ddd=np.max(  np.linalg.eig( An )[0]  )     
    
    eigs[nn, 0] = np.real( ddd )
    eigs[nn, 1] = np.imag( ddd )

end = time.time()

np.save('eigenvalues1', eigs)

print "Elapsed time " + str( round( (end - start) / 60 , 1)  ) + " minutes"



