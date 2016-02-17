# -*- coding: utf-8 -*-
"""
Created on  Oct 14 13:18:38 2015

@author: Inom Mirzaev

This code generates convergence plots for the steady states of the famous Sinko-Streifer model. 
Infinitesimal generator G is apporixmated by an n-by-n matrix G_n. Consequently,
steady states of G is approximated by zeros of the matrix G_n. 

"""

from __future__ import division
from multiprocessing import Pool

import numpy as np
import time

start = time.time()

# Minimum and maximum floc sizes
x0 = 0
x1 = 1
N  = 100

    
# Initializes uniform partition of (x0, x1) and approximate operator F_n
    
#delta x
dx = ( x1 - x0 ) / N

#Uniform partition into smaller frames
nu = x0 + np.arange(N+1) * dx
  
#Fragmentation out
Fout = np.zeros( N )

def q(x ):
    
    return x + 1
    
def g(x):

    return x+1
    
        
   #Removal rate    
def rem(x):

     return x + 1 -x
     

#Removal operator
Removal =  - np.diag( rem( nu[range( 1 , N + 1 )] ) )

#Growth operator
Growth = np.zeros( ( N , N ) )

#Renewal operator
Renewal = np.zeros( ( N , N ) )
Renewal[0,:] = Renewal[0,:] + q( nu[range( 1 , N+1 ) ] )

for jj in range(N-1):
    Growth[jj,jj] = -g( nu[jj+1] ) / dx
    Growth[jj+1,jj] = g( nu[jj+1] ) / dx
    
Growth[N-1, N-1] = -g( nu[N] ) / dx

   
#Given a singular matrix this function returns nullspace of that matrix    
def null(a, rtol=1e-5):
    
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    
    return v[rank:].T.copy() 
 


part = 100

x_ = np.linspace(1, 0,  part, endpoint=False).tolist()
x_.sort()
y_ = np.linspace(1, 0, part, endpoint=False).tolist()
y_.sort()
z_ = np.linspace(1, 0, part, endpoint=False).tolist()
z_.sort()

x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')


x = np.ravel(x)
y = np.ravel(y)
z = np.ravel(z)

myarray = np.array([x, y, z]).T



def region_plots(nn, Growth=Growth, Renewal=Renewal, Removal=Removal, myarray = myarray):
    
    An =  myarray[nn , 0]*Renewal + myarray[nn , 1]*Growth  + myarray[nn, 2]*Removal 
    
    pos_sol = 0
    eigs = 0
    if np.sum( np.abs( null(An) ) ) > 0:
        pos_sol = 1
        eigs = np.max(  np.real ( np.linalg.eig(  An )[0] ) )  

       
    return ( myarray[nn , 0] , myarray[nn , 1] , myarray[nn, 2] , pos_sol , eigs )
    



if __name__ == '__main__':
    pool = Pool(processes =12 )
    ey_nana = range( len( myarray) )
    result = pool.map( region_plots , ey_nana )
    
    output = np.asarray(result)
    
    output = output[ np.nonzero( output[: , 3 ] ) ]
    
    
    fname = 'data_' + time.strftime("%Y_%m_%d_%H_%M", time.gmtime())    
    np.save(fname , output )
    


end = time.time()


print "Elapsed time " + str( round( (end - start) / 60 , 1)  ) + " minutes"


