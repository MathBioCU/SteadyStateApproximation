# -*- coding: utf-8 -*-
"""
Created on  Oct 14 13:18:38 2015

@author: Inom Mirzaev

This code generates convergence plots for the steady states of the famous Sinko-Streifer model. 
Infinitesimal generator G is apporixmated by an n-by-n matrix G_n. Consequently,
steady states of G is approximated by zeros of the matrix G_n. 

"""

from __future__ import division
from model_rates import *


import multiprocessing as mp
import numpy as np
import time

start = time.time()


  
#Given a singular matrix this function returns nullspace of that matrix    
def null(a, rtol=1e-5):
    
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    
    return v[rank:].T.copy() 
 

x = np.ravel( grid_x )
y = np.ravel( grid_y )
z = np.ravel( grid_z )

myarray = np.array([x, y, z]).T


Renewal_mat , Growth_mat , Removal_mat , nu , N , dx  = sinko_initialization( 100 , 1 , 1,  1 )


def region_plots( nn , Renewal_mat=Renewal_mat,
                       Growth_mat=Growth_mat,
                       Removal_mat=Removal_mat, 
                       myarray = myarray):
    
    An =  myarray[nn , 0]*Renewal_mat + myarray[nn , 1]*Growth_mat  + myarray[nn, 2]*Removal_mat 
    
    pos_sol = 0
    eigs = 0
    if np.sum( np.abs( null(An) ) ) > 0:
        pos_sol = 1
        eigs = np.max(  np.real ( np.linalg.eig(  An )[0] ) )  

       
    return ( myarray[nn , 0] , myarray[nn , 1] , myarray[nn, 2] , pos_sol , eigs )
    


if __name__ == '__main__':
    pool = mp.Pool( processes = mp.cpu_count() )
    ey_nana = range( len( myarray) )
    result = pool.map( region_plots , ey_nana )
    
    output = np.asarray(result)
    
    output = output[ np.nonzero( output[: , 3 ] ) ]
    
    
    fname = 'sinko_data'    
    np.save(fname , output )
    

end = time.time()


print "Elapsed time " + str( round( (end - start) / 60 , 1)  ) + " minutes"


