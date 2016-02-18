# -*- coding: utf-8 -*-
#Created on Feb 18, 2016
#@author: Inom Mirzaev

"""
    This code generates existence region for the steady states of the famous Sinko-Streifer model. 
    Infinitesimal generator G is apporixmated by an n-by-n matrix G_n. Consequently,
    steady states of G is approximated by zeros of the matrix G_n. 
    
    Computed regions are plotted in 3D and saved in 'images' folder.
    
    The model rates should be specified in the 'sinko_model_rates.py' file.
    
    The program has been written in parallel. Therefore, for the faster 
    computation, the parameter 'ncpus' in 'sinko_model_rates' should 
    be set to maximum number of cores available. 
"""

from __future__ import division
from sinko_model_rates import *


import multiprocessing as mp
import numpy as np
import time , os



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


#Initialize approximate matrice
Renewal_mat , Growth_mat , Removal_mat , nu , N , dx  = sinko_initialization( 100 , 1 , 1,  1 )


def region_plots( nn , Renewal_mat=Renewal_mat,
                       Growth_mat=Growth_mat,
                       Removal_mat=Removal_mat, 
                       myarray = myarray):
    
    An =  myarray[nn , 0]*Renewal_mat + myarray[nn , 1]*Growth_mat  + myarray[nn, 2]*Removal_mat 
    
    pos_sol = 0
    eigs = 0
    
    #if dimension of the nullspace is nonzero. Return positive steady state exists
    if np.sum( np.abs( null(An) ) ) > 0:
        pos_sol = 1
        eigs = np.max(  np.real ( np.linalg.eig(  An )[0] ) )  

       
    return ( myarray[nn , 0] , myarray[nn , 1] , myarray[nn, 2] , pos_sol , eigs )
    

if __name__ == '__main__':
    
    #Number of CPUs to be used
    pool = mp.Pool( processes = ncpus )
    ey_nana = range( len( myarray) )
    result = pool.map( region_plots , ey_nana )
    
    #The output is saved in the data_files folder    
    output = np.asarray(result)    
    output = output[ np.nonzero( output[: , 3 ] ) ]    
    fname = 'sinko_data'    
    np.save( os.path.join( 'data_files' , fname ) , output )
    

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


fname = 'sinko_data.npy'
output=np.load( os.path.join( 'data_files' , fname ) )
output = output[ np.nonzero( output[: , 3 ] ) ]

points = output[ :, 0:3]
values = output[ : , -1 ]


eigs = griddata( points , values , ( grid_x , grid_y , grid_z ) )
    
out = np.array([np.ravel(grid_x) , np.ravel(grid_y)  , np.ravel(grid_z) , np.ravel(eigs)] ).T
mypts = out[ np.nonzero( np.isnan(out[:, 3] )==False )[0] ]


"""
    Plots the existence region for Sinko-Streifer population model
"""
plt.close('all')

fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(mypts[:, 0] , mypts[:, 1] , mypts[:, 2] , linewidth=0, cmap='jet', shade=True)


ax.view_init( azim=100 , elev=26  )
ax.set_xlabel( '$a$' , fontsize=20 )
ax.set_ylabel( '$b$' , fontsize=20 )
ax.set_zlabel( '$c$' , fontsize=20 )
ax.set_xlim( amin , amax )
ax.set_ylim( bmin , bmax )
ax.set_zlim( cmin , cmax )

plt.savefig( os.path.join( 'images' , 'sinko_exist_region.png' ) , dpi=400)


end = time.time()


print "Elapsed time", round( end - start   , 2 ) ,  "seconds "


