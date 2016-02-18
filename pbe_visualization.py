# -*- coding: utf-8 -*-
"""
Created on Oct 14, 2015

@author: Inom Mirzaev

"""


from __future__ import division
from scipy.spatial import  ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from pbe_model_rates import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time, os





start = time.time()

fname = 'pbe_data.npy'

output=np.load( os.path.join( 'data_files' , fname ) )
output = output[ np.nonzero( output[: , 3 ] ) ]



points = output[ :, 0:3]
values = output[ : , -1 ]

  
eigs = griddata( points , values , ( grid_x , grid_y , grid_z ) )
    
out = np.array([np.ravel(grid_x) , np.ravel(grid_y)  , np.ravel(grid_z) , np.ravel(eigs)] ).T

mypts = out[ np.nonzero( np.isnan(out[:, 3] )==False )[0] ]


plt.close('all')

fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')


hull = ConvexHull( mypts[ : , 0:3] )
simp = hull.points[ hull.vertices ]

ax.plot_trisurf(mypts[:, 0] , mypts[:, 1] , mypts[:, 2] , triangles=hull.simplices, 
                linewidth=0, color='#8A2BE2', shade=False)


ax.view_init(  azim=115 , elev=25 )

ax.set_xlabel( '$a$'    , fontsize=20 )
ax.set_ylabel( '$b$'    , fontsize=20 )
ax.set_zlabel( '$c$'    , fontsize=20 )
ax.set_xlim( amin , amax )
ax.set_ylim( bmin , bmax )
ax.set_zlim( cmin , cmax )


plt.savefig( os.path.join( 'images' , 'existence_region.png' ) , dpi=400 ,bbox_inches='tight')




def shiftedColorMap( cmap , start=0 , midpoint=0.5 , stop=1.0 , name='shiftedcmap'):
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


fig = plt.figure(1)

ax = fig.add_subplot(111, projection='3d')

neg_jac = np.nonzero( mypts[ : , -1] < 0 )[0]
pos_jac = np.nonzero( mypts[ : , -1] >= 0 )[0]

ax.scatter( mypts[ neg_jac , 0] , mypts[ neg_jac , 1 ] , mypts[ neg_jac , 2 ] , color='red' , label = 'stable' )
ax.scatter( mypts[ pos_jac , 0] , mypts[ pos_jac , 1 ] , mypts[ pos_jac , 2 ] , color='blue' , label = 'unstable' )

scatter1_proxy = matplotlib.lines.Line2D( [0],[0] , linestyle="none" , c='blue', marker = 'o' )
scatter2_proxy = matplotlib.lines.Line2D( [0],[0] , linestyle="none" , c='red', marker = 'o' )
ax.legend( [ scatter1_proxy , scatter2_proxy ] , [ 'stable' , 'unstable' ] , numpoints = 1)

ax.view_init(  azim=115 , elev=25 )
ax.set_xlabel( '$a$'    , fontsize=20 )
ax.set_ylabel( '$b$'    , fontsize=20 )
ax.set_zlabel( '$c$'    , fontsize=20 )
ax.set_xlim( amin , amax )
ax.set_ylim( bmin , bmax )
ax.set_zlim( cmin , cmax )

plt.savefig( os.path.join( 'images' , 'stability_region.png' ) , dpi=400 , bbox_inches='tight' )


end = time.time()



print "Visualization time " + str( round( (end - start)  , 2)  ) + " seconds"



