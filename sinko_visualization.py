# -*- coding: utf-8 -*-
"""
Created on Oct 14, 2015

@author: Inom Mirzaev

"""


from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sinko_model_rates import *

import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()



fnames = 'sinko_data.npy'

output=np.load( fnames )
output = output[ np.nonzero( output[: , 3 ] ) ]


points = output[ :, 0:3]
values = output[ : , -1 ]


eigs = griddata( points , values , ( grid_x , grid_y , grid_z ) )
    
out = np.array([np.ravel(grid_x) , np.ravel(grid_y)  , np.ravel(grid_z) , np.ravel(eigs)] ).T

mypts = out[ np.nonzero( np.isnan(out[:, 3] )==False )[0] ]


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

plt.savefig('sinko_exist_region.png', dpi=400)


end = time.time()


print "Elapsed time " + str( round( (end - start)  , 2)  ) + " seconds"



