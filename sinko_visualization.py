# -*- coding: utf-8 -*-
"""
Created on Oct 14, 2015

@author: Inom Mirzaev

"""


from __future__ import division
from scipy.spatial import  ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.legend_handler import HandlerLine2D



import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()



fnames = [ 'data_2016_01_23_22_38.npy']

output=np.load( fnames[0] )
output = output[ np.nonzero( output[: , 3 ] ) ]



#pyfits.writeto('data.fits', output[:, 0:3])
#eigs=np.load('eigenvalues1.npy')



plt.close('all')


fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(output[:, 0] , output[:, 1] , output[:, 2] , linewidth=0, cmap='jet', shade=True)



ax.view_init( azim=131 , elev=26  )
ax.set_xlabel( '$a$' , fontsize=20 )
ax.set_ylabel( '$b$' , fontsize=20 )
ax.set_zlabel( '$c$' , fontsize=20 )
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

plt.savefig('sinko_exist_region.png', dpi=400)



end = time.time()


print "Elapsed time " + str( round( (end - start)  , 2)  ) + " seconds"



