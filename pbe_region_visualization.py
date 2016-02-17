# -*- coding: utf-8 -*-
"""
Created on Oct 14, 2015

@author: Inom Mirzaev

"""


from __future__ import division
from scipy.spatial import  ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time



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


start = time.time()

fnames = [ 'data_2016_01_22_19_31.npy',
          'data_2016_01_25_08_31.npy' ]


output=np.load( fnames[0] )
output = output[ np.nonzero( output[: , 3 ] ) ]


pts = output[0:2]
for nn in range(9 , -1, -1):
    
    ddd = output[ np.nonzero( output[: , 1 ]  >= nn *0.1 +0.05 ) ]
    ddd  = ddd[ np.nonzero( ddd[: , 1 ]  <  (nn+1) *0.1 + 0.05 ) ]
    
    for mm in range(0):
        distances =  cdist( ddd[:, [0,2] ] , ddd[:, [0,2] ]) + np.diag( np.ones( len(ddd) ) )    
        distances[ distances > 0.1*np.sqrt(2) ] = 0
        distances[ distances > 0 ] = 1
        
        odd_indice =  np.sum(distances , axis=1)
        
        ddd = ddd[ np.nonzero( odd_indice > 2) ]
        
    pts = np.append(pts, ddd , axis=0)    

 
pts = np.delete(pts, range(2), axis=0)    
output = pts





points = output[ :, 0:3]
values = output[ : , -1 ]

grid_x, grid_y , grid_z = np.meshgrid( np.linspace(0, 15, 150 ) , np.linspace(0, 1, 10) ,  np.linspace(0,5 , 50) )
    
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



annot_pts = [ [ 5, 0.2 , 0.1 ] , 
              [ 8 , 0.5, 0.1 ] , 
              [ 12 , 0.9 , 0.1] ]

slices = [ 0 , 4, 9]
amax = []
bmax = []
cmax = [] 
             
for nnn in range( len(slices) ) :
    
    nn = slices[ nnn ]    
    ddd = output[ np.nonzero( output[: , 1 ]  >= nn *0.1 +0.05 ) ]
    amat  = ddd[ np.nonzero( ddd[: , 1 ]  <  (nn+1) *0.1 + 0.05 ) ]
    
    pts = amat[: , [0,2] ]
    
    hull                = ConvexHull(pts)
    amax.append( np.max( pts[ hull.vertices , 0 ] ) )
    bmax.append((nn+1) *0.1 )    
    cmax.append( np.max( pts[ hull.vertices , 1 ] ) )
    
    num_vert            = len(hull.vertices)
    myvert              = amat[0 , 1]*np.ones( ( num_vert + 1 , 3 ) )
    myvert[ -1 , 0]     = pts[ hull.vertices[0] , 0 ]
    myvert[ :-1 , 0]    = pts[ hull.vertices , 0 ]
    myvert[ -1 , 2]     = pts[ hull.vertices[0] , 1 ]
    myvert[ :-1 , 2]    = pts[ hull.vertices , 1 ]
    
    ax.plot( myvert[:, 0] , myvert[:, 1] , myvert[:, 2] , 'k', lw=2 )
    
    label = '$b=' + str( ( nn+1) * 0.1 ) +'$'
    ax.text(amax[-1] , bmax[-1] ,  0.1 , label, zdir ='x' , fontsize=15)

ax.plot( amax , bmax , cmax , linewidth=2 , color='black')
#ax.plot([ 0.48 , 12.4 ] , [ 0.9 , 0.9 ] , [ 4.85 , 4.95 ] , linewidth=0.5 , color='black')

ax.view_init(  azim=-90 , elev=40 )
ax.set_xlabel( '$a$'    , fontsize=20 )
ax.set_ylabel( '$b$'    , fontsize=20 )
ax.set_zlabel( '$c$'    , fontsize=20 )
ax.set_xlim(0, 15)
ax.set_ylim(0, 1)
ax.set_zlim(0, 5)



plt.savefig('pos_steady_eigen.png', dpi=400 ,bbox_inches='tight')



plt.figure(1)

nn = 0

ddd = output[ np.nonzero( output[: , 1 ]  >= nn *0.1 +0.05 ) ]
amat  = ddd[ np.nonzero( ddd[: , 1 ]  <  (nn+1) *0.1 + 0.05 ) ]


points = amat[ : , [0,2] ]
values = amat[ : , -1 ]
grid_x, grid_y = np.meshgrid( np.linspace(0.1, 15 , 1000 ) , np.linspace(0.1 , 5 , 1000) )

grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

vmin = np.min( values )
vmax = np.max( values )
mid = 1 - vmax / ( vmax + np.abs( vmin ) )

orig_cmap = matplotlib.cm.bwr
shifted_cmap = shiftedColorMap(orig_cmap, midpoint=mid, name='shifted')

plt.imshow( np.ones_like( grid_z ) , origin='lower' , cmap ='summer' , vmin = 0, vmax = 1)
plt.hold(True)

plt.imshow( grid_z , origin='lower' , cmap = shifted_cmap )


plt.xticks(np.linspace(0, len(grid_x) , 6 ) , np.linspace(0, 15, 6 ) )
plt.yticks(np.linspace(0, len(grid_y) , 6 ) , np.linspace(0, 5, 6  ) )
plt.xlabel( '$a$' , fontsize=20 )
plt.ylabel( '$c$' , fontsize=20 )
plt.colorbar()
plt.savefig('b_0.1.png' , dpi=400 ,bbox_inches='tight')


plt.figure(2)

nn = 4

ddd = output[ np.nonzero( output[: , 1 ]  >= nn *0.1 +0.05 ) ]
amat  = ddd[ np.nonzero( ddd[: , 1 ]  <  (nn+1) *0.1 + 0.05 ) ]


points = amat[ : , [0,2] ]
values = amat[ : , -1 ]
grid_x, grid_y = np.meshgrid( np.linspace(0.1, 15 , 1000 ) , np.linspace(0.1 , 5 , 1000) )

grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

vmin = np.min( values )
vmax = np.max( values )
mid = 1 - vmax / ( vmax + np.abs( vmin ) )

orig_cmap = matplotlib.cm.bwr
shifted_cmap = shiftedColorMap(orig_cmap, midpoint=mid, name='shifted')

plt.imshow( np.ones_like( grid_z ) , origin='lower' , cmap ='summer' , vmin = 0, vmax = 1)
plt.hold(True)

plt.imshow( grid_z , origin='lower' , cmap = shifted_cmap )

plt.xticks(np.linspace(0, len(grid_x) , 6 ) , np.linspace(0, 15, 6 ) )
plt.yticks(np.linspace(0, len(grid_y) , 6 ) , np.linspace(0, 5, 6 ) )
plt.xlabel( '$a$' , fontsize=20 )
plt.ylabel( '$c$' , fontsize=20 )

plt.colorbar()
plt.savefig('b_0.5.png' , dpi=400 ,bbox_inches='tight')



plt.figure(3)

nn = 9

ddd = output[ np.nonzero( output[: , 1 ]  >= nn *0.1 +0.05 ) ]
ddd  = ddd[ np.nonzero( ddd[: , 1 ]  <  (nn+1) *0.1 + 0.05 ) ]
ddd1 = ddd[ np.nonzero( ddd[ : , -1 ] <= 0 )[0] ]
ddd1 = ddd[ np.nonzero( ddd[ : , 0 ] <= 6.3 )[0] ]
ddd2 = ddd[ np.nonzero( ddd[ : , -1 ] > 0)[0] ]
amat = np.append(ddd1, ddd2 , axis=0)


points = amat[ : , [0,2] ]
values = amat[ : , -1 ]
grid_x, grid_y = np.meshgrid( np.linspace( 0.1, 15 , 1000 ) , np.linspace( 0.1 , 5 , 1000) )

grid_z = griddata( points , values , ( grid_x, grid_y ) , method = 'linear' )

vmin = np.min( values )
vmax = np.max( values )
mid = 1 - vmax / ( vmax + np.abs( vmin ) )

orig_cmap = matplotlib.cm.bwr
shifted_cmap = shiftedColorMap(orig_cmap, midpoint=mid, name='shifted')

plt.imshow( np.ones_like( grid_z ) , origin='lower' , cmap ='summer' , vmin = 0, vmax = 1)
plt.hold(True)

plt.imshow( grid_z , origin='lower' , cmap = shifted_cmap )

plt.xticks(np.linspace(0, len(grid_x) , 6 ) , np.linspace(0, 15, 6 ) )
plt.yticks(np.linspace(0, len(grid_y) , 6 ) , np.linspace(0, 5, 6  ) )
plt.xlabel( '$a$' , fontsize=20 )
plt.ylabel( '$c$' , fontsize=20 )

plt.colorbar()
plt.savefig('b_1.0.png' , dpi=400 , bbox_inches='tight' )


end = time.time()



print "Elapsed time " + str( round( (end - start)  , 2)  ) + " seconds"



