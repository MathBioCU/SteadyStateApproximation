# -*- coding: utf-8 -*-
"""
Created on Oct 14, 2015

@author: Inom Mirzaev

"""


from pbe_model_rates import *
from scipy.optimize import fsolve 

import multiprocessing as mp
import time, cPickle


start = time.time()
 
 


amin = 0
amax = 15
apart = 150

x_ = np.linspace( amax , amin , apart , endpoint=False).tolist()
x_.sort()

bmin = 0
bmax = 1
bpart = 10

y_ = np.linspace( bmax , bmin , bpart , endpoint=False).tolist()
y_.sort()

cmin = 0
cmax = 5
cpart = 50

z_ = np.linspace( cmax , cmin , cpart , endpoint=False).tolist()
z_.sort()

x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')


x = np.ravel(x)
y = np.ravel(y)
z = np.ravel(z)

myarray = np.array([x, y, z]).T



def region_plots(nn, myarray = myarray):
    
    pos_sol = 0
    eigs2=0

    An, Ain, Aout, nu, N, dx = initialization( 50 , myarray[nn , 0] , myarray[nn , 1] , myarray[nn, 2] )
    
    root_finding  = partial( approximate_IG , An=An, Aout=Aout, Ain=Ain )    
    exact_jacobian = partial( jacobian_IG , An=An, Aout=Aout, Ain=Ain)

    
    for mm in range( 10 ):

        if mm < 5:    
            seed = 10 * (mm + 1) * np.ones(N)            
            
        else:
            seed = 10* ( mm - 4 ) * np.arange(N)
            
        sol = fsolve( root_finding ,  seed , fprime = exact_jacobian , xtol = 1e-8 , full_output=1 )

        if sol[2]==1 and np.linalg.norm( sol[0] ) > 1 and np.all( sol[0] > 0 ):
            pos_sol = 1
            eigs2 = np.max(  np.real ( np.linalg.eig(  exact_jacobian( sol[0] ) )[0] ) )  

            break
        
    return ( myarray[nn , 0] , myarray[nn , 1] , myarray[nn, 2] , pos_sol ,  eigs2 )
    
    
   
if __name__ == '__main__':
    
    pool = mp.Pool(processes = mp.cpu_count() )
    ey_nana = range( len( myarray) )
    result = pool.map( region_plots , ey_nana )
    
    output = np.asarray(result)
    
    #output = output[ np.nonzero( output[: , 3 ] ) ]
    
    
    fname = 'data'   
    np.save(fname , output )
    
    

end = time.time()


print "Elapsed time " + str( round( (end - start) / 60 , 1)  ) + " minutes"



