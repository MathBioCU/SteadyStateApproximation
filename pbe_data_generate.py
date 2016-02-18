# -*- coding: utf-8 -*-
"""
Created on Oct 14, 2015

@author: Inom Mirzaev

"""


from pbe_model_rates import *
from scipy.optimize import fsolve 

import multiprocessing as mp
import time , os


start = time.time()
 

x = np.ravel( grid_x )
y = np.ravel( grid_y )
z = np.ravel( grid_z )

myarray = np.array([x, y, z]).T



def region_plots(nn, myarray = myarray):
    
    pos_sol = 0
    eigs2=0

    An, Ain, Aout, nu, N, dx = initialization( 50 , myarray[nn , 0] , myarray[nn , 1] , myarray[nn, 2] )
    
    root_finding  = partial( approximate_IG , An=An, Aout=Aout, Ain=Ain )    
    exact_jacobian = partial( jacobian_IG , An=An, Aout=Aout, Ain=Ain)

    
    for mm in range( 10 ):

    
        seed = 2**mm * np.ones(N)            
            
        sol = fsolve( root_finding ,  seed , fprime = exact_jacobian , xtol = 1e-8 , full_output=1 )

        if sol[2]==1 and np.linalg.norm( sol[0] ) > 1 and np.all( sol[0] > 0 ):
            pos_sol = 1
            eigs2 = np.max(  np.real ( np.linalg.eig(  exact_jacobian( sol[0] ) )[0] ) )  

            break
        
    return ( myarray[nn , 0] , myarray[nn , 1] , myarray[nn, 2] , pos_sol ,  eigs2 )
    
    
   
if __name__ == '__main__':
    
    pool = mp.Pool( processes = ncpus )
    ey_nana = range( len( myarray) )
    result = pool.map( region_plots , ey_nana )
    
    output = np.asarray(result)
    fname = 'pbe_data'   
    np.save( os.path.join( 'data_files' , fname ) , output )
    
end = time.time()


print "Data generation time " + str( round( (end - start) / 60 , 1)  ) + " minutes"



