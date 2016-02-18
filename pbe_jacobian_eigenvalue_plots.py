# -*- coding: utf-8 -*-
#Created on Dec 5, 2015
#@author: Inom Mirzaev

"""
    This program plots change in eigenvalues of the Jacobian matrix versus
    the dimension size of the approximate space. This code also plots spectrum
    of the Jacobian matrix for various dimension size n. All the plots are 
    saved in the 'images' folder.
"""


from __future__ import division
from scipy.optimize import fsolve
from pbe_model_rates import *

import matplotlib.pyplot as plt
import scipy.linalg as lin
import os



largest_eig = np.zeros(20)

for nn in range(20):
    
    #Initialize the approximate operators
    An, Ain, Aout, nu, N, dx = initialization( ( nn + 1) * 5  , a , b, c)
    
    root_finding  = partial( approximate_IG , An=An, Aout=Aout, Ain=Ain )    
    exact_jacobian = partial( jacobian_IG , An=An, Aout=Aout, Ain=Ain)

    #Search the root of the system for 10 different initial seeds
    for mm in range( 10 ):

        
        seed = 2**mm * np.ones(N)                  
        sol = fsolve( root_finding ,  seed , fprime = exact_jacobian , xtol = 1e-8 , full_output=1 )
            
        #Break the loop if a positive solution is found    
        if sol[2]==1 and np.linalg.norm( sol[0] ) > 1 and np.all( sol[0] > 0 ):
            break
        
    largest_eig[ nn ] = np.max( np.real( lin.eig( dx * exact_jacobian( sol[0] ) ) [0] ) )
    
 
"""
    Rightmost eigenvalue of Jacobian for n=1000
"""
 
#Initialize the approximate operators 
An, Ain, Aout, nu, N, dx = initialization( 1000 , a , b, c )

root_finding  = partial( approximate_IG , An=An, Aout=Aout, Ain=Ain )    
exact_jacobian = partial( jacobian_IG , An=An, Aout=Aout, Ain=Ain)

#Search the root of the system for 10 different initial seeds
for mm in range( 10 ):


    seed = 2**mm * np.ones(N)                    
    sol = fsolve( root_finding ,  seed , fprime = exact_jacobian , xtol = 1e-8 , full_output=1 )
    
    #Break the loop if a positive solution is found  
    if sol[2]==1 and np.linalg.norm( sol[0] ) > 1 and np.all( sol[0] > 0 ):
        break

larg_eval_for_1000 = np.max( np.real( lin.eig( dx * exact_jacobian( sol[0] ) ) [0] ) )



plt.close('all')


"""
    Plots changes in the rightmost eigenvalue of the Jacobian wrt to chanding 
    dimension size n
"""

plt.figure(1)


dim_size = np.arange(5, 105, 5)

plt.plot(dim_size , largest_eig , linewidth=1 , color='blue' ,
                   marker='o', markersize=10 , )

plt.axhline(y =  larg_eval_for_1000 , color='k' , linestyle='--')

myaxis = list ( plt.axis() )
myaxis[0] = 4.5
myaxis[1] = 100.5
myaxis[-1] = 0.025
plt.axis( myaxis )

plt.xticks( [5, 20 , 40, 60, 80, 100] )

plt.xlabel('$n$' , fontsize=20)
plt.ylabel('Largest real part' , fontsize=15)

fname = 'evals_convergence.png'
plt.savefig( os.path.join( 'images' , fname ) , dpi=400 ,  bbox_inches='tight')



"""
    Plots spectrum  of the Jacobian matrix for various dimension size n.
    Dimensions shoudl be provided in the following list.
"""

dim_list = [20 , 50 , 200]

for nn in dim_list:
    
    #Initialize approximate operators
    An, Ain, Aout, nu, N, dx = initialization( nn , a , b , c )
    
    #Initialize root finding problem
    root_finding  = partial( approximate_IG , An=An, Aout=Aout, Ain=Ain )    
    exact_jacobian = partial( jacobian_IG , An=An, Aout=Aout, Ain=Ain)

    
    #Search the root of the system for 10 different initial seeds
    for mm in range( 10 ):
  
        seed = 2**mm * np.ones(N)                        
        sol = fsolve( root_finding ,  seed , fprime = exact_jacobian , xtol = 1e-8 , full_output=1 )
        
        #Break the loop if a positive solution is found
        if sol[2]==1 and np.linalg.norm( sol[0] ) > 1 and np.all( sol[0] > 0 ):
            break

    evals = np.linalg.eig( dx * exact_jacobian( sol[0] ) )[0]
    real_part = np.real( evals )
    imag_part = np.imag( evals )
        
    plt.close('all')
    
    fig, ax = plt.subplots()
    
    #ax.set_xlim( -2 , 0.5 )
    #ax.set_ylim( -1 , 1 )    
    ax.set_aspect('equal')
    ax.scatter(real_part , imag_part)
    ax.grid(True, which='both')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    fname = 'dx_jac_evals_'+str(nn) +'.png'
    plt.savefig( os.path.join( 'images' , fname ) , dpi=400 ,  bbox_inches='tight' )





