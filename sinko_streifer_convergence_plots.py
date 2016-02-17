# -*- coding: utf-8 -*-
"""
Created on  Oct 14 13:18:38 2015

@author: Inom Mirzaev

This code generates convergence plots for the steady states of the famous Sinko-Streifer model. 
Infinitesimal generator G is apporixmated by an n-by-n matrix G_n. Consequently,
steady states of G is approximated by zeros of the matrix G_n. 

"""

from __future__ import division
from scipy.integrate import quad 
from matplotlib.legend_handler import HandlerLine2D

import numpy as np
import matplotlib.pyplot as plt


# Minimum and maximum floc sizes
x0 = 0
x1 = 1

# Renewal rate q(x)
def q(x , x0=x0 ):
  
    return 1 / np.log(2)*(x + 1)
    
#Removal rate mu(x)    
def rem(x, x0=x0):
   
     return 1
     
     
#Growth rate     
def g(x, x0=x0, x1=x1):

    return x+1

  
# Removal over growth for integration  
def mug_int(x):
       
    return - rem(x) / g(x)
    
    
#Given a singular matrix this function returns nullspace of that matrix    
def null(a, rtol=1e-5):
    
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    
    return v[rank:].T.copy() 
 
 
# Initializes uniform partition of (x0, x1) and approximate operator G_n
def initialization(N , x1=x1 , x0=x0):
    
    dx = ( x1 - x0 ) / N
    nu = x0 + np.arange(N+1) * dx
    
    Gn=np.zeros( ( N , N ) )

    for jj in range(N-1):
        Gn[jj,jj] = -g( nu[jj+1] ) / dx - rem(nu[jj+1])
        Gn[jj+1,jj] = g( nu[jj+1] ) / dx

    Gn[0,:] = Gn[0,:] + q( nu[range( 1 , N+1 ) ] )
    Gn[N-1, N-1] = -g( nu[N] ) / dx - rem(nu[N])

    return (Gn, nu, N, dx)
    
    
    
#An array storing errors in steady state approximation    
conv_error = np.array([])    

dimens = np.array( [100, 110 , 125, 140,  160, 190, 230, 260, 300 , 350, 400, 450, 500] )    
for ndim in range(len(dimens)):

    Gn , nu , N , dx = initialization( dimens[ndim]  )

    #The right hand side of the approximate ODE system
    def root_finding( y ,  Gn=Gn):
    
        if np.any( y < 0):        
            out =  np.abs(y)
        else:
            a = np.zeros_like(y)    
            a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    
            out = np.dot( Gn , y )             
            
        return out  
    
    #Approximate steady state calculated as a root of G_n           
    appr_sol = np.abs( null(Gn) )
      
    #Exact steady state comparison
    xx = np.linspace(0 , 1 , len(appr_sol) )
    yy= np.zeros_like(xx)

    for num in range( len(xx) ):    
        yy[num] = quad( mug_int , 0 , xx[num] )[0]

    actual_sol =  appr_sol[0] * 1 / g(xx) * np.exp( yy )

    #Error between exact and approximate steady state
    conv_error = np.append( conv_error,  np.max( np.abs( actual_sol - appr_sol ) ) ) 
     
    #An array used for plot generation
    if ndim==1:
        
        appr_sol_100 = appr_sol
        actual_sol_100 = actual_sol
        xx_100 = xx
      

plt.close('all')

# Generates Meshsize vs Absolute error plot
plt.figure(1)

step_size = 1/dimens

plt.plot( step_size ,  conv_error , linewidth=1, color='blue', marker='o', markersize=10)

x1, x2, y1, y2 = plt.axis()
plt.axis([x1 - 0.1*x1, x2+0.01*x2, y1, y2])
plt.ylabel( 'Absolute error', fontsize=12)
plt.xlabel( 'Mesh size', fontsize=12)

plt.savefig('convergence_plot_sinko_streifer.png', dpi=400)

#Generates comparison plot for n=100
plt.figure(2)

line1, = plt.plot( xx_100 , actual_sol_100 , linewidth=2 , color='blue', label='Exact solution' )
line2, = plt.plot( xx_100 , appr_sol_100 , linewidth=0 , color='black' ,
                   marker='o', markersize=3 , label='Approximate solution')

plt.ylabel( '$u_{*}(x)$ (number density)', fontsize=12)
plt.xlabel( '$x$ (size of a floc)', fontsize=12)
plt.legend(handler_map={line1: HandlerLine2D()})


plt.savefig( 'sinko_streifer_100.png' , dpi=400 )



