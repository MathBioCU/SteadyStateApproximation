# -*- coding: utf-8 -*-
"""
Created on Dec 5, 2015

@author: Inom Mirzaev


"""


from __future__ import division
from scipy.optimize import fsolve
from scipy.integrate import odeint 
from matplotlib import gridspec

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin


# Minimum and maximum floc sizes
x0 = 0
x1 = 1  


# Post-fragmentation density distribution
def gam( y , x , x0 = x0 ):
    
    out = 6*y * ( x - y )  / (x**3)
    
    if type(x) == np.ndarray or type(y) == np.ndarray:        
        out[y>x] = 0

    return out 
    

# Renewal rate
def q(x , x0=x0 ):
        
    return 1*(x + 1)
    
    
#Removal rate    
def rem(x, x0=x0):

     return 0.05* x
     

#Fragmentation rate
def kf(x, x0=x0):

    return 0.1 * x
    
    
#Growth rate    
def g(x, x0=x0, x1=x1):

    return np.exp(-0.1 * x)


#Aggregation rate
def ka(x,y, x1=x1):
    
    out = ( x ** ( 1/3 ) + y ** ( 1/3 ) ) **3 / ( 10**2 )     

    return out


# Initializes uniform partition of (x0, x1) and approximate operator F_n
def initialization(N , x1=x1 , x0=x0):
    
    #delta x
    dx = ( x1 - x0 ) / N
    
    #Uniform partition into smaller frames
    nu = x0 + np.arange(N+1) * dx
    
    #Aggregation in
    Ain = np.zeros( ( N , N ) )
    
    #Aggregation out
    Aout = np.zeros( ( N , N ) )
    
    #Fragmentation in
    Fin = np.zeros( ( N , N ) )
    
    #Fragmentation out
    Fout = np.zeros( N )


    for mm in range( N ):
    
        for nn in range( N ):
            
            if mm>nn:
            
                Ain[mm,nn] = 0.5 * dx * ka(nu[mm], nu[nn+1])
            
            if mm + nn < N-1 :
                
                Aout[mm, nn] = dx * ka( nu[mm+1] , nu[nn+1] )
                    
            if nn > mm :
            
                Fin[mm, nn] = dx * gam( nu[mm+1], nu[nn+1] ) * kf( nu[nn+1] )


    Fout = 0.5 * kf( nu[range( 1 , N + 1 ) ] ) + rem( nu[range( 1 , N + 1 )] )

    #Growth operator
    Gn=np.zeros( ( N , N ) )

    for jj in range(N-1):
        Gn[jj,jj] = -g( nu[jj+1] ) / dx
        Gn[jj+1,jj] = g( nu[jj+1] ) / dx

    Gn[0,:] = Gn[0,:] + q( nu[range( 1 , N+1 ) ] )
    Gn[N-1, N-1] = -g( nu[N] ) / dx
    
    #Growth - Fragmentation out + Fragmentation in
    An = Gn - np.diag( Fout ) + Fin

    return (An, Ain, Aout, nu, N, dx)


largest_eig = np.zeros(20)


for mm in range(20):
    
    An, Ain, Aout, nu, N, dx = initialization( ( mm + 1) * 5)
    
    def root_finding( y ,  An=An , Aout=Aout , Ain=Ain):
    
        if np.any( y < 0):
        
            out =  np.abs(y)
        else:
            a = np.zeros_like(y)
            
            a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    
    
            out = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , a).T - ( Aout.T * y ).T + An , y ) 
            
        return out
        
    def exact_jacobian(y, An=An, Aout=Aout, Ain=Ain):
    
        a = np.zeros_like(y)
    
        a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ] 
    
        out = An - ( Aout.T * y ).T - np.diag( np.dot(Aout , y) ) + 2*Ain * lin.toeplitz( np.zeros_like(y) , a).T
    
        return out    

    
    sol = fsolve(root_finding ,  500*np.ones(N) , fprime = exact_jacobian , xtol = 1e-8)

    largest_eig[ mm ] = np.max( np.real( lin.eig( dx * exact_jacobian( sol ) ) [0] ) )
    
 
An, Ain, Aout, nu, N, dx = initialization( 1000 )

#The right hand side of the evolution equation
def root_finding( y ,  An=An , Aout=Aout , Ain=Ain):
    
    if np.any( y < 0):
        
        out =  np.abs(y)
    else:
        a = np.zeros_like(y)
    
        a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    
    
        out = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , a).T - ( Aout.T * y ).T + An , y ) 
              
    return out

#Exact Jacobian of the RHS 
def exact_jacobian(y, An=An, Aout=Aout, Ain=Ain):
    
    a = np.zeros_like(y)
    
    a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ] 
    
    out = An - ( Aout.T * y ).T - np.diag( np.dot(Aout , y) ) + 2*Ain * lin.toeplitz( np.zeros_like(y) , a).T
    
    return out    

sol = fsolve(root_finding ,  500*np.ones(N) , fprime = exact_jacobian , xtol = 1e-8)
sol [sol <= 0] = 0

larg_eval_for_1000 = np.max( np.real( lin.eig( dx * exact_jacobian( sol ) ) [0] ) )

dim_size = np.arange(5, 105, 5)


plt.close('all')


plt.figure(1)

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

plt.savefig('evals_convergence.png' , dpi=400 ,  bbox_inches='tight')


for mm in [10, 50 , 200]:
    
    An, Ain, Aout, nu, N, dx = initialization( mm )
    
    def root_finding( y ,  An=An , Aout=Aout , Ain=Ain):
    
        if np.any( y < 0):
        
            out =  np.abs(y)
        else:
            a = np.zeros_like(y)
            
            a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    
    
            out = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , a).T - ( Aout.T * y ).T + An , y ) 
            
        return out
        
    def exact_jacobian(y, An=An, Aout=Aout, Ain=Ain):
    
        a = np.zeros_like(y)
    
        a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ] 
    
        out = An - ( Aout.T * y ).T - np.diag( np.dot(Aout , y) ) + 2*Ain * lin.toeplitz( np.zeros_like(y) , a).T
    
        return out    

    
    sol = fsolve(root_finding ,  500*np.ones(N) , fprime = exact_jacobian , xtol = 1e-8)


    evals = np.linalg.eig( dx*exact_jacobian( sol ) )[0]
    real_part = np.real( evals )
    imag_part = np.imag( evals )
        
    plt.close('all')
    
    
    fig, ax = plt.subplots()
    
    ax.set_xlim( -2 , 0.5 )
    ax.set_ylim( -1 , 1 )    
    ax.set_aspect('equal')
    ax.scatter(real_part , imag_part)
    ax.grid(True, which='both')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    
    plt.savefig('dx_jac_evals_'+str(mm) +'.png' , dpi=400 ,  bbox_inches='tight' )





