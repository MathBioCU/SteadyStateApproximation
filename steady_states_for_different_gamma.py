# -*- coding: utf-8 -*-
"""
Created on Oct 18  2015

@author: Inom Mirzaev

To measure sensitivity of the flocculation model to different the post-fragmentation 
density functions, we generate steady state plots for differrent density functions 
while keeping other rates fixed.

"""

from __future__ import division
from scipy.optimize import newton_krylov 

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin

# Minimum and maximum floc sizes
x0 = 0
x1 = 1


#Renewal rate q(x)
def q(x , x0=x0 ):
        

    return 1*(x + 1)
    

#Removal rate mu(x)    
def rem(x, x0=x0):
   
 
     return 5* x


#Fragmentation rate
def kf(x, x0=x0):

    return 10 * x
    
 
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
            
                Ain[mm,nn] = 0.5 * dx * ka(nu[mm+1], nu[nn+1])
            
            if mm + nn < N-1 :
                
                Aout[mm, nn] = dx * ka( nu[mm+1] , nu[nn+1] )
                    
            if nn > mm :
            
                Fin[mm, nn] = dx * gam( nu[mm+1], nu[nn+1] ) * kf( nu[nn+1] )


    Fout = 0.5 * kf( nu[range( 1 , N + 1 ) ] ) + rem( nu[range( 1 , N + 1 )] )


    Gn=np.zeros( ( N , N ) )

    for jj in range(N-1):
        Gn[jj,jj] = -g( nu[jj+1] ) / dx
        Gn[jj+1,jj] = g( nu[jj+1] ) / dx

    Gn[0,:] = Gn[0,:] + q( nu[range( 1 , N+1 ) ] )
    Gn[N-1, N-1] = -g( nu[N] ) / dx
    
    #Growth - Fragmentation out + Fragmentation in
    An = Gn - np.diag( Fout ) + Fin

    return (An, Ain, Aout, nu, N, dx)
    
    
#Beta distribution with a=b=2    
def gam( y , x , x0 = x0 ):
    
    out = 6*y * ( x - y )  / (x**3)
   
    if type(x) == np.ndarray or type(y) == np.ndarray:
        
        out[y>x] = 0

    return out    
    
    
An, Ain, Aout, nu, N, dx = initialization( 100 )


#The right hand side of the evolution equation
def root_finding( y ,  An=An , Aout=Aout , Ain=Ain):
    
    if np.any( y < 0):
        
        out =  np.abs(y)
    else:
        a = np.zeros_like(y)    
        a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    
    
        out = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , a).T - ( Aout.T * y ).T + An , y )             
            
    return out  





sol = newton_krylov ( root_finding ,  500*np.ones(N)  , method='lgmres', f_tol=1e-10)
sol [sol <= 0] = 0

plt.close('all')

plt.figure(1)
plt.plot( nu[range(N)] ,  sol , linewidth=2, color='blue') 
plt.ylabel( '$p_{*}(x)$ (number density)', fontsize=16)
plt.xlabel( '$x$ (size of a floc)', fontsize=16)
plt.title( r'$p_{*}(x)\ \mathrm{for} \ Beta(2,\  2)$' , fontsize=20)
plt.show()
plt.savefig('stationary_solution_a_b_2.png', dpi=400)


#Beta distribution with a=b=0.5
def gam( y , x , x0 = x0 ):    
  
    out = 1 / np.pi / np.sqrt( y * (x -y) ) 
    
    if type(x) == np.ndarray or type(y) == np.ndarray:
        
        out[y>x] = 0

    return out        

An, Ain, Aout, nu, N, dx = initialization( 100 )


#The right hand side of the evolution equation
def root_finding( y ,  An=An , Aout=Aout , Ain=Ain):
    
    if np.any( y < 0):
        
        out =  np.abs(y)
    else:
        a = np.zeros_like(y)    
        a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    
    
        out = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , a).T - ( Aout.T * y ).T + An , y ) 
            
            
    return out  



sol = newton_krylov ( root_finding ,  500*np.ones(N)  , method='lgmres', f_tol=1e-10)
sol [sol <= 0] = 0

plt.close('all')
plt.figure(1)
plt.plot( nu[range(N)] ,  sol , linewidth=2, color='blue') 
plt.ylabel( '$p_{*}(x)$ (number density)', fontsize=16)
plt.xlabel( '$x$ (size of a floc)', fontsize=16)
plt.title( r'$p_{*}(x)\ \mathrm{for} \ Beta(0.5,\  0.5)$' , fontsize=20)
plt.show()
plt.savefig('stationary_solution_a_b_05.png', dpi=400)

 
#Beta distribution with a=b=5
def gam( y , x , x0 = x0 ):
    
    out  = 630 * y**4 * (x - y)**4 / (x**9)
    
    if type(x) == np.ndarray or type(y) == np.ndarray:
        
        out[y>x] = 0

    return out       

An, Ain, Aout, nu, N, dx = initialization( 100 )


#The right hand side of the evolution equation
def root_finding( y ,  An=An , Aout=Aout , Ain=Ain):
    
    if np.any( y < 0):
        
        out =  np.abs(y)
    else:
        a = np.zeros_like(y)
    
        a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    
    
        out = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , a).T - ( Aout.T * y ).T + An , y ) 
            
            
    return out  





sol = newton_krylov ( root_finding ,  500*np.ones(N)  , method='lgmres', f_tol=1e-10)
sol [sol <= 0] = 0

plt.close('all')
plt.figure(1)
plt.plot( nu[range(N)] ,  sol , linewidth=2, color='blue') 
plt.ylabel( '$p_{*}(x)$ (number density)', fontsize=16)
plt.xlabel( '$x$ (size of a floc)', fontsize=16)
plt.title( r'$p_{*}(x)\ \mathrm{for} \ Beta(5,\  5)$' , fontsize=20)
plt.show()
plt.savefig('stationary_solution_a_b_5.png', dpi=400)

#Beta distribution with a=20 and b=1
def gam( y , x , x0 = x0 ):

    out = 20 * y**19 / (x**20)
    
    if type(x) == np.ndarray or type(y) == np.ndarray:
        
        out[y>x] = 0

    return out        

An, Ain, Aout, nu, N, dx = initialization( 100 )


#The right hand side of the evolution equation
def root_finding( y ,  An=An , Aout=Aout , Ain=Ain):
    
    if np.any( y < 0):
        
        out =  np.abs(y)
    else:
        a = np.zeros_like(y)
    
        a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    
    
        out = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , a).T - ( Aout.T * y ).T + An , y )             
            
    return out  





sol = newton_krylov ( root_finding ,  500*np.ones(N)  , method='lgmres', f_tol=1e-10)
sol [sol <= 0] = 0

plt.close('all')
plt.figure(1)
plt.plot( nu[range(N)] ,  sol , linewidth=2, color='blue') 
plt.ylabel( '$p_{*}(x)$ (number density)', fontsize=16)
plt.xlabel( '$x$ (size of a floc)', fontsize=16)
plt.title( r'$p_{*}(x)\ \mathrm{for} \ Beta(20,\  1)$' , fontsize=20)
plt.show()
plt.savefig('stationary_solution_a_20_b_1.png', dpi=400)



