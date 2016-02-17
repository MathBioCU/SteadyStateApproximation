# -*- coding: utf-8 -*-
"""
Created on Oct 18  2015

@author: Inom Mirzaev

To measure sensitivity of the flocculation model to different the post-fragmentation 
density functions, we generate steady state plots for differrent density functions 
while keeping other rates fixed.

"""

from __future__ import division
from scipy.optimize import fsolve 
from matplotlib.legend_handler import HandlerLine2D

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin
import time


start = time.time()

# Minimum and maximum floc sizes
x0 = 0
x1 = 1


def gam( y , x , x0 = x0 ):
    
    out = 6*y * ( x - y )  / (x**3)
   
    if type(x) == np.ndarray or type(y) == np.ndarray:
        
        out[y>x] = 0

    return out    


#Fragmentation rate

def kf(x, x0=x0):

    return 1 * x
    
#Aggregation rate
def ka(x,y, x1=x1):
    
    out = ( x ** ( 1/3 ) + y ** ( 1/3 ) ) **3      

    return out

    
# Initializes uniform partition of (x0, x1) and approximate operator F_n
def initialization(N , a, b, c, x1=x1 , x0=x0):
    
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
    
    def q(x , x0=x0 , a=a):
        
        return a*(x + 1)
        
    def g(x, x0=x0, x1=x1, b=b):

        return b*(x+1)
        
            
   #Removal rate    
    def rem(x, x0=x0, c=c):

         return c * x
         


    for mm in range( N ):
    
        for nn in range( N ):
            
            if mm>nn:
            
                Ain[mm,nn] = 0.5 * dx * ka( nu[mm] , nu[nn+1] )
            
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

    return (An , Ain , Aout , nu , N , dx)



sol_list=[]

a_list = []
  

for dd in range(4):

    a_list.append(8 + dd*0.3)    
    An, Ain, Aout, nu, N, dx = initialization( 100 ,  a_list[-1] , 1 , 1 )
    
    #The right hand side of the evolution equation
    def root_finding( y ,  An=An , Aout=Aout , Ain=Ain):
        
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
    
    for mm in range( 10):
    
        if mm < 5:    
            seed = 10 * (mm + 1) * np.ones(N)            
            
        else:
            seed = 10* ( mm - 4 ) * np.arange(N)
        
        sol = fsolve( root_finding ,  seed , fprime = exact_jacobian , xtol = 1e-8 , full_output=1 )
    
        if sol[2]==1 and np.linalg.norm( sol[0] ) > 0.1 and np.all( sol[0] > 0 ):
            print dd
            print np.max(  np.real ( np.linalg.eig(  exact_jacobian( sol[0] ) )[0] ) )  

            break

    sol_list.append( sol[0] )
    
plt.close('all')

plt.figure(1)

line1, = plt.plot( nu[range(N)] ,  sol_list[0] , linewidth=2, color='blue' , label='$a=' + str(a_list[0]) + '$' ) 
line1, = plt.plot( nu[range(N)] ,  sol_list[1] , linewidth=2, color='black', label='$a=' + str(a_list[1]) + '$' ) 
line1, = plt.plot( nu[range(N)] ,  sol_list[2] , linewidth=2, color='green', label='$a=' + str(a_list[2]) + '$' )
line1, = plt.plot( nu[range(N)] ,  sol_list[3] , linewidth=2, color='red', label='$a=' + str(a_list[3]) + '$' ) 
 
plt.legend(handler_map={line1: HandlerLine2D()} , loc=2, fontsize=20) 

plt.ylabel( '$u_{*}(x)$', fontsize=20)
plt.xlabel( '$x$', fontsize=20)
plt.show()

plt.savefig('effect_of_renewal.png', dpi=400)


end = time.time()

print "Elapsed time " + str( round( (end - start) , 1)  ) + " seconds"
