# -*- coding: utf-8 -*-
"""
Created on Oct 14, 2015

@author: Inom Mirzaev

This code approximates the stationary solutions of the microbial flocculation model
by the stationary solutions of the system of nonlinear ODEs. 
The code uses scipy.optimize.fsolve to find roots of nonlinear function. 
This function uses Powell's hybrid method. For faster convergence rate, we provided 
the solver with the exact Jacobian defined in the paper. For the purpose of 
illustration, the model parameters have been chosen such that they satisfy existence conditions 
derived in Mirzaev and Bortz (2015).

"""


from __future__ import division
from scipy.optimize import fsolve 
from matplotlib import gridspec
from matplotlib.legend_handler import HandlerLine2D


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin
import time


start = time.time()

# Minimum and maximum floc sizes
x0 = 0
x1 = 1

# Post-fragmentation density distribution
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



An, Ain, Aout, nu, N, dx = initialization( 100 , 5 , 1 , 1 )


print "Largest eigenvalue " + str( np.max( np.real( np.linalg.eig( An )[0] ) ) )


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

for mm in range( 100):

    seed = 10*(mm-1)*np.arange(N)
    
    sol = fsolve( root_finding ,  seed , fprime = exact_jacobian , xtol = 1e-8 , full_output=1 )

    if sol[2]==1 and np.linalg.norm( sol[0] ) > 0.1 and np.all( sol[0] > 0 ):
        print mm
        break

sol = sol[0]

print 'Frobenius norm of obtained equilibrium   '+ str(np.sqrt( np.sum( sol * sol ) ))

a = root_finding( sol )

print 'Absolute error from Powell\'s hybrid method   ' + str(np.linalg.norm(a, np.inf) )

plt.close('all')


# Generates a plot of steady state solution
plt.figure(1)

plt.plot( nu[range(N)] ,  sol , linewidth=2, color='blue') 
plt.ylabel( '$u_{*}(x)$', fontsize=20)
plt.xlabel( '$x$', fontsize=20)
plt.savefig('stationary_solution.png', dpi=400)

plt.show()




print 'Number of particles at steady state   ' + str ( np.sum( dx * sol ) )

aaa = ( nu[range(1,N+1)] ** 2 - nu[range(N)] ** 2 ) / 2
print 'Total mass of particles at steady state   ' + str( np.sum( aaa * sol ) )


#Total number (zeroth moment) of the particles for different n
number = np.zeros(9)
#Total mass (first moment) of the particles for different n
mass   = np.zeros(9)

#List of solution
sol_set = []

#List of uniform partitions
nu_set = []

for nn in range(2 , 11):
    
    An, Ain, Aout, nu, N, dx = initialization( nn * 50 , 5 , 1 , 1)
    
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

        
    for mm in range( 100):
    
        seed = 10*(mm-1)*np.arange(N)
        
        sol = fsolve( root_finding ,  seed , fprime = exact_jacobian , xtol = 1e-8 , full_output=1 )
    
        if sol[2]==1 and np.linalg.norm( sol[0] ) > 0.1 and np.all( sol[0] > 0 ):
            print nn
            break
    
    sol = sol[0]
    sol_set.append(sol)
    nu_set.append(nu[range(N)])
    
    number[nn-2] = dx * np.sum( sol )
    mass[nn-2]   = np.sum( sol * ( nu[range(1,N+1)] ** 2 - nu[range(N)] ** 2 ) / 2)
    

An, Ain, Aout, nu, N, dx = initialization( 1000 , 5 , 1 , 1 )

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


for mm in range( 100):

    seed = 10*mm*np.arange(N)
    
    sol = fsolve( root_finding ,  seed , fprime = exact_jacobian , xtol = 1e-8 , full_output=1 )

    if sol[2]==1 and np.linalg.norm( sol[0] ) > 0.1 and np.all( sol[0] > 0 ):
        print mm
        break


moment0 = dx * np.sum( sol[0] )
moment1 = np.sum( sol[0] * ( nu[range(1,N+1)] ** 2 - nu[range(N)] ** 2 ) / 2)



plt.close('all')

#Generates the plot Change in steady state solutions with increasing dimension n
fig = plt.figure( figsize = ( 12 , 6 ) )
gs = gridspec.GridSpec(nrows=1 , ncols=2 , left=0.05, right=0.95 , 
                       wspace=0.3 , hspace=0.2 , width_ratios=[1 , 1 ] , height_ratios=[1])


ax0=plt.subplot( gs[0] )
ax1=plt.subplot( gs[1] )
    

leg1,  = ax0.plot( nu_set[0] ,  sol_set[0] , color='blue' , label='$n=100$', linewidth=2)
leg2, = ax0.plot( nu_set[-1] , sol_set[-1] , 'black' , label='$n=500$' , linewidth=2)
ax0.set_xlabel( '$x$' , fontsize = 20)
ax0.set_ylabel( '$u_{*}(x)$' , fontsize = 20 )

ax0.legend(handler_map={leg1: HandlerLine2D()}, fontsize=20)


ax1.axhline(y =  moment0 , color='red' , linestyle='--' , linewidth=2)

leg3, = ax1.plot( 50*np.arange( 2 , len(number) + 2 ) , number , label='Total number', color='red' , linewidth=2)
leg4, = ax1.plot( 50*np.arange( 2 , len(mass) + 2 ) , mass , label='Total mass', color='green' , linewidth=2)


ax1.axhline(y =  moment1 , color='green' , linestyle=':' , linewidth=2)
myaxis = list ( ax1.axis() )
myaxis[-1] = moment0 + 5
ax1.axis( myaxis )
ax1.legend(handler_map={leg3: HandlerLine2D()}, fontsize=16)   
ax1.set_xlabel( '$n$ (dimension)' , fontsize = 12 )


rect=fig.patch
rect.set_facecolor('white')
plt.show()

plt.savefig('change_in_moments.png', dpi=400 , bbox_inches='tight'  )

end = time.time()

print "Elapsed time " + str( round( (end - start) , 1)  ) + " seconds"



