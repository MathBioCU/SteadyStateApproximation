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
from scipy.optimize import fsolve , newton_krylov
from scipy.integrate import odeint , quad
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
    

#Renewal rate
def q(x , x0=x0 ):
        
    return 1*(x+1)
    
    
#Removal rate    
def rem(x, x0=x0):

     return 0.05 * x 
     

#Fragmentation rate
def kf(x, x0=x0):

    return 0.1 * x
    
    
#Growth rate    
def g(x, x0=x0, x1=x1):

    return 1*np.exp( -0.1 * x )


#Aggregation rate
def ka(x,y, x1=x1):
    
    out = ( x ** ( 1/3 ) + y ** ( 1/3 ) ) **3 / ( 10**(2) )     

    return out


def first_int(y , x):
    
    return q(y) / g(y)  + quad(gam, 0, x, args=(y,) )[0] * kf(y) / g(y)
    
def second_int(y):

    return ( q(y) + 0.5*kf(y) - rem(y) ) / g(y)    


def stability_condition(x, x0=x0 , x1=x1):
    
    first   = q(x) + 0.5*kf(x) - rem(x)
    second  =  1 -  ( quad( first_int , x , x1 , args=(x,) )[0] ) - ( quad( second_int , 0 , x )[0] ) 
    
    return ( first , second )


check_points = np.arange(0, 1 , 0.01)
conds        = np.zeros( (len(check_points) , 2) ) 

for mm in range( len( check_points ) ):
    
    conds[mm] = stability_condition( check_points[mm] )
    
        
      

if np.any( conds < 0 ):
    
    print 'Parameters do not satisfy the existence conditions.' 
    print 'Convergence to a steady state is not guaranteed...'    

else:
    print 'Parameters satisfy the existence conditions.'
    
    

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

An, Ain, Aout, nu, N, dx = initialization( 100 )


#Gives the largest eigenvalue of the linear part
print 'Largest eigenvalue of the linear part  ' + str( np.max( np.real( lin.eig( An ) [0] ) ) )


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

#sol = newton_krylov ( root_finding ,  100*np.ones(N)  , method='lgmres', f_tol=1e-10)
sol = fsolve(root_finding ,  1e2*np.ones(N) , fprime = exact_jacobian , xtol = 1e-8 , full_output=1 )
print sol[3]

sol = sol[0]
sol [sol <= 0] = 0


print 'Frobenius norm of obtained equilibrium   '+ str(np.sqrt( np.sum( sol * sol ) ))

a = root_finding( sol )

print 'Absolute error from Powell\'s hybrid method   ' + str( np.linalg.norm(a) )



plt.close('all')


# Generates a plot of steady state solution
plt.figure(1)

plt.plot( nu[range(N)] ,  sol , linewidth=2, color='blue') 
plt.ylabel( '$u_{*}(x)$ (number density)', fontsize=12)
plt.xlabel( '$x$ (size of a floc)', fontsize=12)
plt.title( 'Stationary size-distribution' , fontsize=16)
#plt.savefig('stationary_solution.png', dpi=400)

plt.show()

"""
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

for mm in range(2 , 11):
    
    An, Ain, Aout, nu, N, dx = initialization( mm * 50)
    
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

    sol [sol <= 0] = 0
    
    sol_set.append(sol)
    nu_set.append(nu[range(N)])
    
    number[mm-2] = dx * np.sum( sol )
    mass[mm-2]   = np.sum( sol * ( nu[range(1,N+1)] ** 2 - nu[range(N)] ** 2 ) / 2)
    

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

moment0 = dx * np.sum( sol )
moment1 = np.sum( sol * ( nu[range(1,N+1)] ** 2 - nu[range(N)] ** 2 ) / 2)




plt.close('all')

#Generates the plot Change in steady state solutions with increasing dimension n
fig = plt.figure( figsize = ( 12 , 12 ) )
gs = gridspec.GridSpec( 2 , 2 )

ax0=plt.subplot( gs[0] )
ax1=plt.subplot( gs[1] )
ax2=plt.subplot( gs[2] )
ax3=plt.subplot( gs[3] )



def plotfun(ax , x , y , ttl , xlab , ylab):
    ax.plot( x , y , linewidth=2, color='blue' )
    ax.set_title( ttl , fontsize = 16 )
    ax.set_xlabel( xlab , fontsize = 12 )
    ax.set_ylabel( ylab , fontsize = 12 )
    return ax
    

ax0=plotfun( ax0 , nu_set[0] ,  sol_set[0] , 'Steady state for $n=100$','$x$ (size of a floc)' , '$u_{*}(x)$ (number density)' )

ax1.axhline(y =  moment0 , color='k' , linestyle='--' , linewidth=2)
ax1=plotfun( ax1 , 50*np.arange(2 , len(number)+2) , number , 'Change in total number' , '$n$' , 'Number of flocs' )
myaxis = list ( ax1.axis() )
myaxis[-1] = np.floor(moment0) + 1
ax1.axis( myaxis )

ax2=plotfun( ax2 , nu_set[-1] , sol_set[-1] , 'Steady state for $n=500$' , '$x$ (size of a floc)' , '$u_{*}(x)$ (number density)')

ax3=plotfun( ax3 , 50*np.arange(2 , len(mass)+2) , mass , 'Change in total mass' , '$n$' , 'Total mass of flocs')
ax3.axhline(y =  moment1 , color='k' , linestyle=':' , linewidth=2)
myaxis = list ( ax3.axis() )
myaxis[-1] = np.floor(moment1) + 0.2
ax3.axis( myaxis )
    
plt.tight_layout()
rect=fig.patch
rect.set_facecolor('white')
plt.show()

#plt.savefig('change_in_moments.png', dpi=400)


An, Ain, Aout, nu, N, dx = initialization( 100 )

# Ode simulations with arbitrary small initial condition
def myode(y , t, An=An, Aout=Aout, Ain=Ain):
    
    a = np.zeros_like(y)    
    a [ range( 1 , len( a ) ) ] = y [ range( len( y ) - 1 ) ]    
    
    out = np.dot( Ain * lin.toeplitz( np.zeros_like(y) , a).T  - ( Aout.T * y ).T + An , y )             
    out[ out < 0 ] = 0
            
    return out    
 
#Time of the simulation
times = np.arange(0, 100, 0.01)

#Initial condition
y0  =  10**(-6 ) *( 0.2 + 0.1*np.sin( 10*np.linspace( 0 , np.pi , N ) ) )


#Output of the ODE simulation
yout = odeint( myode, y0 , times )

plt.close('all')

#Total number
a = ( nu[range(1,N+1)] ** 2 - nu[range(N)] ** 2 ) / 2

#Total mass
b = dx * yout.sum(axis=1)

fig = plt.figure( figsize = ( 12 , 12 ) )
gs = gridspec.GridSpec(2,2)

ax0=plt.subplot( gs[0] )
ax1=plt.subplot( gs[1] )
ax2=plt.subplot( gs[2] )
ax3=plt.subplot( gs[3] )


#Modified plot function
def plotfun(ax , x , y , ttl , xlab , ylab):
    ax.plot( x , y , linewidth=2 )
    ax.set_title( ttl , fontsize = 16 )
    ax.set_xlabel( xlab , fontsize = 12 )
    ax.set_ylabel( ylab , fontsize = 12 )
    return ax

ax0=plotfun( ax0 , nu[ range(N)] ,  yout[0,:] , 'Initial Condition', '$x$ (size of a floc)' , '$u_{*}(x)$ (number density)' )
ax1=plotfun( ax1 , times , dx * yout.sum(axis=1) , 'Evolution of total number' , 'Time' , 'Total mumber of flocs' )
ax2=plotfun( ax2 , nu[range(N)] , yout[len(times)-1,:] , 'Number density at $t=100$' , '$x$ (size of a floc)' , '$u_{*}(x)$ (number density)')
ax3=plotfun( ax3 , times , np.dot( yout , a) , 'Evolution of total mass' , 'Time' , 'Total mass of flocs')


plt.tight_layout()
rect=fig.patch
rect.set_facecolor('white')
plt.show()

#plt.savefig('semi_stable_equilibrium.png', dpi=400)

"""



