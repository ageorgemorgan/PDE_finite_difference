# Code for simulating solutions of phi^4 model by the second-oder finite difference method (FD) from

# "Resonance structure in kink-antikink interactions in phi^4 theory"
#  D.K. Campbell et al., Physica D Vol. 9, no.1-2, pp. 1-32 1983.

# We use inhomogeneous Dirichlet boundary conditions to simplify the code, as in the parent paper

# There are two default options for the simulation: a static kink with an odd initial perturbation,
# and a weakly interacting kink-antikink pair 

import numpy as np

from scipy.linalg import lu_factor, lu_solve
from scipy import sparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.colors as colors
import cmocean
import cmocean.cm as cmo

import time
import sys

# Prescribe space-time domain [-L,L] x [0,T]
L= 100.
T= 80. 

# Prescribe space and time stepsize
dt = 0.09
dx = 0.1

# Number of nodes in space and steps in time
Nt = np.int(T/dt)
Nx = np.int(2.*L/dx) # includes boundary points

# Define "discretized domain" vector ie. points where interior solution is sampled
xdata= np.linspace(-L,L, Nx)

# Define vector of time sample points
tdata= np.linspace(0,T, Nt+1) # +1 bcz of initial state

# Initialize solution matrix
udata = np.zeros([Nt+1, Nx] ,dtype=float)

# Specify whether you want to see the kink stability test (Option 1) or the kink-antikink
# interaction test (Option 2). 

option_number = 2

# With the test specified, the code automatically generates the correct initial conditions
# and asymptotic field states 

if option_number == 1: 

    #### PHYSICAL PARAMETERS
    v = 0.0 # soliton velocity
    x0 = 0. #soliton initial position

    gamma = 1./np.sqrt(1.-v**2)

    # Define initial conditions as a function of x (a kink with odd perturbation)

    def initial_state(x):
    
        kink = np.tanh((gamma/np.sqrt(2))*(x-x0))
        amp = 3.5
        width = 0.04
        perturbation = -2.*amp*width*(x-x0)*np.exp(-width*(np.power(x-x0,2)))
        
        return kink + perturbation
   
    def initial_speed(x): 
    
        kink_speed = -(gamma/np.sqrt(2))*v/((np.cosh((gamma/np.sqrt(2))*(x-x0)))**2) 
        
        return kink_speed
    
    # Set asymptotic values of field
    u_inf = 1
    u_neginf = -1
    
    # Plot initial state
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(xdata, initial_state(xdata), '-', color ='xkcd:goldenrod', linewidth=2)

    plt.xlim([-L, L])
    plt.ylim([-1.2, 1.2])
    
    plt.xlabel("$x$", fontsize=23)
    plt.ylabel(r"$\varphi|_{t=0}$", fontsize=23)
    plt.tick_params(axis='x', which='both', top='off')
    plt.xticks(fontsize=12, rotation=0)
    plt.tick_params(axis='y', which='both', right='off')
    plt.yticks(fontsize=12, rotation=0)

    plt.title(r"Initial Field Configuration", fontsize=20)

    #fignamepng = 'stability_ICs' +'.png'
    #plt.savefig(fignamepng, bbox_inches='tight', dpi=600)

    plt.tight_layout()

    plt.show()
    #sys.exit()

if option_number == 2: 

    #### PHYSICAL PARAMETERS
    v = 0.2 # soliton velocity
    x0 = 7. #soliton initial position

    gamma = 1./np.sqrt(1.-v**2)

    # Define initial conditions as a function of x (weakly-interacting kink-antikink pair)

    def initial_state(x):
    
        kink = np.tanh((gamma/np.sqrt(2))*(x-x0))
        akink = -np.tanh((gamma/np.sqrt(2))*(x+x0)) 
        
        return kink + akink + 1
   
    def initial_speed(x):
    
        kink_speed = (gamma/np.sqrt(2))*v/((np.cosh((gamma/np.sqrt(2))*(x-x0)))**2) 
        akink_speed = (gamma/np.sqrt(2))*v/((np.cosh((gamma/np.sqrt(2))*(x+x0)))**2) 
        
        return kink_speed+akink_speed

    # Set asymptotic values of field
    u_inf = 1
    u_neginf = 1

    # Plot initial state
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(xdata, initial_state(xdata), '-', color ='xkcd:goldenrod', linewidth=2)

    plt.xlim([-L, L])
    plt.ylim([-1.2, 1.2])
    
    plt.xlabel("$x$", fontsize=23)
    plt.ylabel(r"$\varphi|_{t=0}$", fontsize=23)
    plt.tick_params(axis='x', which='both', top='off')
    plt.xticks(fontsize=12, rotation=0)
    plt.tick_params(axis='y', which='both', right='off')
    plt.yticks(fontsize=12, rotation=0)

    plt.title(r"Initial Field Configuration", fontsize=20)

    #fignamepng = 'collision_ICs' +'.png'
    #plt.savefig(fignamepng, bbox_inches='tight', dpi=600)

    plt.tight_layout()

    plt.show()
    #sys.exit()
   
# Set initial state
udata[0][:] = initial_state(xdata)

# set boundary contribution correctly   
udata[0][-1]= u_inf
udata[0][0]= u_neginf

# Check the Courant number
courant = dt/dx

if courant > 1: 
    raise Exception('Estimated Courant number is greater than one, decrease the time step size')

# Prescription of the physical + computational parameters is now complete, and we can move on to defining and implementing the FD scheme

# Generate 2nd order finite difference approx of (d/dx)^2
#Uses a routine from the sparse matrix package (need to make sure you import it in the preamble!)

diagA=[np.ones([1,Nx-1], dtype=float), -2*np.ones([1,Nx], dtype=float) ,np.ones([1,Nx-1], dtype=float)]
A=sparse.diags(diagA, [-1,0,1], shape=[Nx,Nx]).toarray()

A[0][0]= 1
A[0][1]= 0

A[-1][-1]= 1
A[-1][-2]= 0

start=time.time()

# Solve for solution at first time step using forward Euler

udata[1][:] = udata[0][:] + dt*initial_speed(xdata) + (dt**2)*(-np.power(udata[0][:],3)+udata[0][:]+((1./dx)**2)*(np.dot(A,udata[0][:])))

udata[1][-1]= u_inf
udata[1][0]= u_neginf

# Define time loop.  
for m in range(2,Nt+1): 
    
    udata[m][:]=  2.*udata[m-1][:] - udata[m-2][:] + (dt**2)*(-np.power(udata[m-1][:],3)+udata[m-1][:]+((1./dx)**2)*(np.dot(A,udata[m-1][:])))
    
    udata[m][-1]= u_inf
    udata[m][0]= u_neginf
    
    # Uncomment to see movie while code runs
    """
    plt.clf()
    plt.plot(xdata, udata[m][:], '-b', linewidth=2)
    plt.xlim(-L,L)
    plt.ylim(-2, 2)
    plt.grid('on')
    plt.pause(0.001)
    plt.draw()
    """
    
end=time.time()  

runtime = end-start

print('Solve Time =', f'{runtime:.4g}', 'seconds')  

#sys.exit()  

# Produce and save a Hovmoller plot of solution (temp. plot of u(x,t) in the (x,t) plane)
try:    
    Xdata, Tdata= np.meshgrid(xdata, tdata)
    
    plt.figure()
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    try:
    
        if option_number == 1: 
        
            my_levels = np.linspace(-1.4,1.4,200)
            
        if option_number == 2: 
        
            my_levels = np.linspace(-1.05,1.7,200)
    
        CS = plt.contourf(Xdata, Tdata, udata , cmap=cmo.curl, levels=my_levels)
        
    except ValueError:
        pass
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    CB.set_label(r"$\varphi(t,x)$", fontsize=23)
    
    plt.xlim([-L, L])
    plt.ylim([0., T])
    
    plt.xlabel("$x$", fontsize=23)
    plt.ylabel("$t$", fontsize=23)
    plt.tick_params(axis='x', which='both', top='off')
    plt.xticks(fontsize=12, rotation=90)
    plt.tick_params(axis='y', which='both', right='off')
    plt.yticks(fontsize=12, rotation=90)
    CB.ax.tick_params(labelsize=12) 
    
    plt.tight_layout()
    
    #fignamepng = 'collision' +'.png'
    #fignamepng = 'stability' +'.png'
    #plt.savefig(fignamepng, bbox_inches='tight', dpi=600)
    
    plt.show()
    sys.exit()
    
except IOError:
    pass 



