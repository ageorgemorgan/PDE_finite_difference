# Code for simulating solutions of unit-speed wave equation by a second-oder finite difference method (FD). We use homogeneous Dirichlet boundary conditions to simplify the code. 

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

# Define initial conditions as a function of x
# Default is an initially stationary Gaussian bump

x0 = 0. # centre of Gaussian

def initial_state(x):
   amp = 1
   width = 0.02
   state = amp*np.exp(-width*(np.power(x-x0,2)))
   return state
   
def initial_speed(x): 
    speed = 0.
    return speed
    
# Set asymptotic values of field
u_inf = 0
u_neginf = 0

# Set initial state in solution array  
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

# Solve for solution at first time step using forward Euler applied to du/dt

udata[1][:] = udata[0][:] + dt*initial_speed(xdata) + (dt**2)*(((1./dx)**2)*(np.dot(A,udata[0][:])))

udata[1][-1]= u_inf
udata[1][0]= u_neginf

# Define time loop.  
for m in range(2,Nt+1): 
    
    udata[m][:]=  2.*udata[m-1][:] - udata[m-2][:] + (dt**2)*(((1./dx)**2)*(np.dot(A,udata[m-1][:])))
    
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
        CS = plt.contourf(Xdata, Tdata, udata , cmap=cmo.haline, levels=np.linspace(-0.1,1.01,200)) 
        
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
    
    #fignamepng = 'dalembert' +'.png'
    #plt.savefig(fignamepng, bbox_inches='tight', dpi=600)
    plt.show()
    sys.exit()
    
except IOError:
    pass 



