# Code for simulating solutions of Schrodinger by a second-oder finite difference method (FD). We use homogeneous Dirichlet boundary conditions on the wavefunction, so this is a "particle in a large box" playing the role of a particle moving on the real line. 

import numpy as np

from scipy.linalg import lu_factor, lu_solve
from scipy import sparse
from scipy.integrate import simps # for testing conservation of L2 mass. 

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
T= 27. 

# Prescribe space and time stepsize
dt = 0.1
dx = 0.09

# get stability-type parameter "r" appearing in finite difference formulation
r = 0.5*dt/(dx*dx)

# Number of nodes in space and steps in time
Nt = np.int(T/dt)
Nx = np.int(2.*L/dx) # includes boundary points

# Define "discretized domain" vector ie. points where interior solution is sampled
xdata= np.linspace(-L,L, Nx)

# Define vector of time sample points
tdata= np.linspace(0,T, Nt+1) # +1 bcz of initial state

# Initialize solution matrix
udata = np.zeros([Nt+1, Nx], dtype=float)
udata = udata +0j

# Define the potential V(x) the particle experiences. These is done by changing the 
# "option_number" keyword

option_number = 2

if option_number == 1: 

    # Option 1: free particle (V=0)
    V = np.zeros(Nx, dtype = float)
    
elif option_number == 2:

    # Option 2: sech^2 potential (AKA the Poschl-Teller potential)
    V_centre = 0. 
    V_scale = 1.6

    def sechsquared(x):
        return np.cosh((x-V_centre))**(-2.)
   
    V = -0.5*V_scale*(V_scale+1) * sechsquared(xdata)

# Define initial conditions as a function of x
# Default is a Gaussian modulating a single-mode wave

x0 = -35. # initial mean position of particle (centre of modulating Gaussian)

start=time.time()
def initial_state(x):
   amp = 1.
   width = 0.02
   state = amp*np.exp(-width*(np.power(x-x0,2)))*(np.cos(0.5*np.pi*(x-x0))- np.sin(0.5*np.pi*(x-x0))*1j)
   return state
    
# Set asymptotic values of field
u_inf = 0 + 0j
u_neginf = 0 + 0j

# Set initial state in solution array  
udata[0][:] = initial_state(xdata)

# set boundary contribution correctly   
udata[0][-1]= u_inf
udata[0][0]= u_neginf

# Prescription of the physical + computational parameters is now complete, and we can move on to defining and implementing the FD scheme

# Generate 2nd order finite difference approx of (d/dx)^2
# Uses a routine from the sparse matrix package (need to make sure you import it in the preamble!)

diagA=[np.ones([1,Nx-1], dtype=float), -2*np.ones([1,Nx], dtype=float) ,np.ones([1,Nx-1], dtype=float)]
A=sparse.diags(diagA, [-1,0,1], shape=[Nx,Nx]).toarray()

A[0][0]= 1
A[0][1]= 0

A[-1][-1]= 1
A[-1][-2]= 0

A = A+ 0j # turn A into matrix of complex numbers

start=time.time()

# Form the matrix B that needs to be inverted at each time step. 
B =  (0+1j)*np.identity(Nx) - 0.5*dt*sparse.diags([V], [0], shape=[Nx,Nx]).toarray() - r*A

B[0][0]= 1 +0j
B[0][1]= 0 +0j

B[-1][-1]= 1+0j
B[-1][-2]= 0+0j

# Find LU factorization of B for efficiency later on.
lu, piv = lu_factor(B)

# Define the matrix C we need to multiply against solution at each time step. 
C =  (0+1j)*np.identity(Nx) + 0.5*dt*sparse.diags([V], [0], shape=[Nx,Nx]).toarray() + r*A

C[0][0]= 1+0j
C[0][1]= 0+0j

C[-1][-1]= 1+0j
C[-1][-2]= 0+0j

# Define time loop.  
for m in range(1,Nt+1): 

    RHS = np.dot(C, udata[m-1][:])
    
    udata[m][:]= lu_solve((lu,piv), RHS.T).T
    
    udata[m][-1]= u_inf 
    udata[m][0]= u_neginf 
    
    # Uncomment to see movie while code runs
    """
    plt.clf()
    plt.plot(xdata, np.real(udata[m][:]), '-',color='xkcd:ocean blue', linewidth=2)
    plt.xlim(-L,L)
    plt.ylim(-1.5, 1.5)
    plt.grid('on')
    plt.pause(0.001)
    plt.draw()
    """
      
end=time.time()  

runtime = end-start

print('Solve Time =', f'{runtime:.4g}', 'seconds')   

#sys.exit()  

# Produce and save a Hovmoller plot of solution (temp. plot of u(t,x) in the (x,t) plane)
try:    
    Xdata, Tdata= np.meshgrid(xdata, tdata)
    
    plt.figure()
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    try:
        CS = plt.contourf(Xdata, Tdata, np.real(udata) , cmap=cmo.haline, levels=np.linspace(-1.2,1.35,200))
        
    except ValueError:
        pass
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    CB.set_label(r"$\mathrm{Re} \ u(t,x)$", fontsize=23)
    
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
    
    #fignamepng = 're_u' +'.png'
    #plt.savefig(fignamepng, bbox_inches='tight', dpi=600)
    plt.show()
    #sys.exit()
    
except IOError:
    pass 


# Produce and save a Hovmoller plot of solution (temp. plot of u(x,t) in the (x,t) plane)
try:    
    Xdata, Tdata= np.meshgrid(xdata, tdata)
    
    plt.figure()
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    try:
        CS = plt.contourf(Xdata, Tdata, np.imag(udata) , cmap=cmo.haline, levels=np.linspace(-1.3,1.35,200)) 
        
    except ValueError:
        pass
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    CB.set_label(r"$\mathrm{Im} \ u(t,x)$", fontsize=23)
    
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
    
    #fignamepng = 'im_u' +'.png'
    #plt.savefig(fignamepng, bbox_inches='tight', dpi=600)
    plt.show()
    #sys.exit()
    
except IOError:
    pass 

# Now, we see how well our simulation conserves the L2 norm of the solution
#"""
mass =np.zeros(Nt+1)

for m in np.arange(0,Nt+1):
    
    mass[m] = simps(np.absolute(udata[m,:])**2, xdata) #numerical integration routine
    
fig, ax =plt.subplots()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
    
mass_err = np.abs(mass-mass[0])/mass[0]
plt.plot(tdata, mass_err, color='xkcd:moss green', linewidth='2')

plt.xlim([0., T])

plt.xlabel(r"$t$", fontsize=26)

#from matplotlib.ticker import FormatStrFormatter
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.ylabel(r"Relative Mass Error ", fontsize=26)
plt.tick_params(axis='x', which='both', top='off')
plt.xticks(fontsize=20, rotation=90)
plt.tick_params(axis='y', which='both', right='off')
plt.yticks(fontsize=20, rotation=90)
ax.locator_params(axis='y', nbins=6)


plt.tight_layout()
    
#fignamepng = 'mass_error'+'.png'
#plt.savefig(fignamepng, bbox_inches='tight', dpi=600)

plt.show()
#"""


