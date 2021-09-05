#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import sys
from functions_FM import IniSinCos
from functions_FM import ImplicitEuler2D_FT

plt.ion()

# Discretization (x,y)
M=32
N = 2*M                     
h = 2*math.pi/N            
x = h*np.arange(0,N)    
y = h*np.arange(0,N)    
xm, ym = np.meshgrid(x, y)

# Discretizazion (t)
t=0
dt = 5e-2
tmax=5.0

#Initial Condition and parameters
C = 0.5        
A=1
B=0.5
kx=1
ky=2
uini = IniSinCos(xm,ym,kx,ky,A,B)

# Discrete set of k_x, k_y
I = complex(0,1)
kx = np.array([I*y for y in list(range(0,M)) + list([0]) + list(range(-M+1,0)) ])
ky = np.array([I*y for y in list(range(0,M)) + list([0]) + list(range(-M+1,0)) ])
kx2 = np.zeros((N,N), dtype=complex)
ky2 = np.zeros((N,N), dtype=complex)
for i in range(N):
    for j in range(N):
        kx2[i,j] = kx[i]**2
        ky2[i,j] = ky[j]**2

fig=plt.figure(1,figsize=[10,8])
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=16)
ax = fig.gca(projection='3d')
surf=ax.plot_surface(xm, ym, uini, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-(A+B+0.01), (A+B+0.01))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)$|_t$')
tt=ax.text(1.8,1.8,3.0,'Progress: '+"{:.2f}".format(t)+'/'+"{:.2f}".format(tmax),fontsize=16)
ntstep=int(round(tmax/dt))
u_hat = np.zeros((N,N),dtype=complex)

#DECOMPOSE / ANALYSE
u_hat = np.fft.fft2(uini) 
#SOLVE
for n in range(ntstep):
    u_hat = ImplicitEuler2D_FT(u_hat,kx2,ky2,C,dt)
    #SYNTETHISE
    u = np.real(np.fft.ifft2(u_hat))	#ifft to have the solution in real space 
    t=t+dt;
    ax.collections.remove(surf)
    surf=ax.plot_surface(xm, ym, u, rstride=1, cstride=1, cmap=cm.coolwarm, vmin=uini.min(), vmax=uini.max(), linewidth=0, antialiased=False)
    tt.set_text('Progress: '+"{:.2f}".format(t)+'/'+"{:.2f}".format(tmax))
    plt.draw()
    plt.savefig('diff'+str(n)+'.png')
