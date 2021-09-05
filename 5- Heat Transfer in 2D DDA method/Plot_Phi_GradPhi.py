#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from functions import GaussIni
from functions import PhiCircle
from functions import NormGrad

#Parameters from command line (with exception)
try:
    r = float(sys.argv[1])
    L_OmegaDDA = float(sys.argv[2])
    eps = float(sys.argv[3])
except:
    L_OmegaDDA = 10.0
    r = 5.0
    eps = 0.5

# Discretization (x,y)
ds=0.05
x=np.arange(-(L_OmegaDDA), (L_OmegaDDA), ds)
y=x
xm, ym = np.meshgrid(x, y)

#Initial Condition and parameters
A=1
B=0.5
phi = PhiCircle(xm,ym,r,eps)
NormGradPhi = NormGrad(phi,ds) 

plt.figure(1,figsize=[8,6])
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=16)
plt.title('$\\phi(\\mathbf{x})$ [$L_{\\Omega_{\rm DDA}}=$'+str(L_OmegaDDA)+', $r=$'+str(r)+', $\\epsilon=$'+str(eps)+']')
levelvec=np.linspace(0,1,50,endpoint=False)
plt.contourf(xm,ym,phi,levels = np.append(levelvec,[1]),cmap='seismic')
cbar=plt.colorbar()
cbar.set_ticks(np.append(np.linspace(0,1,10,endpoint=False),1.0))
plt.ylabel('$y$',size=20)
plt.xlabel('$x$',size=20)
plt.savefig('phi_eps'+str(eps)+'.png')
#plt.show()
plt.close()

plt.figure(1,figsize=[8,6])
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=16)
plt.title('$|\\nabla \\phi(\\mathbf{x})|$ [$L_{\\Omega_{\rm DDA}}$='+str(L_OmegaDDA)+', $r=$'+str(r)+', $\\epsilon=$'+str(eps)+']')
val=4
levelvec=np.linspace(0,val,50,endpoint=False)
plt.contourf(xm,ym,NormGradPhi,levels = np.append(levelvec,[val]),cmap='seismic')
cbar=plt.colorbar()
cbar.set_ticks(np.append(np.linspace(0,val,10,endpoint=False),val))
plt.ylabel('$y$',size=20)
plt.xlabel('$x$',size=20)
plt.savefig('Gradphi_eps'+str(eps)+'.png')
#plt.show()
plt.close()




