#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import sys
from functions import GaussIni
from functions import PhiCircle
from functions import IntegrationScheme
from functions import NormGrad
from mpltools import annotation


def default_plotting():
    plt.figure(1,figsize=[10,8])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=18)

def timevec(dt,T):
    return np.linspace(0,T,int(T/dt))

def mesh(L,ds):
    x=np.arange(-(L), (L), ds)
    y=x
    return np.meshgrid(x, y)

def update_mesh_function(r,L,eps,ds,A,B):
    xm, ym = mesh(L,ds)
    u = GaussIni(xm,ym,A,B)
    phi = PhiCircle(xm,ym,r,eps)
    return xm,ym,u,phi


def heat_DDA(uini,phi,f,g,ds,dt,Tmax,nplot,print_steps=False,enable_plotting=True):
    t=0
    j=0
    ntsteps=int(Tmax/dt)
    vecout=np.zeros(ntsteps)
    #
    for i in range(0,ntsteps):
        #optional printout of the progress
        if(print_steps==True):
            print(i,'/',ntsteps)

        #iteration
        if(i==0):   
            u=uini  #initialization of u
        else:       
            u=IntegrationScheme(u,phi,ds,dt,f,g)

        vv=u[:,:]*phi[:,:]
        vecout[i]=vv.max()
        if((i) % nplot == 0):
            plt.figure(2,figsize=[8,6])
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif',size=16)
            plt.title('Progress: '+"{:.2f}".format(t)+'/'+"{:.2f}".format(Tmax))
            val=2
            levelvec=np.linspace(-val,val,256,endpoint=False)
            plt.contourf(xm,ym,phi[:,:]*u[:,:],levels = np.append(levelvec,[val]),cmap='hot')
            cbar=plt.colorbar()
            cbar.set_ticks(np.append(np.linspace(-val,val,10,endpoint=False),val))
            plt.ylabel('$y$',size=20)
            plt.xlabel('$x$',size=20)
            plt.savefig('HeatDissipation_g'+str(round(g,2))+'_'+str(j)+'.png')
            plt.close(2)
            j=j+1
        t=t+dt
    #
    return vecout

#Radius of the Circular domain Omega
r=5.0                               
L=2*r #Size L_OmegaDDA

#Parameters for the Initial Condition
A=2.0
B=1.0
eps=2.0

#Plot Solution with different ds
Enable=1
if(Enable):
    default_plotting()
    plt.ylabel('max$[u(\mathbf{x},t)\phi(\mathbf{x})]$',size=20)
    plt.xlabel('$t$',size=20)
    ds=0.4 
    dt=0.01 
    T=20.0
    xm,ym,f,phi=update_mesh_function(r,L,eps,ds,A,B)
    u=np.zeros(xm.shape)
    nplot=100
    g=0
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,nplot,print_steps=True,enable_plotting=False),':b',linewidth=3,label='$g=$'+str(g))
    g=-sum(sum(f))/sum(sum(NormGrad(phi,ds)))
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,nplot,print_steps=True,enable_plotting=False),'--r',linewidth=3,label='$g=-\int_{\\Omega_{\\rm DDA}} f(\mathbf{x}) /\int_{\\Omega_{\\rm DDA}} |\\nabla \\phi|\\approx$'+str(round(g, 2)))
    g=-1
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,nplot,print_steps=True,enable_plotting=False),'-g',linewidth=3,label='$g=$'+str(g))
    #plt.title('Dissipation')
    plt.legend(loc='best')
    plt.savefig('DISSIPATION'+str(eps)+'.png')
    plt.show()
    plt.close(1)


