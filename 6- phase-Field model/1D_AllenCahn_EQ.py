#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import sys
from scipy.optimize import curve_fit
from functions import ExactMCF
from functions import PhiCircle
from functions import PhiEq_x0
from functions import IntegrationScheme1D
from functions import DBC_2ghostpoints_1D
from mpltools import annotation
from numpy import random

def default_plotting():
    plt.figure(1,figsize=[10,8])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=18)

def timevec(dt,T):
    return np.linspace(0,T,int(T/dt))


def AllenCahn_FD(phi,eps,ds,dt,L,Tmax,print_steps=False):
    #
    #label select the information to extract, namely
    #'n_plot': plot some plots during the time iteration (aux_variable is the number of plots)
    #'max_over_time': max(u) as function of t
    #'int_u': integral of u as function of t
    #
    t=0
    j=0
    ntsteps=int(Tmax/dt)
    vecout=np.zeros(ntsteps)
    #i
    for i in range(0,ntsteps):

        #optional printout of the progress
        if(print_steps==True):
            print(i,'/',ntsteps)

        #iteration
        if(i==0):   
            u=phi  #initialization of phi
        else:       
            u=IntegrationScheme1D(u,eps,ds,dt)
            u=DBC_2ghostpoints_1D(u,1,0)

    return u


#1D domain size
L=10

#Parameters for the Initial Condition
eps=2.0

#Numerical Parameters
dt=0.001 
ds=eps/10.0

#Spatial discretization
x=np.arange(-(L/2.)-2*ds,(L/2)+2*ds, ds)

phiLinear=0.5-x/L
philargeeps=PhiEq_x0(x,0,4*eps)
phismalleps=PhiEq_x0(x,0,0.25*eps)
sd=20584
random.seed(sd)
phiRandom=random.random(x.shape[0])
phiRandom=DBC_2ghostpoints_1D(phiRandom,1,0)
#From Linear distribution To Equilibrium
Enable=0
if(Enable):
    default_plotting()
    plt.ylabel('$\\varphi(x)$',size=20)
    plt.xlabel('$x$',size=20)
    phi=phiLinear
    plt.plot(x, PhiEq_x0(x,0,eps),'grey',linewidth=3,label='$\\epsilon_{\\rm eq}$')
    plt.plot(x, phi,'--b',linewidth=3,label='Ini. $\\varphi$ Linear')
    T=0.05
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--r',linewidth=3,label='$T=$'+str(T))
    T=0.1
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--g',linewidth=3,label='$T=$'+str(T))
    T=0.2
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--c',linewidth=3,label='$T=$'+str(T))
    T=0.4
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--y',linewidth=3,label='$T=$'+str(T))
    plt.title('From Linear to $\\varphi$ with $\\epsilon=\\epsilon_{\\rm eq}$')
    plt.legend(loc='best')
    plt.axis([-L/2,L/2,-0.1,1.1])
    plt.savefig('Linear_To_Phi'+str(eps)+'.png')
    plt.show()
    plt.close()


#From small eps To Equilibrium
Enable=0
if(Enable):
    default_plotting()
    plt.ylabel('$\\varphi(x)$',size=20)
    plt.xlabel('$x$',size=20)
    phi=phismalleps
    plt.plot(x, PhiEq_x0(x,0,eps),'grey',linewidth=3,label='$\\epsilon_{\\rm eq}$')
    plt.plot(x, phi,'--b',linewidth=3,label='Ini. $\epsilon < \\epsilon_{\\rm eq}$')
    T=0.02
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--r',linewidth=3,label='$T=$'+str(T))
    T=0.05
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--g',linewidth=3,label='$T=$'+str(T))
    T=0.1
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--c',linewidth=3,label='$T=$'+str(T))
    T=0.2
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--y',linewidth=3,label='$T=$'+str(T))
    plt.title('$\\epsilon_{\\rm ini} < \\epsilon_{\\rm eq}$')
    plt.legend(loc='best')
    plt.axis([-2.2,2.2,-0.1,1.1])
    plt.savefig('Smalleps_To_Phi'+str(eps)+'.png')
    plt.show()
    plt.close()


#From big eps To Equilibrium
Enable=0
if(Enable):
    default_plotting()
    plt.ylabel('$\\varphi(x)$',size=20)
    plt.xlabel('$x$',size=20)
    phi=philargeeps
    plt.plot(x, PhiEq_x0(x,0,eps),'grey',linewidth=3,label='$\\epsilon_{\\rm eq}$')
    plt.plot(x, phi,'--b',linewidth=3,label='Ini. $\epsilon > \\epsilon_{\\rm eq}$')
    T=0.05
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--r',linewidth=3,label='$T=$'+str(T))
    T=0.1
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--g',linewidth=3,label='$T=$'+str(T))
    T=0.2
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--c',linewidth=3,label='$T=$'+str(T))
    T=0.4
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--y',linewidth=3,label='$T=$'+str(T))
    plt.title('$\\epsilon_{\\rm ini} > \\epsilon_{\\rm eq}$')
    plt.legend(loc='best')
    plt.axis([-L/2,L/2,-0.1,1.1])
    plt.savefig('Largeeps_To_Phi'+str(eps)+'.png')
    plt.show()
    plt.close()

#From Random distribution To Equilibrium
Enable=1
if(Enable):
    default_plotting()
    plt.ylabel('$\\varphi(x)$',size=20)
    plt.xlabel('$x$',size=20)
    phi=phiRandom
    plt.plot(x, PhiEq_x0(x,0,eps),'grey',linewidth=3,label='$\\epsilon_{\\rm eq}$')
    plt.plot(x, phi,'--b',linewidth=1,label='Ini. $\epsilon > \\epsilon_{\\rm eq}$')
    T=0.05
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--r',linewidth=3,label='$T=$'+str(T))
    T=0.1
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--g',linewidth=3,label='$T=$'+str(T))
    T=0.2
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--c',linewidth=3,label='$T=$'+str(T))
    T=0.4
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--y',linewidth=3,label='$T=$'+str(T))
    T=2.0
    plt.plot(x, AllenCahn_FD(phi,eps,ds,dt,L,T,print_steps=True),'--k',linewidth=3,label='$T=$'+str(T))
    plt.title('Ini. Random. Seed = '+str(sd)+'; $\\Delta s$='+str(ds))
    plt.legend(loc='best')
    plt.axis([-L/2,L/2,-0.1,1.1])
    plt.savefig('RandomToPhi'+str(eps)+'seed'+str(sd)+'.png')
    plt.show()
    plt.close()

