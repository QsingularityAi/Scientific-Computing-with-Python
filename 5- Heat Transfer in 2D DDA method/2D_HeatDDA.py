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


def heat_DDA(uini,phi,f,g,ds,dt,Tmax,label='n_plot',aux_variable=0,print_steps=False):
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
    if(label=='n_plot'):
        nplot=aux_variable
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

        #set output type
        if(label=='max_over_time'):
            vecout[i]=u.max()
        if(label=='int_u'):
            vecout[i]=sum(sum(u[:,:]*phi[:,:]))*ds**2
        if(label=='n_plot'):
            if((i) % nplot == 0):
                plt.figure(1,figsize=[8,6])
                plt.rc('text', usetex=True)
                plt.rc('font', family='serif',size=16)
                plt.title('Progress: '+"{:.2f}".format(t)+'/'+"{:.2f}".format(Tmax))
                val=1
                levelvec=np.linspace(0,val,256,endpoint=False)
                plt.contourf(xm,ym,abs(phi[:,:]*u[:,:]),levels = np.append(levelvec,[val]),cmap='hot')
                #plt.contourf(xm,ym,phi[:,:]*u[:,:],256,cmap='hot')
                cbar=plt.colorbar()
                cbar.set_ticks(np.append(np.linspace(0,val,10,endpoint=False),val))
                plt.ylabel('$y$',size=20)
                plt.xlabel('$x$',size=20)
                plt.savefig('Heat_DDA'+str(j)+'.png')
                #plt.show()
                plt.close()
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
eps=1.0

#Plot Solution with different ds
Enable=0
if(Enable):
    default_plotting()
    plt.ylabel('max$[u(x,t)]$',size=20)
    plt.xlabel('$t$',size=20)
    f=0
    g=0
    dt=0.001
    T=0.1
    ds=0.1
    xm,ym,u,phi=update_mesh_function(r,L,eps,ds,A,B)
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,'max_over_time',0,print_steps=True),'--r',linewidth=3,label='$\\Delta s=$'+str(ds))
    ds=0.2
    xm,ym,u,phi=update_mesh_function(r,L,eps,ds,A,B)
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,'max_over_time',0,print_steps=True),':b',linewidth=3,label='$\\Delta s=$'+str(ds))
    ds=0.5
    xm,ym,u,phi=update_mesh_function(r,L,eps,ds,A,B)
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,'max_over_time',0,print_steps=True),'-.g',linewidth=3,label='$\\Delta s=$'+str(ds))
    ds=1.0
    xm,ym,u,phi=update_mesh_function(r,L,eps,ds,A,B)
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,'max_over_time',0,print_steps=True),'-y',linewidth=3,label='$\\Delta s=$'+str(ds))
    plt.title('Dependence of the solution on the spatial discretization')
    plt.legend(loc='best')
    plt.savefig('HeatDDASol_gridsize_eps'+str(eps)+'.png')
    plt.show()
    plt.close()


#Plot Solution with different dt
Enable=0
if(Enable):
    default_plotting()
    ds=0.2
    xm,ym,u,phi=update_mesh_function(r,L,eps,ds,A,B)
    f=0
    g=0
    T=0.1
    dt=0.0001
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,'max_over_time',0,print_steps=True),'--r',linewidth=3,label='$\\Delta t=$'+str(dt))
    dt=0.001
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,'max_over_time',0,print_steps=True),':b',linewidth=3,label='$\\Delta t=$'+str(dt))
    dt=0.005
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,'max_over_time',0,print_steps=True),'-.g',linewidth=3,label='$\\Delta t=$'+str(dt))
    dt=0.01
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,'max_over_time',0,print_steps=True),'-y',linewidth=3,label='$\\Delta t=$'+str(dt))
    dt=0.02
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,'max_over_time',0,print_steps=True),'-..c',linewidth=3,label='$\\Delta t=$'+str(dt))
    plt.title('Dependence of the solution on the timestep')
    plt.ylabel('max$[u(x,t)]$',size=20)
    plt.xlabel('t',size=20)
    plt.legend(loc='best')
    plt.savefig('HeatDDASol_timestep.png')
    plt.show()
    plt.close()


#Plot C_\epsilon (t)
Enable=0
if(Enable):
    default_plotting()
    ds=0.2
    dt=0.001
    f=0
    g=0
    T=0.1
    eps=1.0
    xm,ym,u,phi=update_mesh_function(r,L,eps,ds,A,B)
    phi = PhiCircle(xm,ym,r,eps)
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,'int_u',0,print_steps=True),'--r',linewidth=3,label='$\\epsilon=$'+str(eps))
    eps=2.0
    phi = PhiCircle(xm,ym,r,eps)
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,'int_u',0,print_steps=True),':b',linewidth=3,label='$\\epsilon=$'+str(eps))
    eps=4.0
    phi = PhiCircle(xm,ym,r,eps)
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,'int_u',0,print_steps=True),'-.g',linewidth=3,label='$\\epsilon=$'+str(eps))
    eps=8.0
    phi = PhiCircle(xm,ym,r,eps)
    plt.plot(timevec(dt,T), heat_DDA(u,phi,f,g,ds,dt,T,'int_u',0,print_steps=True),'-y',linewidth=3,label='$\\epsilon=$'+str(eps))
    plt.title('$C_\\epsilon (t)$ for different $\\epsilon$')
    plt.ylabel('$C_\\epsilon$',size=20)
    plt.xlabel('t',size=20)
    plt.legend(loc='best')
    plt.savefig('HeatDDASol_eps.png')
    plt.show()
    plt.close()

#Plot Convergence of C_\epsilon for \epsilon -> 0
Enable=0
if(Enable):
    default_plotting()
    ds=0.02
    dt=0.001
    f=0
    g=0
    T=0.002
    eps0=0.05
    xm,ym,u,phi=update_mesh_function(r,L,eps0,ds,A,B)
    vec_int0=heat_DDA(u,phi,f,g,ds,dt,T,'int_u',0,print_steps=True)
    eps1=0.1
    xm,ym,u,phi=update_mesh_function(r,L,eps1,ds,A,B)
    vec_int1=heat_DDA(u,phi,f,g,ds,dt,T,'int_u',0,print_steps=True)
    eps2=0.2
    phi = PhiCircle(xm,ym,r,eps2)
    vec_int2=heat_DDA(u,phi,f,g,ds,dt,T,'int_u',0,print_steps=True)
    eps3=0.4
    phi = PhiCircle(xm,ym,r,eps3)
    vec_int3=heat_DDA(u,phi,f,g,ds,dt,T,'int_u',0,print_steps=True)
    eps4=0.8
    phi = PhiCircle(xm,ym,r,eps4)
    vec_int4=heat_DDA(u,phi,f,g,ds,dt,T,'int_u',0,print_steps=True)
    epsV=np.array([eps1,eps2,eps3,eps4])
    intV=abs(np.array([np.average(vec_int1),np.average(vec_int2),np.average(vec_int3),np.average(vec_int4)])-np.average(vec_int0))
    plt.ylabel('$|C_\\epsilon-C_{0.05}|$',size=20)
    plt.xlabel('$\\epsilon$',size=20)
    plt.loglog(epsV,intV,'--sb')
    annotation.slope_marker((0.3, 1e-6), (2, 1))
    plt.title('Convergence for $\\epsilon \\rightarrow 0$')
    plt.savefig('Convergence_eps.png')
    plt.show()
    plt.close()


#Full Dynamics
Enable=1
if(Enable):
    ds=0.2
    dt=0.001
    T=10.0
    eps=1.0
    f=0
    g=0
    xm,ym,u,phi=update_mesh_function(r,L,eps,ds,A,B)
    heat_DDA(u,phi,f,g,ds,dt,T,'n_plot',100,print_steps=True)


