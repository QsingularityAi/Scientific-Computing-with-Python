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
from functions import PhiEq_1D
from functions import IntegrationScheme
from functions import DBC_2ghostpoints_2D
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

def update_mesh_function(r,L,eps,ds):
    xm, ym = mesh(L,ds)
    phi = PhiCircle(xm,ym,r,eps)
    return xm,ym,phi


def AllenCahn_FD(phi,eps,ds,dt,L,Tmax,label,print_steps=False):
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
    nplot=100
    #i
    xfit=np.linspace(0,L,int(xm.shape[0]/2))
    for i in range(0,ntsteps):

        #optional printout of the progress
        if(print_steps==True):
            print(i,'/',ntsteps)

        #iteration
        if(i==0):   
            u=phi  #initialization of phi
        else:       
            u=IntegrationScheme(u,eps,ds,dt)
   
        if(label=='r_t'):

            #I exploit the function PhiEq_1D(x,r,eps) to fit the data and extract the best r associated to the current phi distribution
            #As input i take (with no loss of generality the positive x axis and the associated phi values
            #plt.plot(xfit,yfit)
            #plt.plot(xfit,PhiEq_1D(xfit,popt[0],popt[1]))
            #plt.show()
            #print(popt)

            yfit=u[int(xm.shape[0]/2)-1,int(xm.shape[0]/2):int(xm.shape[0])]
            if(i==0):
                popt0, pcov0 = curve_fit(PhiEq_1D, xfit, yfit)
                popt=popt0
            else:
                popt, pcov = curve_fit(PhiEq_1D, xfit, yfit)
            vecout[i]=popt[0]
        #set output type
        if(label=='n_plot'):
            if((i) % nplot == 0):
                plt.figure(1,figsize=[8,6])
                plt.rc('text', usetex=True)
                plt.rc('font', family='serif',size=16)
                plt.title('Progress: '+"{:.2f}".format(t)+'/'+"{:.2f}".format(Tmax))
                val=1
                levelvec=np.linspace(0,val,256,endpoint=False)
                plt.contourf(xm,ym,u[:,:],levels = np.append(levelvec,[val]),cmap='seismic')
                cbar=plt.colorbar()
                cbar.set_ticks(np.append(np.linspace(0,val,10,endpoint=False),val))
                plt.ylabel('$y$',size=20)
                plt.xlabel('$x$',size=20)
                plt.savefig('AllenCahn_MCF'+str(j)+'.png')
                #plt.show()
                plt.close()
                j=j+1

        t=t+dt
    #
    return vecout

#Radius of the Circular domain Omega
r=4.0                              
L=2*r #Size L_OmegaDDA

#Parameters for the Initial Condition
eps=1.0

#Convergence to Exact solution for different ds -> r_n
Enable=0
if(Enable):
    default_plotting()
    plt.ylabel('$r(t)$',size=20)
    plt.xlabel('$t$',size=20)
    dt=0.0001
    T=((r**2)/2.0)/1000.0
    plt.plot(timevec(dt,T), ExactMCF(r,timevec(dt,T),1),'grey',linewidth=3,label='Exact $r(t)$')
    ds=0.1
    xm,ym,phi=update_mesh_function(r,L,eps,ds)
    e1=AllenCahn_FD(phi,eps,ds,dt,L,T,label='r_t',print_steps=True)
    plt.plot(timevec(dt,T),e1,'--r',linewidth=3,label='$\\Delta s=$'+str(ds))
    ds=0.2
    xm,ym,phi=update_mesh_function(r,L,eps,ds)
    e2=AllenCahn_FD(phi,eps,ds,dt,L,T,label='r_t',print_steps=True)
    plt.plot(timevec(dt,T),e2,':b',linewidth=3,label='$\\Delta s=$'+str(ds))
    ds=0.4
    xm,ym,phi=update_mesh_function(r,L,eps,ds)
    e3=AllenCahn_FD(phi,eps,ds,dt,L,T,label='r_t',print_steps=True)
    plt.plot(timevec(dt,T),e3,'-.g',linewidth=3,label='$\\Delta s=$'+str(ds))
    ds=0.8
    xm,ym,phi=update_mesh_function(r,L,eps,ds)
    e4=AllenCahn_FD(phi,eps,ds,dt,L,T,label='r_t',print_steps=True)
    plt.plot(timevec(dt,T),e4,'-y',linewidth=3,label='$\\Delta s=$'+str(ds))
    plt.title('Convergence varying $\\Delta s$ in terms of $r(t)$. $\\epsilon$='+str(eps)+" ; $\\Delta t=$"+str(dt))
    plt.legend(loc='best')
    plt.savefig('AllenCahn_gridsize'+str(eps)+'_1.png')
    plt.show()
    plt.close()

#Convergence to Exact solution for different ds -> Normalized radius
    default_plotting()
    plt.ylabel('$r(t)$',size=20)
    plt.xlabel('$t$',size=20)
    plt.plot(timevec(dt,T),ExactMCF(r,timevec(dt,T),1)/ExactMCF(r,0,1),'grey',linewidth=3,label='Exact $r(t)$')
    ds=0.1
    plt.plot(timevec(dt,T),e1/e1[0],'--r',linewidth=3,label='$\\Delta s=$'+str(ds))
    ds=0.2
    plt.plot(timevec(dt,T),e2/e2[0],':b',linewidth=3,label='$\\Delta s=$'+str(ds))
    ds=0.4
    plt.plot(timevec(dt,T),e3/e3[0],'-.g',linewidth=3,label='$\\Delta s=$'+str(ds))
    ds=0.8
    plt.plot(timevec(dt,T),e4/e4[0],'-y',linewidth=3,label='$\\Delta s=$'+str(ds))
    plt.title('Convergence varying $\\Delta s$ in terms of $r(t)/r(0)$. $\\epsilon$='+str(eps)+" ; $\\Delta t=$"+str(dt))
    plt.legend(loc='best')
    plt.savefig('AllenCahn_gridsize'+str(eps)+'_2.png')
    plt.show()
    plt.close()


#Convergence for \epsilon -> 0
Enable=1
if(Enable):
    default_plotting()
    plt.ylabel('$r(t)$',size=20)
    plt.xlabel('$t$',size=20)
    dt=0.0001
    ds=0.1
    T=((r**2)/2.0)/1000.0
    ExEx=ExactMCF(r,timevec(dt,T),1)
    eps0=0.25
    xm,ym,phi=update_mesh_function(r,L,eps0,ds)
    e0=AllenCahn_FD(phi,eps0,ds,dt,L,T,label='r_t',print_steps=True)
    eps1=0.5
    xm,ym,phi=update_mesh_function(r,L,eps1,ds)
    e1=AllenCahn_FD(phi,eps1,ds,dt,L,T,label='r_t',print_steps=True)
    eps2=1.0
    xm,ym,phi=update_mesh_function(r,L,eps2,ds)
    e2=AllenCahn_FD(phi,eps2,ds,dt,L,T,label='r_t',print_steps=True)
    eps3=2.0
    xm,ym,phi=update_mesh_function(r,L,eps3,ds)
    e3=AllenCahn_FD(phi,eps3,ds,dt,L,T,label='r_t',print_steps=True)
    eps4=3.0
    xm,ym,phi=update_mesh_function(r,L,eps4,ds)
    e4=AllenCahn_FD(phi,eps4,ds,dt,L,T,label='r_t',print_steps=True)
    print(e0[e0.shape[0]-1]/e0[0],e1[e1.shape[0]-1]/e1[0],e2[e2.shape[0]-1]/e2[0],e3[e3.shape[0]-1]/e3[0],e4[e4.shape[0]-1]/e4[0])
    plt.plot(timevec(dt,T),e0/e0[0],label='e0')
    plt.plot(timevec(dt,T),e1/e1[0],label='e1')
    plt.plot(timevec(dt,T),e2/e2[0],label='e2')
    plt.plot(timevec(dt,T),e3/e3[0],label='e3')
    plt.plot(timevec(dt,T),e4/e4[0],label='e4')
    plt.legend()
    plt.title('Convergence for $\\epsilon \\rightarrow 0$')
    plt.axis([0,T,e4[e4.shape[0]-1]/e4[0],1.0])
    plt.savefig('Convergence_eps_1.png')
    plt.show()
    epsV=np.array([eps1,eps2,eps3,eps4])
    drV=abs(np.array([min(e1/e1[0]),min(e2/e2[0]),min(e3/e3[0]),min(e4/e4[0])])-(min(e0/e0[0])))
    default_plotting()
    plt.ylabel('$|r_\\epsilon(t)-r_{\\rm ref}|$',size=20)
    plt.xlabel('$\\epsilon$',size=20)
    plt.loglog(epsV,drV,'--sb')
    annotation.slope_marker((1, 1E-4), (2, 1))
    plt.title('Convergence for $\\epsilon \\rightarrow 0$')
    plt.savefig('Convergence_eps_2.png')
    plt.show()
    plt.close()



#Full Dynamics
Enable=0
if(Enable):
    ds=0.2
    dt=0.001
    eps=1.0
    T=((r**2)/2.0)/1.0
    xm,ym,phi=update_mesh_function(r,L,eps,ds)
    AllenCahn_FD(phi,eps,ds,dt,L,T,label='n_plot',print_steps=True)


