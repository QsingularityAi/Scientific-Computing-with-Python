#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib.ticker import LinearLocator
import time
from itertools import cycle
from functions_FT import IniRand
from functions_FT import Energy
from functions_FT import AllenCahn2D_FT
from functions_FT import CahnHilliard2D_FT


def default_plotting():
    plt.figure(1,figsize=[10,8])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=18)


# Discretization (x)
M=64
N =2*M                     
h = 2*np.pi/N            
x = h*np.arange(0,N)    
y = h*np.arange(0,N)    

xm, ym = np.meshgrid(x, y)

eps=2*np.pi/20.0

TM=30.0

#Initial Condition and parameters
C = 1.0        
#seed for one final phase (AC) = 129864 

# Discrete set of k 
I = complex(0,1)
kx = np.array([I*y for y in list(range(0,M)) + list([0]) + list(range(-M+1,0)) ])
ky = np.array([I*y for y in list(range(0,M)) + list([0]) + list(range(-M+1,0)) ])

kx2 = np.zeros((N,N), dtype=complex)
ky2 = np.zeros((N,N), dtype=complex)
for i in range(N):
    for j in range(N):
        kx2[i,j]=kx[i]**2
        ky2[i,j]=ky[j]**2



#Allen Cahn 2D by Fourier Method (flat_surf)
Enable=1
if(Enable): 
    default_plotting()
    uini = IniRand(x,y,0.4,0.6,184620)
    u_hatAC = np.fft.fft2(uini) 
    uAC=uini
    # Discretizazion (t)
    t=0
    dt = 1e-2
    tmax=TM
    ntstep=int(round(tmax/dt))
    EnergyAC=np.zeros(ntstep)
    IntU_AC=np.zeros(ntstep)
    tvec_AC=np.zeros(ntstep)
    for n in range(0,ntstep):
        print(t)
        if(n%10==0):
            plt.contourf(xm,ym,uAC,50,cmap="seismic",vmin=0.0,vmax=1.0)
            plt.colorbar()
            plt.title('Progress: '+"{:.2f}".format(t)+'/'+"{:.2f}".format(dt*ntstep))
            plt.ylabel('$y$',size=18)
            plt.xlabel('$x$',size=18)
            plt.savefig("allencahn_flatsurf_FT"+str(int(n/10))+".png")
            plt.close()
        #Integration scheme
        unl2=np.array(uAC**2, dtype=complex)
        unl3=np.array(uAC**3, dtype=complex)
        unl2_hat=np.fft.fft2(unl2) 
        unl3_hat=np.fft.fft2(unl3) 
        u_hatAC=AllenCahn2D_FT(u_hatAC,unl2_hat,unl3_hat,kx2,ky2,C,dt,eps)
        uAC=np.real(np.fft.ifft2(u_hatAC))
        EnergyAC[n]=Energy(uAC,eps,h)
        IntU_AC[n]=np.average(uAC)
        tvec_AC[n]=t
        t=t+dt;
 
#Allen Cahn 2D by Fourier Method (single_phase)
Enable=1
if(Enable): 
    uini = IniRand(x,y,0.4,0.6,129864)
    u_hatAC2 = np.fft.fft2(uini) 
    uAC2=uini
    # Discretizazion (t)
    t=0
    dt = 1e-2
    tmax=TM
    ntstep=int(round(tmax/dt))
    EnergyAC2=np.zeros(ntstep)
    IntU_AC2=np.zeros(ntstep)
    tvec_AC2=np.zeros(ntstep)
    for n in range(0,ntstep):
        print(t)
        if(n%10==0):
            plt.contourf(xm,ym,uAC2,50,cmap="seismic",vmin=0.0,vmax=1.0)
            plt.colorbar()
            plt.title('Progress: '+"{:.2f}".format(t)+'/'+"{:.2f}".format(dt*ntstep))
            plt.ylabel('$y$',size=18)
            plt.xlabel('$x$',size=18)
            plt.savefig("allencahn_singlephase_FT"+str(int(n/10))+".png")
            plt.close()
        #Integration scheme
        unl2=np.array(uAC2**2, dtype=complex)
        unl3=np.array(uAC2**3, dtype=complex)
        unl2_hat=np.fft.fft2(unl2) 
        unl3_hat=np.fft.fft2(unl3) 
        u_hatAC2=AllenCahn2D_FT(u_hatAC2,unl2_hat,unl3_hat,kx2,ky2,C,dt,eps)
        uAC2=np.real(np.fft.ifft2(u_hatAC2))
        EnergyAC2[n]=Energy(uAC2,eps,h)
        IntU_AC2[n]=np.average(uAC2)
        tvec_AC2[n]=t
        t=t+dt;
  
#Cahn Hilliard 2D by Fourier Method 
Enable=1
if(Enable):
    default_plotting()
    uini = IniRand(x,y,0.4,0.6,129864)
    u_hatCH = np.fft.fft2(uini) 
    uCH=uini
    # Discretizazion (t)
    t=0
    dt = 1e-2
    tmax=TM
    ntstep=int(round(tmax/dt))
    EnergyCH=np.zeros(ntstep)
    IntU_CH=np.zeros(ntstep)
    tvec_CH=np.zeros(ntstep)
    for n in range(0,ntstep):
        print(t)
        if(n%10==0):
            plt.contourf(xm,ym,uCH,50,cmap="seismic",vmin=0.0,vmax=1.0)
            plt.colorbar()
            plt.title('Progress: '+"{:.2f}".format(t)+'/'+"{:.2f}".format(dt*ntstep))
            plt.ylabel('$y$',size=18)
            plt.xlabel('$x$',size=18)
            plt.savefig("cahn_hilliard_FT"+str(int(n/10))+".png")
            plt.close()
        #Integration scheme
        unl2=np.array(uCH**2, dtype=complex)
        unl3=np.array(uCH**3, dtype=complex)
        unl2_hat=np.fft.fft2(unl2) 
        unl3_hat=np.fft.fft2(unl3) 
        u_hatCH=CahnHilliard2D_FT(u_hatCH,unl2_hat,unl3_hat,kx2,ky2,C,dt,eps)
        uCH=np.real(np.fft.ifft2(u_hatCH))
        EnergyCH[n]=Energy(uCH,eps,h)
        IntU_CH[n]=np.average(uCH)
        tvec_CH[n]=t
        t=t+dt;


#Plots integrals of phi and energies over time/nstep
Enable=1
if(Enable):
    default_plotting()
    plt.plot(tvec_CH,EnergyCH,'--b',linewidth=3,label="$F[\\varphi(t)]$ AC (flat)")
    plt.plot(tvec_AC,EnergyAC,'--g',linewidth=3,label="$F[\\varphi(t)]$ AC (single)")
    plt.plot(tvec_AC2,EnergyAC2,'--r',linewidth=3,label="$F[\\varphi(t)]$ CH")
    plt.ylabel('$F[\\varphi(t)]$',size=20)
    plt.xlabel('$t$',size=20)
    plt.title('Comparison of Energies over time')
    plt.legend(loc="best")
    plt.savefig("Energies.png")
    plt.close()
    #
    default_plotting()
    plt.plot(tvec_CH,IntU_CH,'--b',linewidth=3,label="$F[\\varphi(t)]$ AC (flat)")
    plt.plot(tvec_AC,IntU_AC,'--g',linewidth=3,label="$F[\\varphi(t)]$ AC (single)")
    plt.plot(tvec_AC2,IntU_AC2,'--r',linewidth=3,label="$F[\\varphi(t)]$ CH")
    plt.ylabel('$(1/|\\Omega|)\int\\varphi (t) d \\Omega$',size=20)
    plt.xlabel('$t$',size=20)
    plt.title('Comparison of $(1/|\\Omega|)\int\\varphi d \\Omega$ over time')  
    plt.legend(loc="best")
    plt.savefig("INTPHI.png")
    plt.close()

