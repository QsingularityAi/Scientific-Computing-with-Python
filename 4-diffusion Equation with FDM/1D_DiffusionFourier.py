#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import cycle
from functions_FM import u_exact
from functions_FM import ExplicitEulerFT
from functions_FM import ImplicitEulerFT
from mpltools import annotation

def fourier_method(u_ini,tmax,dt,label='FE'):
    t=0
    alpha=0.5
    ntstep=int(round(tmax/dt))
    #DECOMPOSE / ANALYSE
    u_hat = np.fft.fft(uini) 
    i=0
    #SOLVE EQUATION FOR \hat{u}_k
    for n in range(ntstep):
        if(label=='FE'):
            u_hat = ExplicitEulerFT(u_hat,k2,alpha,dt)
        elif(label=='BE'):
            u_hat = ImplicitEulerFT(u_hat,k2,alpha,dt)
        t=t+dt;
    #SYNTETHISE
    return np.real(np.fft.ifft(u_hat))	#ifft to have the solution in real space

def compute_error_from_num(u_ini,t_max,dt,scheme):
    alpha=0.5
    discVector=dt
    err=np.zeros(discVector.shape[0])
    minVal=min(discVector)/10.0
    for i in range(discVector.shape[0]):
        err[i]=max(abs(fourier_method(u_ini,tmax,discVector[i],scheme)-fourier_method(u_ini,tmax,minVal,scheme)))
    return err

def default_plotting():
    plt.figure(1,figsize=[10,8])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=14)
    plt.ylabel('u(x)',size=18)
    plt.xlabel('x',size=18)

# MAIN PROGRAM
# Discretization (x)
M = 32
N = 2*M                     
dx = 2*math.pi/N            
x = dx*np.arange(0,N)    

# Discretizazion (t)
t=0
tmax=10.0

#Aux. definitions
diff=np.zeros(N)

#Initial Condition and parameters
alpha = 0.5        
uini = u_exact(x,0,alpha)

# Discrete set of i*k (with i the imaginary unit)  
I = complex(0,1)
k = np.array([I*y for y in ( list(range(0,M)) + list([0]) + list(range(-M+1,0))) ])
k2=k**2;

t=0
dt = 1e-3
ntstep=int(round(tmax/dt))

#EXPLICIT/FORWARD EULER
#
default_plotting()
plt.plot(x ,u_exact(x,0,alpha),'k',label='t=0')
T=0.5
plt.plot(x, fourier_method(uini,T,dt,'FE'),'ob',label='num., t='+str(T))
plt.plot(x ,u_exact(x,T,alpha),'--b',label='exact, t='+str(T))
T=1.0
plt.plot(x, fourier_method(uini,T,dt,'FE'),'or',label='num., t='+str(T))
plt.plot(x ,u_exact(x,T,alpha),'--r',label='exact, t='+str(T))
T=2.0
plt.plot(x, fourier_method(uini,T,dt,'FE'),'og',label='num., t='+str(T))
plt.plot(x ,u_exact(x,T,alpha),'--g',label='exact, t='+str(T))
T=3.5
plt.plot(x, fourier_method(uini,T,dt,'FE'),'oy',label='num., t='+str(T))
plt.plot(x ,u_exact(x,T,alpha),'--y',label='exact, t='+str(T))
T=8.0
plt.plot(x, fourier_method(uini,T,dt,'FE'),'oc',label='num., t='+str(T))
plt.plot(x ,u_exact(x,T,alpha),'--c',label='exact, t='+str(T))
plt.title('Time evolution FE [$\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx))+' , $\\Delta$t='+str(dt)+']')
plt.legend(loc='best')
plt.savefig('TimeEvolution_FFE.png')
#plt.show()
plt.close()
#
#different dt
#
default_plotting()
T=3.0
#
plt.figure(1,figsize=[10,8])
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=14)
plt.ylabel('u(x)',size=18)
plt.xlabel('x',size=18)
dt=1e-4
plt.plot(x ,u_exact(x,T,alpha),'--k',label='exact, t='+str(T))
plt.plot(x, fourier_method(uini,T,dt,'FE'),'ob',label='dt='+str(dt))
dt=1e-3
plt.plot(x, fourier_method(uini,T,dt,'FE'),'sr',label='dt='+str(dt))
dt=4e-3
plt.plot(x, fourier_method(uini,T,dt,'FE'),'^g',label='dt='+str(dt))
plt.title('Solution (FE) at t='+str(T)+' with different $\\Delta$t, $\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx)))
plt.legend(loc='best')
plt.savefig('FFE_dt.png')
#plt.show()
plt.close()
#compute error
dtV_FE=np.logspace(-4,np.log10(4e-3),4)
err_FE=compute_error_from_num(uini,T,dtV_FE,'FE')
####


#BACKWARD/IMPLICT EULER
#
plt.figure(1,figsize=[10,8])
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=14)
plt.ylabel('u(x)',size=18)
plt.xlabel('x',size=18)
plt.plot(x ,u_exact(x,0,alpha),'k',label='t=0')
T=0.5
plt.plot(x, fourier_method(uini,T,dt,'BE'),'ob',label='num., t='+str(T))
plt.plot(x ,u_exact(x,T,alpha),'--b',label='exact, t='+str(T))
T=1.0
plt.plot(x, fourier_method(uini,T,dt,'BE'),'or',label='num., t='+str(T))
plt.plot(x ,u_exact(x,T,alpha),'--r',label='exact, t='+str(T))
T=2.0
plt.plot(x, fourier_method(uini,T,dt,'BE'),'og',label='num., t='+str(T))
plt.plot(x ,u_exact(x,T,alpha),'--g',label='exact, t='+str(T))
T=3.5
plt.plot(x, fourier_method(uini,T,dt,'BE'),'oy',label='num., t='+str(T))
plt.plot(x ,u_exact(x,T,alpha),'--y',label='exact, t='+str(T))
T=8.0
plt.plot(x, fourier_method(uini,T,dt,'BE'),'oc',label='num., t='+str(T))
plt.plot(x ,u_exact(x,T,alpha),'--c',label='exact, t='+str(T))
plt.title('Time evolution BE [$\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx))+' , $\\Delta$t='+str(dt)+']')
plt.legend(loc='best')
plt.savefig('TimeEvolution_FBE.png')
#plt.show()
plt.close()
#
#different dt
#
default_plotting()
T=3.0
#
plt.figure(1,figsize=[10,8])
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=14)
plt.ylabel('u(x)',size=18)
plt.xlabel('x',size=18)
dt=1e-2
plt.plot(x ,u_exact(x,T,alpha),'--k',label='exact, t='+str(T))
plt.plot(x, fourier_method(uini,T,dt,'BE'),'ob',label='dt='+str(dt))
dt=1e-1
plt.plot(x, fourier_method(uini,T,dt,'BE'),'sr',label='dt='+str(dt))
dt=0.5
plt.plot(x, fourier_method(uini,T,dt,'BE'),'vy',label='dt='+str(dt))
dt=1.0
plt.plot(x, fourier_method(uini,T,dt,'BE'),'^g',label='dt='+str(dt))
plt.title('Solution (BE) at t='+str(T)+' with different $\\Delta$t, $\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx)))
plt.legend(loc='best')
plt.savefig('FBE_dt.png')
#plt.show()
plt.close()
#compute error
dtV_BE=np.logspace(-4,-1,4)
err_BE=compute_error_from_num(uini,T,dtV_BE,'BE')
####

#Error ( dt )
plt.figure(1,figsize=[10,8])
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=16)
plt.ylabel('E($\\Delta$t)',size=20)
plt.xlabel('$\\Delta$t',size=18)
plt.loglog(dtV_BE,err_BE,'--or',label='BE')
plt.loglog(dtV_FE,err_FE,'--sb',label='FE')
plt.loglog(dtV_BE,np.ones(dtV_BE.shape[0])*1e-5,'--k',label='$\\epsilon_1$')
plt.loglog(dtV_BE,np.ones(dtV_BE.shape[0])*2e-4,':k',label='$\\epsilon_2$')
annotation.slope_marker((0.007, 2.6e-5), (1, 1))
plt.legend(loc='best')
plt.title('Error as function of $\\Delta$t, $\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx)))
plt.savefig('Error_FM_dt.png')
#plt.show()
plt.close()
