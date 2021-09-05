#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math
from functions import u0
from functions import u1
from functions import integration_scheme_wave

#definition of parameters
L=1.0
c=2.0

tau=0.5 #period
T=1.0

nx1=51
nx2=52
nt=50*2+1

#Set discretization(s)
u_1=np.zeros((nx1,nt)) #solution discretization 1 nx=50
u_2=np.zeros((nx2,nt)) #solution discretization 2 nx=51

x1=np.linspace(0, L, nx1) #x-discretization 1
x2=np.linspace(0, L, nx2) #x-discretization 2

t=np.linspace(0, T, nt)   #t-discretization 1

#x,t-steps
dx1=x1[2]-x1[1]
dx2=x2[2]-x2[1]
dt=t[2]-t[1]

#t=0 (n=0), t=dt (n=1)
u_1[:,0]=u0(x1)
u_1[:,1]=u_1[:,0]+dt*u1(x1)
u_2[:,0]=u0(x2)
u_2[:,1]=u_2[:,0]+dt*u1(x2)

#Time iteration
for i in range(1,nt-1):
	u_1[:,i+1]=integration_scheme_wave(u_1[:,i],u_1[:,i-1],dx1,dt,c)
	u_2[:,i+1]=integration_scheme_wave(u_2[:,i],u_2[:,i-1],dx2,dt,c)


print("Plotting....")
plt.figure(1,figsize=[10,8])
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=13)

ax=plt.subplot(221)
ax.plot(np.linspace(0,L,200), u0(np.linspace(0,L,200)),'r')
ax.plot(x1, u_1[:,int((nt-1)/2)],'--bo',fillstyle='none')
plt.title('time=0.5, s=1')
plt.gca().legend(('Exact sol','Num. sol.'),loc=3)

ax=plt.subplot(222)
ax.plot(np.linspace(0,L,200), u0(np.linspace(0,L,200)),'r')
ax.plot(x1, u_1[:,int(nt-1)],'--bo',fillstyle='none')
plt.title('time=1.0, s=1')
plt.gca().legend(('Exact sol','Num. sol.'),loc=3)

ax=plt.subplot(223)
ax.plot(np.linspace(0,L,200), u0(np.linspace(0,L,200)),'r')
ax.plot(x2, u_2[:,int((nt-1)/2)],'--bo',fillstyle='none')
plt.title('time=0.5, s=1.02')
plt.gca().legend(('Exact sol','Num. sol.'),loc=3)

ax=plt.subplot(224)
ax.plot(np.linspace(0,L,200), u0(np.linspace(0,L,200)+1.0*2),'r')
ax.plot(x2, u_2[:,int(nt-1)],'--bo',fillstyle='none')
plt.title('time=1.0, s=1.02')
plt.gca().legend(('Exact sol','Num. sol.'),loc=3)



plt.show()

print("Done.\n")






