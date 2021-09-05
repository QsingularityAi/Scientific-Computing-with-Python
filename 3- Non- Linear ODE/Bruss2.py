#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.lib.scimath import sqrt as csqrt
from scipy.integrate import odeint
from functions_bruss import bruss_2


#Stable
A1=1
v1=0.9
n=500
tend=30
t=np.linspace(0, tend, n)

#Initial Condition X0=1, Y0=2, Z0=1
sol1 = odeint(bruss_2, [1, 2, 1], t, args=(A1, v1))
#Initial Condition X0=2, Y0=2, Z0=2
sol2 = odeint(bruss_2, [2, 2, 2], t, args=(A1, v1))


#Unstable but bounded
A2=1
v2=1.3
n2=500
tend=60
t2=np.linspace(0, tend, n2)

#Initial Condition X0=1, Y0=2, Z0=1
sol3 = odeint(bruss_2, [1, 2, 1], t2, args=(A2, v2))
#Initial Condition X0=2, Y0=2, Z0=2
sol4 = odeint(bruss_2, [2, 2, 2], t2, args=(A2, v2))


#Unstable 
A3=1
v3=1.52
n3=200
tend=60
t3=np.linspace(0, tend, n3)

#Initial Condition X0=1, Y0=2, Z0=1
sol5 = odeint(bruss_2, [1, 2, 1], t3, args=(A3, v3))
#Initial Condition X0=2, Y0=2, Z0=2
sol6 = odeint(bruss_2, [2, 2, 2], t3, args=(A3, v3))


#Plotting
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=13)
plt.figure(1,figsize=[22,18])
plt.figure(1)
ax=plt.subplot(331)
ax.plot(t, sol1[:,0],'k',linewidth=2, label='X')
ax.plot(t, sol1[:,1],'k--',linewidth=2, label='Y')
ax.plot(t, sol1[:,2],'k:',linewidth=2, label='Z')
plt.legend(loc='best')
plt.title('STABLE (A=1,v=0.9) - X, Y and Z vs time - U0[1,2,1] ')
plt.xlabel('t')
plt.ylabel('X(t) and Y(t)')

ax=plt.subplot(332)
ax.plot(sol1[:,0], sol1[:,1],'k', label='U0=[1,2,1]')
ax.plot(sol2[:,0], sol2[:,1],'k--', label='U0=[2,2,2]')
plt.legend(loc='best')
plt.title('STABLE (A=1,v=0.9) - Y(t)  vs  X(t)')
plt.xlabel('X(t)')
plt.ylabel('Y(t)')

ax=plt.subplot(333)
ax.plot(sol1[:,0], sol1[:,2],'k', label='U0=[1,2,1]')
ax.plot(sol2[:,0], sol2[:,2],'k--', label='U0=[2,2,2]')
plt.legend(loc='best')
plt.title('STABLE (A=1,v=0.9) - Z(t)  vs  X(t)')
plt.xlabel('X(t)')
plt.ylabel('Z(t)')


ax=plt.subplot(334)
ax.plot(t2, sol3[:,0],'k',linewidth=2, label='X')
ax.plot(t2, sol3[:,1],'k--',linewidth=2, label='Y')
ax.plot(t2, sol3[:,2],'k:',linewidth=2, label='Z')
plt.legend(loc='best')
plt.title('PERIODIC (A=1,v=1.3) - X, Y and Z vs time - U0[1,2,1] ')
plt.xlabel('t')
plt.ylabel('X(t) and Y(t)')

ax=plt.subplot(335)
ax.plot(sol3[:,0], sol3[:,1],'k', label='U0=[1,2,1]')
ax.plot(sol4[:,0], sol4[:,1],'k--', label='U0=[2,2,2]')
plt.legend(loc='best')
plt.title('PERIODIC (A=1,v=1.3) - Y(t)  vs  X(t)')
plt.xlabel('X(t)')
plt.ylabel('Y(t)')

ax=plt.subplot(336)
ax.plot(sol3[:,0], sol3[:,2],'k', label='U0=[1,2,1]')
ax.plot(sol4[:,0], sol4[:,2],'k--', label='U0=[2,2,2]')
plt.legend(loc='best')
plt.title('PERIODIC (A=1,B=1.3) - Z(t)  vs  X(t)')
plt.xlabel('X(t)')
plt.ylabel('Z(t)')

ax=plt.subplot(337)
ax.plot(t3, sol5[:,0],'k',linewidth=2, label='X')
ax.plot(t3, sol5[:,1],'k--',linewidth=2, label='Y')
ax.plot(t3, sol5[:,2],'k:',linewidth=2, label='Z')
plt.legend(loc='best')
plt.title('UNSTABLE (A=1,v=1.3) - X, Y and Z vs time - U0[1,2,1] ')
plt.xlabel('t')
plt.ylabel('X(t) and Y(t)')

ax=plt.subplot(338)
ax.plot(sol5[:,0], sol5[:,1],'k', label='U0=[1,2,1]')
ax.plot(sol6[:,0], sol6[:,1],'k--', label='U0=[2,2,2]')
plt.legend(loc='best')
plt.title('UNSTABLE (A=1,v=1.3) - Y(t)  vs  X(t)')
plt.xlabel('X(t)')
plt.ylabel('Y(t)')

ax=plt.subplot(339)
ax.plot(sol5[:,0], sol5[:,2],'k', label='U0=[1,2,1]')
ax.plot(sol6[:,0], sol6[:,2],'k--', label='U0=[2,2,2]')
plt.legend(loc='best')
plt.title('UNSTABLE (A=1,B=1.3) - Z(t)  vs  X(t)')
plt.xlabel('X(t)')
plt.ylabel('Z(t)')



plt.savefig('bruss_2.png')
plt.show()







