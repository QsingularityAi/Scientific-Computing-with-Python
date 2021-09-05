#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.lib.scimath import sqrt as csqrt
from scipy.integrate import odeint
from functions_bruss import bruss_1


#Stable
A1=1
B1=0.9
n=200
tend=10
t=np.linspace(0, tend, n)

#Initial Condition X0=2, Y0=1
sol1 = odeint(bruss_1, [2, 1], t, args=(A1, B1))
#Initial Condition X0=0.25, Y0=0.1
sol2 = odeint(bruss_1, [0.25, 0.1], t, args=(A1, B1))


#Periodic (Unstable but bounded)
A2=1
B2=2.1
n=500
tend=30
t2=np.linspace(0, tend, n)

#Initial Condition X0=2, Y0=1
sol3 = odeint(bruss_1, [2, 1], t2, args=(A2, B2))
#Initial Condition X0=0.25, Y0=0.1
sol4 = odeint(bruss_1, [0.25, 0.1], t2, args=(A2, B2))


#Plotting
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=13)
plt.figure(1,figsize=[16,12])
ax=plt.subplot(221)
ax.plot(t, sol1[:,0],'g',linewidth=2, label='X - U0=[2,1]')
ax.plot(t, sol1[:,1],'g--',linewidth=2, label='Y - U0=[2,1]')
ax.plot(t, sol2[:,0],'b-.',linewidth=2, label='X - U0=[0.25,0.1]')
ax.plot(t, sol2[:,1],'b:',linewidth=2, label='Y - U0=[0.25,0.1]')
plt.legend(loc='best')
plt.title('STABLE (A=1,B=0.9) - X  and  Y vs time ')
plt.xlabel('t')
plt.ylabel('X(t) and Y(t)')

ax=plt.subplot(222)
ax.plot(sol1[:,0], sol1[:,1],'k', label='U0=[2,1]')
ax.plot(sol2[:,0], sol2[:,1],'k--', label='U0=[0.25,0.1]')
plt.legend(loc='best')
plt.title('STABLE (A=1,B=0.9) - X(t)  vs  Y(t)')
plt.xlabel('X(t)')
plt.ylabel('Y(t)')

ax=plt.subplot(223)
ax.plot(t2, sol3[:,0],'g',linewidth=2, label='X - U0=[2,1]')
ax.plot(t2, sol3[:,1],'g--',linewidth=2, label='Y - U0=[2,1]')
ax.plot(t2, sol4[:,0],'b-.',linewidth=2, label='X - U0=[0.25,0.1]')
ax.plot(t2, sol4[:,1],'b:',linewidth=2, label='Y - U0=[0.25,0.1]')
plt.legend(loc='best')
plt.title('BOUNDED/PERIODIC (A=1,B=2.1) - X  and  Y vs time ')
plt.xlabel('t')
plt.ylabel('X(t) and Y(t)')

ax=plt.subplot(224)
ax.plot(sol3[:,0], sol3[:,1],'k', label='U0=[2,1]')
ax.plot(sol4[:,0], sol4[:,1],'k--', label='U0=[0.25,0.1]')
plt.legend(loc='best')
plt.title('BOUNDED/PERIODIC (A=1,B=2.1) - X(t)  vs  Y(t)')
plt.xlabel('X(t)')
plt.ylabel('Y(t)')

plt.savefig('bruss_1.png')
plt.show()







