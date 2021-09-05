#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math
from functions import decay_an
from functions import ExplicitEuler
from functions import RungeKutta4


#definition of parameters
alpha=4.0
f=0.0
u0=1.0

tini=0.0
tend=3.0

h1=1.0/2.0
h2=1.0/8.0

#Set discretization(s)
Tn1=np.arange(tini, tend+0.001, h1)
Tn2=np.arange(tini, tend+0.001, h2)
#Tn2=np.linspace(tini, tend, N) with N the total number of steps, it works for uniform discretization


#Plotting exploiting integrations schemes defined in the Module "functions"
print("Plotting....")
plt.figure(1,figsize=[15,6])
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=16)
#
ax = plt.subplot(121)
ax.plot(np.arange(tini,tend,0.01), decay_an(np.arange(tini,tend,0.01),f,alpha,u0),lw=2)
ax.plot(Tn1, ExplicitEuler(Tn1,f,alpha,u0),'--', marker='s',color='g')
ax.plot(Tn2, ExplicitEuler(Tn2,f,alpha,u0),'--', marker='o',color='r')
plt.title('Explicit Euler')
plt.gca().legend(('Exact','h=1/2','h=1/8'),loc=1)
#
ax = plt.subplot(122)
ax.plot(np.arange(tini,tend,0.01), decay_an(np.arange(tini,tend,0.01),f,alpha,u0),lw=2)
ax.plot(Tn1, RungeKutta4(Tn1,f,alpha,u0),'--', marker='s',color='g')
ax.plot(Tn2, RungeKutta4(Tn2,f,alpha,u0),'--', marker='o',color='r')
plt.title('Runge-Kutta-4th')
plt.gca().legend(('Exact','h=1/2','h=1/8'),loc=1)
#
plt.show()
print("Done.\n")






