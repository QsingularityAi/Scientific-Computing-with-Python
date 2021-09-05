#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.lib.scimath import sqrt as csqrt
from numpy import linalg as LA

# 1 parameter fixed (A)
n=200

A=1.0
v=np.linspace(0, 2, n)
max_re_eig=np.zeros(n)

for i in range(n):
	ev=LA.eigvals(np.array([ [(v[i]/A)-1,1,-1] , [-v[i],-1,1] , [-v[i],0,-1] ] ))
	max_re_eig[i]=(ev.real).max()


# 2 free parameter
n2=100
A2=np.linspace(0.5, 1.5, n2)
v2=np.linspace(0, 2.0, n2)
max_re_eig2=np.zeros((n2,n2))

for i in range(n2):
    for j in range(A2.shape[0]):
        ev=LA.eigvals(np.array([ [(v2[i]/A2[j])-1,1,-1] , [-v2[i],-1,1] , [-v2[i],0,-1] ] ))
        max_re_eig2[i,j]=(ev.real).max()



plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=13)
plt.figure(1,figsize=[10,4])

ax=plt.subplot(121)
ax.plot(v, max_re_eig,'b--')
ax.plot(v, np.zeros(n),'k')
plt.title('A=1  and  v=[0:2]')
plt.xlabel('v')
plt.ylabel('$\max_i$[Re($\\lambda_i$)]')

X,Y=np.meshgrid(A2, v2)
ax=plt.subplot(122)
ax = plt.contourf(X,Y,max_re_eig2, 30, cmap=plt.cm.bwr,vmin=-1,vmax=1)
CS2 = plt.contour(ax, levels=[0.0], colors='g')
plt.annotate('$\max_i$[Re($\\lambda_i$)]=0', (1.12, 1.65),color='g')
plt.plot(1.0,0.9,'go')
plt.annotate('[1.0,0.9]', (0.8, 0.85),color='g')
plt.plot(1.0,1.3,'gs')
plt.annotate('[1.0,1.3]', (0.8, 1.25),color='g')
plt.plot(1.0,1.53,'gD')
plt.annotate('[1.0,1.53]', (0.78, 1.47),color='g')
plt.clim(-1,1)
plt.title('A=[0.5:1.5]  and   v=[0:2]')
plt.xlabel('A')
plt.ylabel('v')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
plt.savefig('stability_2.png')
plt.show()







