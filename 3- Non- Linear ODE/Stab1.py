#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.lib.scimath import sqrt as csqrt


# 1 parameter fixed (A)
n=100

A=1.0
B=np.linspace(0, 4, n)

eigenvalue_1=np.zeros((n),dtype=complex)
eigenvalue_2=np.zeros((n),dtype=complex)
max_re_eig=np.zeros(n)

D=(A**2-B+1)**2-4*A**2

eigenvalue_1=B-A**2-1+csqrt(D)
eigenvalue_2=B-A**2-1-csqrt(D)


for i in range(n):
	max_re_eig[i]=max(eigenvalue_1.real[i],eigenvalue_2.real[i])


# 2 free parameter
n2=100
A2=np.linspace(0, 2, n2)
B2=np.linspace(0, 4, n2)
D2=np.zeros((n2,n2))

eigenvalue2_1=np.zeros((n2,n2),dtype=complex)
eigenvalue2_2=np.zeros((n2,n2),dtype=complex)
max_re_eig2=np.zeros((n2,n2))

#Alternative implementation: create a grid from A and B values and evaluate the following expression
#directly with matrices without "for" cycle. In python use A2g, B2g =np.meshgrid(A2,B2)
for i in range(n2):
	for j in range(n2):
		D2[i,j]=(A2[i]**2-B2[j]+1)**2-4*A2[i]**2
		eigenvalue2_1[i,j]=B2[j]-A2[i]**2-1+csqrt(D2[i,j])
		eigenvalue2_2[i,j]=B2[j]-A2[i]**2-1-csqrt(D2[i,j])
		max_re_eig2[j,i]=max(eigenvalue2_1.real[i,j],eigenvalue2_2.real[i,j])

plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=13)
plt.figure(1,figsize=[10,4])
ax=plt.subplot(121)
ax.plot(B, max_re_eig,'b--')
ax.plot(B, np.zeros(n),'k')
plt.title('A=1  and  B=[0:4]')
plt.xlabel('B')
plt.ylabel('$\max_i$[Re($\\lambda_i$)]')

X,Y=np.meshgrid(A2, B2)
ax=plt.subplot(122)
ax = plt.contourf(X,Y,max_re_eig2, 30, cmap=plt.cm.bwr,vmin=-5,vmax=5)
CS2 = plt.contour(ax, levels=[0.0],colors='g')
plt.annotate('$\max_i$[Re($\\lambda_i$)]=0', (0.76, 3.3),color='g')
plt.plot(1.0,0.9,'go')
plt.annotate('[1.0,0.9]', (0.6, 0.9),color='g')
plt.plot(1.0,2.1,'gs')
plt.annotate('[1.0,2.1]', (0.6, 2.1),color='g')
plt.title('A=[0:2]  and   B=[0:4]')
plt.xlabel('A')
plt.ylabel('B')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

plt.savefig('stability_1.png')
plt.show()







