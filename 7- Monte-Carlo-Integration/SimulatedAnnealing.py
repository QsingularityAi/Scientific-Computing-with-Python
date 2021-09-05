import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys

def default_plotting():
    plt.figure(1,figsize=[10,8])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=18)

def funcV(P):
    obj = 0.2 + P[0]**2 + P[1]**2 - 0.1*np.cos(6.0*np.pi*P[0]) - 0.1*np.cos(6.0*np.pi*P[1])
    return obj

def cooling_schedule(Tini,Tend,n,N,label='linear'):
    if(label=='linear'):
        return Tini+(n)*(Tend-Tini)/(N-1) 
    if(label=='exp'):
        return Tend+(Tini-Tend)*np.exp(-5.0*(n-1)/N)

#Temperature bounds and cooling schedule
try:  
    lab_cooling=sys.argv[1]
    tini=float(sys.argv[2])
except:
    lab_cooling='linear'
    tini=10.0
tend=1E-10
    

#PLOT V(x1,x2)
L=1.0
xx = np.arange(-L, L, 0.01)
yy = np.arange(-L, L, 0.01)
x1m, x2m = np.meshgrid(xx, yy)
fm = np.zeros(x1m.shape)
fm=funcV(np.array([x1m,x2m]))
default_plotting()
CS = plt.contour(x1m, x2m, fm, 10, cmap='bone')
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

#Initial Position
xini = [0.9*L, -0.9*L]
sd=138105
random.seed(sd)

# Simulated Annealing
n = 40
na = 0.0

# Initialize x
x = np.zeros((n,2))
x[0,:] = xini
na = na + 1.0

# Current best results so far
xc = np.zeros(2); xi=np.zeros(2)
xc[0] = xini[0]
xc[1] = xini[1]
fc = funcV(xini)
fs = np.zeros(n+1)
fs[0] = fc
t = tini
i=1

nreject=0
while(i<n):
    print('Iteration: ' + str(i) + ' T=' + str(t))
    # Generate new trial points
    xi[0] = xc[0] + 0.5*(2* random.random() - 1)
    xi[1] = xc[1] + 0.5*(2* random.random() - 1)
    xi[0] = max(min(xi[0],1.0),-1.0)
    xi[1] = max(min(xi[1],1.0),-1.0)
    DeltaE = funcV(xi)-fc
    if (DeltaE>0):
        p = np.exp(-DeltaE/t)
        if (random.random()<p):
            accept = True
        else:
            accept = False
    else:
        accept = True

    if(accept==True):
        xc[0] = xi[0]
        xc[1] = xi[1]
        fc = funcV(xc)
        na = na + 1.0
        x[i][0] = xc[0]
        x[i][1] = xc[1]
        fs[i] = fc
        i=i+1
    else:
        nreject=nreject+1
    t = cooling_schedule(tini,tend,i,n,label=lab_cooling)


#PLOT POSITIONS (ON V(x1,x2), see above)
plt.plot(x[:,0],x[:,1],'b--o',alpha=0.5,linewidth=2,ms=10)
plt.plot(xini[0],xini[1],'g*',ms=16)
plt.text(xini[0]+0.075,xini[1],'Start',color='green',bbox=dict(boxstyle="round",ec=(1.0, 1.0, 1.0),fc=(1.0, 1.0, 1.0)))
plt.plot(xc[0],xc[1],'r*',ms=16)
plt.text(xc[0]+0.075,xc[1],'End',color='red',bbox=dict(boxstyle="round",ec=(1.,1.0,1.0),fc=(1., 1.0, 1.0)))
plt.title("$T_{\\rm ini}=$"+str(tini))
plt.savefig('SA_'+str(lab_cooling)+'Tini'+str(tini)+'.png')
plt.axis([-1,1,-1,1])
#plt.show()
plt.close()


#PLOT_DATA
fig = plt.figure(1,figsize=[10,8])
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=18)
plt.title("Targeted moves: "+str(n)+"     Rejected Iterations: "+str(nreject))
ax1 = fig.add_subplot(211)
ax1.plot(fs,'k-')
ax1.legend(['Energy'])
ax2 = fig.add_subplot(212)
ax2.plot(x[:,0],'b.-')
ax2.plot(x[:,1],'r--')
ax2.set_xlabel('$i$')
ax2.legend(['x1','x2'])
plt.savefig('Data_'+str(lab_cooling)+'Tini'+str(tini)+'.png')
#plt.show()
plt.close()
