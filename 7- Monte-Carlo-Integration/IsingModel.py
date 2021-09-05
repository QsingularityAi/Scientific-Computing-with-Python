from __future__ import division
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

def default_plotting():
    plt.figure(1,figsize=[10,8])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=18)

def initial_conf(N):
    return 2*np.random.randint(2, size=(N,N))-1

def mcmove(c_spin, beta):
    for i in range(N):
        for j in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s =  c_spin[a, b]
                nb = c_spin[(a+1)%N,b] + c_spin[a,(b+1)%N] + c_spin[(a-1)%N,b] + c_spin[a,(b-1)%N]
                cost = 2*s*nb
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost*beta):
                    s *= -1
                c_spin[a, b] = s
    return c_spin

def Energy(c_spin):
    energy = 0
    for i in range(len(c_spin)):
        for j in range(len(c_spin)):
            S = c_spin[i,j]
            nb = c_spin[(i+1)%N, j] + c_spin[i,(j+1)%N] + c_spin[(i-1)%N, j] + c_spin[i,(j-1)%N]
            energy += -nb*S
    return energy/4.

def Mag(c_spin):
    return np.sum(c_spin)

sd=9895
np.random.seed(sd)

nt=50                     #  number of temperature sampled
N=10                      #  size of the lattice, N x N
mcSteps=500               #  number of calls to MC 

T=np.linspace(1.0, 4.0, nt);
E=np.zeros(nt)
M=np.zeros(nt)
n1,n2=1.0/(mcSteps*N*N),1.0/(mcSteps*mcSteps*N*N)

#Simulations for different T
arr_output=np.zeros([N,N,nt])
for tt in range(nt):
    print(tt)
    Et = 0
    Mt = 0 
    c_spin = initial_conf(N)
    iT=1.0/T[tt]; iT2=iT*iT;

    for i in range(mcSteps):
        mcmove(c_spin, iT)
        En = Energy(c_spin) 
        MM = Mag(c_spin)        
        Et = Et + En
        Mt = Mt + MM

    if(tt%10==0 or tt==(nt-1)):
        default_plotting()
        plt.imshow(c_spin,cmap='Greys',interpolation='nearest')
        plt.xlabel("x", fontsize=20);
        plt.ylabel("y", fontsize=20);
        plt.title("T="+str(np.round(T[tt],2))+"\ \ \ \ \ \ white$\\rightarrow$ $s_i=1$, black$\\rightarrow$ $s_i=-1$")
        plt.savefig("IsingN"+str(N)+"T"+str(tt)+".png")
        #plt.show()
        plt.close()

    E[tt] = n1*Et
    M[tt] = n1*Mt



default_plotting()
plt.plot(T, E,'-o', color='r')
plt.xlabel("Temperature ($T$)", fontsize=20);
plt.ylabel("Energy ", fontsize=20);
plt.title(str(N)+"x"+str(N)+"\ spin-lattice")
plt.savefig("IsingTemperature"+str(N)+".png")
plt.show()
plt.close()

default_plotting()
plt.plot(T, abs(M),'-o', color='b')
plt.xlabel("Temperature ($T$)", fontsize=20); 
plt.ylabel("Magnetization ($|m|$) ", fontsize=20);  
plt.title(str(N)+"x"+str(N)+"\ spin-lattice")
plt.savefig("IsingEnergy"+str(N)+".png")
plt.show()
plt.close()

