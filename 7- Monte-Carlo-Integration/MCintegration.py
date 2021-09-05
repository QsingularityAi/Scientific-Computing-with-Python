import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import random
from mpltools import annotation
from scipy.optimize import curve_fit

def default_plotting():
    plt.figure(1,figsize=[10,8])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=18)

def funcexp(x, a=1, b=0.5):
    return a * x**b 

def gauss1D_norm(x,s):
    return((1/np.sqrt(2*(s**2)*np.pi))*np.exp(-(x[:]**2)/(2*s**2)))

def gaussND(x):
    res=1
    for i in range(x.shape[0]):
        res=res*np.exp(-(x[i]**2))
    return res

def Hit_method_gauss(N,s,L):
    default_plotting()
    X=np.linspace(-L,L,10000)
    val,x,y,ck=MCintegrate(-L,L,'gauss1D_norm',s,N)
    plt.plot(X,gauss1D_norm(X,1),'-r',linewidth=3)
    plt.scatter(x,y,c=ck)
    plt.title('Hit method, N='+str(N))
    plt.ylabel('$y$',size=20)
    plt.xlabel('$x$',size=20)
    plt.savefig('Hit_n'+str(N)+'.png')
    plt.show()

def MCintegrate(x1,x2,func,s,n,print_steps=True,st=1000):
    try:
        dim=x1.shape[0]
    except:
        dim=1
        x=np.zeros(1000)
    
    volume=0
    y1=0

    if(func=='gauss1D_norm'):
        y2=max(gauss1D_norm(x,s))+0.1
    elif(func=='gaussND'):
        y2=1.1

    if(dim==1):
        volume=(x2-x1)*(y2-y1)
    else:
        ddxx=x2-x1
        volume=(y2-y1)*np.prod(ddxx)

    check=[]
    xs=np.zeros([n,dim])
    ys=np.zeros([n])

    for i in range(n):
        if(dim==1):
            xs[i]=np.random.uniform(x1,x2,1)
        else:
            for j in range(dim): 
                xv=np.random.uniform(x1[j],x2[j],1)
                xs[i,j]=xv
        ys[i]=np.random.uniform(y1,y2,1)
        if(func=='gauss1D_norm'):
            func_x=gauss1D_norm(xs[i],s)
        elif(func=='gaussND'):
            func_x=gaussND(xs[i,:])
        if abs(ys[i])>abs(func_x):
            check.append(0)
        else:
            check.append(1)
        if(print_steps):
            if(i%st==0):
                print(i," / ",n,"  I=",np.average(check)*volume)
    return((np.average(check))*volume,xs,ys,check)

s=0
sd=138105
random.seed(sd)

#
#Illustration of Hit Method
#
Enable=0
if(Enable):
    s=1
    Hit_method_gauss(100,s,5.0*s)
    Hit_method_gauss(1000,s,5.0*s)
    Hit_method_gauss(10000,s,5.0*s)
    Hit_method_gauss(100000,s,5.0*s)

#
#Integration of G
#
sd=138105
random.seed(sd)
Enable=0
if(Enable):
    default_plotting()
    s=1
    L=5.0*s
    N=2000000

    #Integration
    val,x,y,ck=MCintegrate(-L,L,'gauss1D_norm',s,N,print_steps=True,st=100000)

    #Definition of additional variables
    dev=np.zeros(N);devBIN=np.zeros(N);vvv=np.zeros(N)
    xs=np.linspace(-L,L,10000)
    yy=max(gauss1D_norm(x,s))+0.1
    area=(2*L)*yy
    dev[0]=ck[0]*area-val
    vvv[0]=ck[0]*area

    #Extracting info. as function of n
    for i in range(1,int(N/2)):
        if(i%100000==0):
            print("i=",i)
        vvv[i]=(((vvv[i-1])/(area)*i + ck[i])*area)/(i+1)
        dev[i]=(((dev[i-1]+val)/(area)*i + ck[i])*area)/(i+1)-val
        p=(vvv[i]/area)
        devBIN[i]=np.sqrt(p*(1-p)/(i+1))*area

    #We show results from iteration # n=nini to n=N/2 (with reference values what it obtained with n=N)
    #
    #
    nini=100
    nvec=np.linspace(nini,int(N/2),(int(N/2)-nini))
    popt, pcov = curve_fit(funcexp, nvec, abs(dev[nini:int(N/2)]))
    plt.loglog(nvec,abs(dev[nini:int(N/2)]),'-r')
    plt.loglog(nvec,funcexp(nvec,*popt),'--b',label="Fitted Exponent: "+str(np.around(popt[1],6)))
    annotation.slope_marker((3e4, 0.001), (-0.5, 1))
    plt.title('Convergence rate (to $\hat{G}_N$), N='+str(N))
    plt.ylabel('$\Delta_G(n,N)$',size=20)
    plt.xlabel('$n$',size=20)
    plt.legend(loc="best")
    plt.savefig('Rate_G.png')
    plt.show()
    plt.close
    #
    #
    default_plotting()
    plt.fill_between(nvec,2*devBIN[nini:int(N/2)]+vvv[nini:int(N/2)],-2*devBIN[nini:int(N/2)]+vvv[nini:int(N/2)],color='y',alpha=0.25,label="$\hat{G}_n\\pm 2\sigma$")
    plt.fill_between(nvec,devBIN[nini:int(N/2)]+vvv[nini:int(N/2)],-devBIN[nini:int(N/2)]+vvv[nini:int(N/2)],alpha=0.25,label="$\hat{G}_n \\pm \sigma$")
    plt.plot(nvec,vvv[nini:int(N/2)],'-b',linewidth=2,label='$\hat{G}_n$')
    plt.plot(nvec,val*np.ones(nvec.shape[0]),'--r',linewidth=2,label="$$\hat{G}_N$$")
    plt.title('Convergence to $\hat{G}_N$, N='+str(N))
    plt.legend(loc="best")
    plt.ylabel('$\hat{G}(n)$',size=20)
    plt.xlabel('$n$',size=20)
    plt.axis([nini,int(N/2),0.95,1.05])
    plt.savefig('Conv_G.png')
    plt.show()
    plt.close()


#
#Integration of F
#
sd=138105
random.seed(sd)
Enable=1
if(Enable):
    default_plotting()
    s=1
    L=3.0*np.ones(4)
    dim=len(L)
    N=2000000

    #Integration
    val,xN,yN,ck=MCintegrate(-L,L,'gaussND',1,N,print_steps=True,st=100000)

    #Definition of additional variables
    dev=np.zeros(N);devBIN=np.zeros(N); vvv=np.zeros(N)
    yy=1.1
    vol=1
    for i in range(dim):
        vol=vol*2*L[i]
    vol=vol*yy
    print(vol,ck[0],val)
    dev[0]=ck[0]*vol-val
    vvv[0]=ck[0]*vol

    #Extracting info. as function of n
    for i in range(1,int(N/2)):
        if(i%100000==0):
            print("i=",i)
        vvv[i]=(((vvv[i-1])/vol)*i + ck[i])*vol/(i+1)
        dev[i]=(((dev[i-1]+val)/(vol)*i + ck[i])*vol)/(i+1)-val
        p=(vvv[i]/vol)
        devBIN[i]=np.sqrt(p*(1-p)/(i+1))*vol

    #We show results from iteration # n=nini to n=N/2 (with reference values what it obtained with n=N)
    #
    #
    nini=100
    nvec=np.linspace(nini,int(N/2),(int(N/2)-nini))
    popt, pcov = curve_fit(funcexp, nvec, abs(dev[nini:int(N/2)]))
    plt.loglog(nvec,abs(dev[nini:int(N/2)]),'-r')
    plt.loglog(nvec,funcexp(nvec,*popt),'--b',label="Fitted Exponent: "+str(np.around(popt[1],6)))
    annotation.slope_marker((2e2, 0.001), (-0.5, 1))
    plt.title('Convergence rate (to $\hat{F}_N$), N='+str(N))
    plt.ylabel('$\Delta_F(n,N)$',size=20)
    plt.xlabel('$n$',size=20)
    plt.legend(loc="best")
    plt.savefig('Rate_F.png')
    plt.show()
    plt.close
    #
    #
    default_plotting()
    plt.fill_between(nvec,2*devBIN[nini:int(N/2)]+vvv[nini:int(N/2)],-2*devBIN[nini:int(N/2)]+vvv[nini:int(N/2)],color='y',alpha=0.25,label="$\hat{F}_n\\pm 2\sigma$")
    plt.fill_between(nvec,devBIN[nini:int(N/2)]+vvv[nini:int(N/2)],-devBIN[nini:int(N/2)]+vvv[nini:int(N/2)],alpha=0.25,label="$\hat{F}_n \\pm \sigma$")
    plt.plot(nvec,vvv[nini:int(N/2)],'-b',linewidth=2,label='$\hat{F}_n$')
    plt.plot(nvec,val*np.ones(nvec.shape[0]),'--r',linewidth=2,label="$$\hat{F}_N$$")
    plt.title('Convergence to $\hat{F}_N$, N='+str(N))
    plt.legend(loc="best")
    plt.ylabel('$\hat{F}(n)$',size=20)
    plt.xlabel('$n$',size=20)
    plt.axis([nini,int(N/2),np.pi**2-2,np.pi**2+2])
    plt.savefig('Conv_F.png')
    plt.show()
    plt.close()






