#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math
from functions_FD import u_exact
from functions_FD import d2u_3ptstencil
from functions_FD import d2u_5ptstencil
from mpltools import annotation

def t_vec(T=3,dt=0.01):
    Nt=int((T+dt)/dt)
    t=np.linspace(0,T,Nt)
    return t

def x_vec(L=2.0*np.pi,dx=2.0*np.pi/20):
    Nx=int((L+dx)/dx)
    x=np.linspace(0,L,Nx)
    return x

def diff_3p_5p(L=2.0*np.pi,Ttot=3,dx=2.0*np.pi/20,dt=0.01,label='3pi'):

    #definition of parameters
    alpha=0.5

    Nx=int((L+dx)/dx)
    Nt=int((Ttot+dt)/dt)

    #Set discretization(s)
    u_3pt=np.zeros((Nx,Nt)) #solution with 3pt stencil
    u_5pt=np.zeros((Nx,Nt)) #solution with 5pt stencil
    x=x_vec(L,dx)      #x-discretization
    t=t_vec(Ttot,dt)   #t-discretization

    #t=0 (n=0)
    u_3pt[:,0]=u_exact(x,0.0,alpha)
    u_5pt[:,0]=u_exact(x,0.0,alpha)

    #Time iteration
    for i in range(0,Nt-1):
        if(label=='3pt'):
            u_3pt[:,i+1]=d2u_3ptstencil(u_3pt[:,i],dx,dt,alpha,L)
        elif(label=='5pt'):
            u_5pt[:,i+1]=d2u_5ptstencil(u_5pt[:,i],dx,dt,alpha,L)

    #Return
    if(label=='3pt'):
        return u_3pt[:,Nt-1]
    elif(label=='5pt'):
        return u_5pt[:,Nt-1]

def compute_error_from_exactsol(L,T,dx,dt,label,scheme='3pt'):
    alpha=0.5
    if(label=='dt'):
        discVector=dt
    elif(label=='dx'):
        discVector=dx
    err=np.zeros(discVector.shape[0])
    for i in range(discVector.shape[0]):
       if(label=='dt'):
           err[i]=max(abs(diff_3p_5p(L,T,dx,discVector[i],scheme)-u_exact(x_vec(L,dx),T,alpha)))
       if(label=='dx'):
           err[i]=max(abs(diff_3p_5p(L,T,discVector[i],dt,scheme)-u_exact(x_vec(L,discVector[i]),T,alpha)))
    return err

def compute_error_from_num(L,T,dx,dt,label,scheme='3pt'):
    alpha=0.5
    if(label=='dt'):
        discVector=dt
    elif(label=='dx'):
        discVector=dx
    err=np.zeros(discVector.shape[0])
    minVal=min(discVector)/10.0
    for i in range(discVector.shape[0]):
       if(label=='dt'):
           err[i]=max(abs(diff_3p_5p(L,T,dx,discVector[i],scheme)-diff_3p_5p(L,T,dx,minVal,scheme)))
       if(label=='dx'):
           err[i]=max(abs(diff_3p_5p(L,T,discVector[i],dt,scheme)-diff_3p_5p(L,T,minVal,dt,scheme)))
    return err



def default_plotting():
    plt.figure(1,figsize=[10,8])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=14)
    plt.ylabel('u(x)',size=18)
    plt.xlabel('x',size=18)
    tick_pos= [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4,  2*np.pi]
    labels = ['0','$\pi/4$','$\pi/2$','$3\pi/4$','$\pi$','$5\pi/4$','$3\pi/2$','$7\pi/4$','$2\pi$']
    plt.xticks(tick_pos, labels)


### ### ### ##
#MAIN_PROGRAM#
### ### ### ##
L=2.0*np.pi
alpha=0.5

#TIME EVOLUTION
default_plotting()
dx=2.0*np.pi/80
dt=0.01
plt.plot(x_vec(L,dx),u_exact(x_vec(L,dx),0,alpha),'k',label='t=0')
dx=2.0*np.pi/32
T=0.5
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'3pt'),'--b',label='t='+str(T)+', 3pt')
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'ob',label='t='+str(T)+', 5pt')
T=1.0
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'3pt'),'--r',label='t='+str(T)+', 3pt')
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'or',label='t='+str(T)+', 5pt')
T=2.0
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'3pt'),'--g',label='t='+str(T)+', 3pt')
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'og',label='t='+str(T)+', 5pt')
T=3.5
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'3pt'),'--y',label='t='+str(T)+', 3pt')
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'oy',label='t='+str(T)+', 5pt')
T=8.0
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'--c',label='t='+str(T)+', 3pt')
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'oc',label='t='+str(T)+', 5pt')
plt.title('Time evolution [$\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx))+' , $\\Delta$t='+str(dt)+']')
plt.gca().legend(loc=3)
#plt.show()
plt.savefig('TimeEvolution.png')
plt.close()



#ACCURACY OF APPROX OF TIME DERIVATIVE
#
T=2.0
dx=2.0*np.pi/20
#3-POINT_STENCIL
default_plotting()
plt.plot(x_vec(L,dx),u_exact(x_vec(L,dx),T,alpha),'k',label='ref. sol.')
dt=0.125
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'3pt'),'--oy',fillstyle='none',label='$\\Delta$t='+str(dt))
dt=0.25
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'3pt'),'--ob',fillstyle='none',label='$\\Delta$t='+str(dt))
dt=0.5
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'3pt'),'--or',fillstyle='none',label='$\\Delta$t='+str(dt))
dt=1.0
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'3pt'),'--og',fillstyle='none',label='$\\Delta$t='+str(dt))
plt.title('3pt-stencil vs Exact Solution at t='+str(T)+', $\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx)))
plt.gca().legend(loc=3)
plt.savefig('3pt_dt.png')
#plt.show()
plt.close()
#
#5-POINT-STENCIL
default_plotting()
plt.plot(x_vec(L,dx),u_exact(x_vec(L,dx),T,alpha),'k',label='ref. sol.')
dt=0.125
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'--oy',fillstyle='none',label='$\\Delta$t='+str(dt))
dt=0.25
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'--ob',fillstyle='none',label='$\\Delta$t='+str(dt))
dt=0.5
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'--or',fillstyle='none',label='$\\Delta$t='+str(dt))
dt=1.0
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'--og',fillstyle='none',label='$\\Delta$t='+str(dt))
plt.title('5pt-stencil vs Exact Solution at t='+str(T)+', $\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx)))
plt.gca().legend(loc=3)
plt.savefig('5pt_dt.png')
#plt.show()
plt.close()
#
#compute error
T=3.0
dx_err=2.0*np.pi/16.0
dtV=np.array([0.125/128.0,0.125/64.0,0.125/32.0,0.125/16.0,0.125/8.0,0.125/4.0,0.125/2.0,0.125])
err_5p_dt=compute_error_from_num(L,T,dx_err,dtV,'dt','5pt') 
err_3p_dt=compute_error_from_num(L,T,dx_err,dtV,'dt','3pt')
####


#ACCURACY OF APPROX OF SPACE DERIVATIVE
#
T=3.0
dt=0.01
#3-POINT_STENCIL
default_plotting()
dx=2.0*np.pi/40
plt.plot(x_vec(L,dx),u_exact(x_vec(L,dx),T,alpha),'k',label='ref. sol.')
dx=2.0*np.pi/32
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'--og',label='$\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx)))
dx=2.0*np.pi/16
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'--oy',label='$\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx)))
dx=2.0*np.pi/8
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'--ob',label='$\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx)))
dx=2.0*np.pi/4
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'5pt'),'--or',label='$\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx)))
plt.title('5pt-stencil vs exact solution at t='+str(T)+', dt='+str(dt))
plt.gca().legend(loc=3)
#plt.show()
plt.savefig('5pt_dx.png')
plt.ylim(0.18,0.245)
plt.xlim(np.pi/4,3*np.pi/4)
plt.title('(ZOOM) 5pt-stencil vs exact solution at t='+str(T)+', dt='+str(dt))
#plt.show()
plt.savefig('Zoom_5pt_dx.png')
plt.close()
#
#5-POINT_STENCIL
default_plotting()
dx=2.0*np.pi/40
plt.plot(x_vec(L,dx),u_exact(x_vec(L,dx),T,alpha),'k',label='ref, sol.')
dx=2.0*np.pi/32
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'3pt'),'--og',label='$\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx)))
dx=2.0*np.pi/16
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'3pt'),'--oy',label='$\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx)))
dx=2.0*np.pi/8
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'3pt'),'--ob',label='$\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx)))
dx=2.0*np.pi/4
plt.plot(x_vec(L,dx),diff_3p_5p(L,T,dx,dt,'3pt'),'--or',label='$\\Delta$x=$2\pi$/'+str(int((2.0*np.pi)/dx)))
plt.title('3pt-stencil vs exact solution at t='+str(T)+', $\\Delta$t='+str(dt))
plt.gca().legend(loc=3)
#plt.show()
plt.savefig('3pt_dx.png')
plt.ylim(0.20,0.30)
plt.xlim(np.pi/4,3*np.pi/4)
plt.title('(ZOOM) 3pt-stencil vs exact solution at t='+str(T)+', $\\Delta$t='+str(dt))
#plt.show()
plt.savefig('Zoom_3pt_dx.png')
plt.close()
#
##Compute Error
T=3.0
dt_err=0.00001
dxV=np.array([2.0*np.pi/64,2.0*np.pi/32,2.0*np.pi/16.,2.0*np.pi/8.,2.0*np.pi/4.])
err_3p_dx=compute_error_from_exactsol(L,T,dxV,dt_err,'dx','3pt')
err_5p_dx=compute_error_from_exactsol(L,T,dxV,dt_err,'dx','5pt')
#

#PLOTS-ERRORS
#
#Error ( dt )
plt.figure(1,figsize=[10,8])
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=16)
plt.ylabel('E($\\Delta$t)',size=20)
plt.xlabel('$\\Delta$t',size=18)
plt.loglog(dtV,err_3p_dt,'--sb',label='EE (time), 3-point stencil (space)')
plt.loglog(dtV,err_5p_dt,'--or',fillstyle='none',label='EE (time), 5-point stencil (space)')
annotation.slope_marker((3E-2, 0.002), (1, 1))
plt.gca().legend(loc=4)
plt.title('Error as function of $\\Delta$t, $\\Delta$x=2$\pi$/'+str(2.*np.pi/dx_err))
plt.savefig('Error_dt.png')
#plt.show()
plt.close()
#
#Error ( dt )
plt.figure(1,figsize=[10,8])
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=16)
plt.ylabel('E($\\Delta$x)',size=20)
plt.xlabel('$\\Delta$x',size=18)
plt.loglog(dxV,err_3p_dx,'--sb',label='EE (time), 3-point stencil (space)')
plt.loglog(dxV,err_5p_dx,'--or',fillstyle='none',label='EE (time), 5-point stencil (space)')
annotation.slope_marker((0.8, 0.012), (2, 1))
annotation.slope_marker((0.8, 0.0008), (4, 1))
plt.title('Error as function of $\\Delta$x, $\\Delta$t='+str(dt_err))
plt.savefig('Error_dx.png')
#plt.show()
plt.close()



print("Done.\n")






