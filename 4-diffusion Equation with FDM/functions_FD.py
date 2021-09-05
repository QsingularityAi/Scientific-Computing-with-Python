def u_exact(x,t,alpha):
    import numpy as np
    return np.sin(x)*np.exp(-alpha*t)


def PBC(i,n):
        if (i<0):
                return int(i+n-1)
        elif (i>n-2):
                return int(i-n+1)
        else:
                return int(i)

def d2u_3ptstencil(uold,dx,dt,alpha,L):
    import numpy as np
    n=uold.shape[0]
    unew=np.zeros(n)
    s=alpha*(dt/(dx**2.0))
    for i in range(0,n):
        unew[i]=uold[i]+s*(uold[PBC(i+1,n)]+uold[PBC(i-1,n)]-2.0*uold[i])
    return unew

def d2u_5ptstencil(uold,dx,dt,alpha,L):
    import numpy as np
    n=uold.shape[0]
    unew=np.zeros(n)
    s=alpha*(dt/(12.*dx**2.0))
    for i in range(0,n):
        unew[i]=uold[i]+s*(-uold[PBC(i+2,n)]-uold[PBC(i-2,n)]+16.0*uold[PBC(i+1,n)]+16.0*uold[PBC(i-1,n)]-30.0*uold[i])
    return unew




