def GaussIni(xm,ym,A,B):
    import numpy as np
    return A*np.exp(-( ( xm[:,:] )**2 + ( ym[:,:] )**2 ) / (2*B**2) )

def PhiCircle(xm,ym,r,eps):
    import numpy as np
    return 0.5*(1-np.tanh(3.0*((xm[:,:]**2+ym[:,:]**2)**0.5-r)/eps))

def PBC(i,n):
        if (i<0):
                return int(i+n-1)
        elif (i>n-2):
                return int(i-n+1)
        else:
                return int(i)

def d1f_5pst(u,dx,label='x'):
    import numpy as np
    n=u.shape[0]
    d1f=np.zeros(u.shape)
    for i in range(0,n):
        for j in range(0,n):
            if(label=='x'):
                d1f[i,j]=(-u[PBC(i+2,n),j]+8*u[PBC(i+1,n),j]-8*u[PBC(i-1,n),j]+u[PBC(i-2,n),j])/(12*dx)
            if(label=='y'):
                d1f[i,j]=(-u[i,PBC(j+2,n)]+8*u[i,PBC(j+1,n)]-8*u[i,PBC(j-1,n)]+u[i,PBC(j-2,n)])/(12*dx)
    return d1f

def d2f_5pst(u,dx,label='x'):
    import numpy as np
    n=u.shape[0]
    d2f=np.zeros(u.shape)
    for i in range(0,n):
        for j in range(0,n):
            if(label=='x'):
                d2f[i,j]=(-u[PBC(i+2,n),j]+16*u[PBC(i+1,n),j]-30*u[i,j]+16*u[PBC(i-1,n),j]-u[PBC(i-2,n),j])/(12*dx*dx)
            if(label=='y'):
                d2f[i,j]=(-u[i,PBC(j+2,n)]+16*u[i,PBC(j+1,n)]-30*u[i,j]+16*u[i,PBC(j-1,n)]-u[i,PBC(j-2,n)])/(12*dx*dx)
    return d2f


def NormGrad(f,ds):
    import numpy as np
    d1xU=d1f_5pst(f,ds,'x')
    d1yU=d1f_5pst(f,ds,'y')
    return (d1xU**2+d1yU**2)**0.5

def DivPhiGradU(u,phi,ds):
    import numpy as np
    Div_phi_Grad_u=np.zeros(u.shape)
    Div_phi_Grad_u=np.zeros(u.shape)
    d1xU=d1f_5pst(u,ds,'x') # du(x,y)/dx 
    d1yU=d1f_5pst(u,ds,'y') # du(x,y)/dy
    return d1f_5pst(phi*d1xU,ds,'x')+d1f_5pst(phi*d1yU,ds,'y') # d( phi * du(x,y)/dx ) / dx + d( phi * du(x,y)/dy ) / dy


def IntegrationScheme(u,phi,ds,dt,f,g):
    alpha=1.0
    return u[:,:]+dt*(alpha*DivPhiGradU(u,phi,ds)+g*NormGrad(phi,ds)+f*phi)*(1.0/(phi+1E-6))






