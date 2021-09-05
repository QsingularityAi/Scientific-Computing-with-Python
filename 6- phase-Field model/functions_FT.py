def IniRand(x,y,vmin,vmax,seed):
    import numpy as np
    import scipy as sp
    import scipy.ndimage	
    np.random.seed(seed)
    res=np.random.rand(x.shape[0],x.shape[0])*(vmax-vmin)+vmin
    res=ndimage.filters.gaussian_filter(res, 2, mode='nearest')
    return res

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

def NormGrad2(f,ds):
    import numpy as np
    d1xU=d1f_5pst(f,ds,'x')
    d1yU=d1f_5pst(f,ds,'y')
    return (d1xU**2+d1yU**2)

def Energy(u,eps,ds):
    EE=(18/eps)*((1-u)**2)*(u**2)+eps*NormGrad2(u,ds)
    return sum(sum(EE))

def ImplicitEuler2D_FT(u,kx2,ky2,C,dt):
    import numpy as np
    res=u/(1.0-dt*C*(kx2+ky2))
    return res

def IniRand(x,y,vmin,vmax,s):
    import numpy as np
    import scipy as sp
    import scipy.ndimage	
    np.random.seed(s)
    res=(vmax-vmin)*np.array([[np.random.random() for e in list(range(x.shape[0]))] for e in list(range(x.shape[0]))])+vmin
    #res=sp.ndimage.filters.gaussian_filter(res, [5.0,5.0], mode='constant')
    return res

def AllenCahn2D_FT(u,u2,u3,kx2,ky2,C,dt,eps):
    import numpy as np
    #36 x - 108 x^2 + 72 x^3
    res=(u+dt*108*u2/eps-dt*72*u3/eps)/(1.0-dt*eps*C*(kx2+ky2)+36.0*dt/eps)
    #(u+108*u2/eps-72*u3/eps)/(1.0-dt*eps*C*(kx2+ky2)+36.0*dt/eps)
    return res

def CahnHilliard2D_FT(u,u2,u3,kx2,ky2,C,dt,eps):
    import numpy as np
    res=(u-dt*(kx2+ky2)*108*u2/eps+(kx2+ky2)*dt*72*u3/eps)/(1.0+dt*eps*C*(kx2**2+ky2**2)-36.0*(kx2+ky2)*dt/eps)
    #(u-dt*u2*(kx2+ky2))/(1.0-dt*(kx2**2+ky2**2)-dt*(kx2+ky2))
    return res


