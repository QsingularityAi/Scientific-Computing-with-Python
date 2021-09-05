def u_exact(x,t,alpha):
    import numpy as np
    return np.sin(x)*np.exp(-alpha*t)

def ExplicitEulerFT(u,k,C,dt):
    import numpy as np
    res=u+dt*C*k*u
    return res

def ImplicitEulerFT(u,k,C,dt):
    import numpy as np
    res=u/(1.0-dt*C*k)
    return res

def IniSinCos(x,y,kx,ky,A,B):
        import numpy as np
        res=A*np.sin(kx*x)+B*np.cos(ky*y)
        return res

def ImplicitEuler2D_FT(u,kx2,ky2,C,dt):
        import numpy as np
        res=u/(1.0-dt*C*(kx2+ky2))
        return res



