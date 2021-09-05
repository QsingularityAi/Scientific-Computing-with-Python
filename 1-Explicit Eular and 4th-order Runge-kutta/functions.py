def decay_an(Tn,f,alpha,u0):
    import numpy as np
    u=f/alpha+np.exp(-alpha*Tn)*(u0-f/alpha)
    return u

def function_rs(x,f,alpha):
    import numpy as np
    u=f-alpha*x
    return u

def ExplicitEuler(Tn,f,alpha,u0):
    import numpy as np
    u=np.zeros(Tn.shape[0])
    u[0]=u0
    for i in range(0,Tn.shape[0]-1):
        u[i+1]=u[i]+(Tn[i+1]-Tn[i])*function_rs(u[i],f,alpha)
    return u

def RungeKutta4(Tn,f,alpha,u0):
    import numpy as np
    u=np.zeros(Tn.shape[0])
    u[0]=u0
    for i in range(0,Tn.shape[0]-1):
        k1=(Tn[i+1]-Tn[i])*function_rs(u[i],f,alpha)
        k2=(Tn[i+1]-Tn[i])*function_rs(u[i]+k1/2.0,f,alpha)
        k3=(Tn[i+1]-Tn[i])*function_rs(u[i]+k2/2.0,f,alpha)
        k4=(Tn[i+1]-Tn[i])*function_rs(u[i]+k3,f,alpha)
        u[i+1]=u[i]+(1.0/6.0)*(k1+2.0*k2+2.0*k3+k4)
    return u


