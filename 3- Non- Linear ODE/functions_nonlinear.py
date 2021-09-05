import numpy as np

def Residual(u,a,b,c):
    return a*u**2 + b*u + c

def FE(u0, dt, Nt):
    #
    #Forward Euler 
    #
    u = np.zeros(Nt+1)
    u[0] = u0
    for n in range(Nt):
        u[n+1] = u[n] + dt*(u[n] - u[n]**2)
    return u

def quadratic_roots(a, b, c):
    delta = b**2 - 4*a*c
    r2 = (-b + np.sqrt(delta))/float(2*a)
    r1 = (-b - np.sqrt(delta))/float(2*a)
    return r1, r2

def BE(u0, dt, Nt, choice='Picard',eps_r=1E-3, omega=1, max_iter=1000):
    #
    #Backward Euler
    #choice: - Picard, 
    #        - Newtown, 
    #        - r1,r2, exact solution of the 2nd-order algebric Eq.as defined in "quadratic_roots" 
    #
    u = np.zeros(Nt+1)
    iterations = []
    u[0] = u0
    for n in range(1, Nt+1):
        a = dt
        b = 1 - dt
        c = -u[n-1]
        if choice in ('r1', 'r2'):
            r1, r2 = quadratic_roots(a, b, c)
            u[n] = r1 if choice == 'r1' else r2
            iterations.append(0)

        u_ = u[n-1]
        k = 0
        if choice == 'Picard':
            #
            while abs(Residual(u_,a,b,c)) > eps_r and k < max_iter:
                u_ = omega*(-c/(a*u_ + b)) + (1-omega)*u_
                k += 1
            u[n] = u_
            iterations.append(k)

        elif choice == 'Newton':
            #
            while abs(Residual(u_,a,b,c)) > eps_r and k < max_iter:
                u_ = u_ - omega*Residual(u_,a,b,c)/(2*a*u_ + b)
                k += 1
            u[n] = u_
            iterations.append(k)

    return u, iterations
