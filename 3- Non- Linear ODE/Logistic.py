import numpy as np
from matplotlib.pyplot import *
import sys
from functions_nonlinear import *

#Total duration
T = 9

#Reading parameters or setting default values
try:
    dt = float(sys.argv[1])
    eps_r = float(sys.argv[2])
    omega = float(sys.argv[3])
except:
    dt = 0.8
    eps_r = 1E-3
    omega = 1

#Total number of timesteps
N = int(round(T/float(dt)))

#Compute solutions with different methods
u_FE = FE(0.1, dt, N)
u_BE_E, iter_BE2 = BE(0.1, dt, N, 'r2')
u_BE_P1, iter_BE_P1 = BE(0.1, dt, N, 'Picard', eps_r, omega,1)
u_BE_P, iter_BE_P = BE(0.1, dt, N, 'Picard', eps_r, omega)
u_BE_N1, iter_BE_N1 = BE(0.1, dt, N, 'Newton', eps_r, omega,1)
u_BE_N, iter_BE_N = BE(0.1, dt, N, 'Newton', eps_r, omega)

print('Picard mean no of iterations (dt=%g):' % dt, int(round(np.mean(iter_BE_P))))
print('Newton mean no of iterations (dt=%g):' % dt, int(round(np.mean(iter_BE_N))))

#Plotting Figure u(t) vs t
filename = 'logistic_dt'+str(dt)+'_eps'+str(eps_r)+'_omega'+str(omega)
rc('text', usetex=True)
rc('font', family='serif',size=13)
figure(1)
title('$\\Delta t$='+str(dt)+', $\\epsilon_r=$'+str(eps_r)+', $\omega=$'+str(omega))
xlabel('t')
ylabel('u(t)')
t = np.linspace(0, dt*N, N+1)
plot(t, u_FE,'-o',label='FE')
plot(t, u_BE_E,'-s',label='BE Exact')
plot(t, u_BE_P,'--X',label='BE Picard')
plot(t, u_BE_P1,':^',label='BE Pic. k=1')
plot(t, u_BE_N,'--*',label='BE Newton')
plot(t, u_BE_N1,':v',label='BE New. k=1')
legend()
#show()
savefig(filename + '_u.png')

#Plotting # of Iteration vs # timesteps
figure(2)
title('$\\Delta t$='+str(dt)+', $\\epsilon_r=$'+str(eps_r)+', $\omega=$'+str(omega))
xlabel('timesteps')
ylabel('No. of iterations')
plot(range(1, len(iter_BE_P)+1), iter_BE_P, 'r-o',label='Picard')
plot(range(1, len(iter_BE_N)+1), iter_BE_N, 'b-o',label='Newton')
legend()
#show()
savefig(filename + '_iter.png')
