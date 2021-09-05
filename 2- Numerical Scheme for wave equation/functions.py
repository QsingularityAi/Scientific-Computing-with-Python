def u0(x):
	import numpy as np
	u=np.sin(2*np.pi*x)+0.25*np.sin(10*np.pi*x)
	return u

def u1(x):
	import numpy as np
	u=np.zeros(x.shape)
	return u

def PBC(i,n):
	if (i<0):
		return int(i+n+1)	
	elif (i>n-1):
		return int(i-n-1)
	else:		
		return int(i)

def integration_scheme_wave(un,un_old,dx,dt,c):
	import numpy as np
	n=un.shape[0]
	u=np.zeros(n)
	s=c*dt/dx
	for i in range(1,n-1):
		u[i]=2*(1-s**2)*un[i]+(s**2)*(un[PBC(i-1,n)]+un[PBC(i+1,n)])-un_old[i]
	return u



