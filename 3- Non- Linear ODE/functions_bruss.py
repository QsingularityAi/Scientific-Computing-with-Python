def bruss_1(U,t,A,B):
    X,Y=U
    dUdt=[A-(B+1)*X+Y*X**2,B*X-Y*X**2]
    return dUdt

def bruss_2(U,t,A,v):
    X,Y,Z=U
    dUdt=[A-(Z+1)*X+Y*X**2,Z*X-Y*X**2,-X*Z+v]
    return dUdt



