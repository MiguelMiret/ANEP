# Práctica 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def variables(M,N,T):
    #separamos esto para no sobrecargar la funcion main
    xmin = 0; xmax = 1; x = np.linspace(xmin,xmax,N+1)
    tmin = 0; tmax = T; t =np.linspace(tmin,tmax,M+1)
    return(x,t)

def buildmatrix(Theta,Lamda,N):
    a = 1 + 2*Theta*Lamda
    b = -Theta*Lamda
    A = np.zeros([N+1,N+1])
    v = np.zeros(N+1)
    v[0] = b
    v[1] = a
    v[2] = b

    #Cuerpo central de la matriz

    for i in range(1,N):
        d = np.roll(v,i-1)
        A[i] = d
    
    #Condiciones de contorno

    A[0,0] = a
    A[N,N] = a
    A[0,1] = 2*b
    A[N,N-1] = 2*b

    return(A)

def sourcematrix(t,x):
    f = np.zeros([np.size(t),np.size(x)])
    for i,ti in enumerate(t):
        if ti <= 1:
            for j,xj in enumerate(x):
                f[i,j] = (1-ti)**2*xj*(1-xj)
    return(f)


def solutionmatrix(t,x):
    u = np.zeros([np.size(t),np.size(x)])
    # Condicion inicial
    u[0] = np.multiply(x**2,(1-x)**2)
    return(u)

def plotting(u,x,t):
    plt.pcolormesh(x, t, u)
    plt.show()

def main():  #problema neumann
    N = 200
    M = 100
    T = 2
    (x,t) = variables(M,N,T)
    dt = t[1]-t[0]
    dx = x[1]-x[0]

    #valor de theta y lambda
    Theta = 1
    Lamda = dt/dx**2
    print(Lamda)
    

    #matriz
    (A) = buildmatrix(Theta,Lamda,N)
    print(A)

    #matriz solución
    u = solutionmatrix(t,x)

    #matriz fuente
    f = sourcematrix(t,x)

    #construimos el vector d
    d = np.zeros([np.size(t),np.size(x)])
    for i in range(0,M):
        for j in range (1,N):  #rellenamos el vector d en el tronco central
            d[i,j] = u[i,j]*(1-2*(1-Theta)*Lamda) + (1-Theta)*Lamda*(u[i,j-1]+u[i,j+1]) + dt*Theta*f[i+1,j] + dt*(1-Theta)*f[i,j]
        
        #ahora lo rellenamos en los bordes teniendo en cuenta las condiciones de contorno
        d[i,0] = u[i,0]*(1-2*(1-Theta)*Lamda) + (1-Theta)*Lamda*2*u[i,1] + dt*Theta*f[i+1,0] + dt*(1-Theta)*f[i,0]
        d[i,N] = u[i,N]*(1-2*(1-Theta)*Lamda) + (1-Theta)*Lamda*2*u[i,N-1] + dt*Theta*f[i+1,N] + dt*(1-Theta)*f[i,N]

        #ahora resolvemos el sistema lineal que nos queda para cada fila

        u[i+1] = np.linalg.solve(A,d[i])

   
    # Ploteamos el resultado
    plotting(u,x,t)

    U = np.zeros(M+1)
    for i in range (0,M+1):
        U[i] = np.sum(u[i])/(N+1)  #equivalente a integrar u
    
    plt.plot(t,U)
    plt.show()


    #exportamos a excel

    data = pd.DataFrame(u)
    data.to_excel('Data_Neuman.xlsx', sheet_name='Neuman', index=False)





        
        




    











if __name__ == '__main__':
	main()

