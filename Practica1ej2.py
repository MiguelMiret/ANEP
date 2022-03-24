import numpy as np
import matplotlib.pyplot as plt
from numba import jit, int64, float64


def metodoUno(x,h,i,a):
    d = (a[0]*np.cos(x[i])+a[1]*np.cos(x[i]+h)+a[2]*np.cos(x[i]+2*h)+a[3]*np.cos(x[i]+3*h))/h
    return d

#para mantener un orden de error parecido (el mismo) completamos la serie a calcular con el método backawards de mismo orden


def metodoUno2(x,h,i,a):
    d = (-a[0]*np.cos(x[i])-a[1]*np.cos(x[i]-h)-a[2]*np.cos(x[i]-2*h)-a[3]*np.cos(x[i]-3*h))/h
    return d

@jit([float64(float64[:],float64,int64,float64[:])],nopython=True)
def metodoDos(x,h,i,b):  
    d = (b[0]*np.cos(x[i]-2*h)+b[1]*np.cos(x[i]-h)+b[2]*np.cos(x[i]+h)+b[3]*np.cos(x[i]+2*h))/h
    return d

def calculo(n,a,b):  #función que hace los cálculos dado h
    h = 2*np.pi/n
    print('El h elegido es: ', h)
    x = np.linspace(0, 2*np.pi, n)
    s = -np.sin(x)
    dUno = np.zeros(n)
    dDos = np.zeros(n)

    for i in range (0,n-3):               # casos que se pueden hacer con el forward
        dUno[i] = metodoUno(x,h,i,a)

    for i in range (n-3, n):              # casos restantes
        dUno[i] = metodoUno2(x,h,i,a)
    
    for i in range (2,n-2):               # casos que se pueden hacer con las centradas
        dDos[i] = metodoDos(x,h,i,b)
    
    for i in range (0, 2):                # casos restantes A
        dDos[i] = metodoUno(x,h,i,a)
    
    for i in range (n-2, n):              # casos restantes B
        dDos[i] = metodoUno2(x,h,i,a)
    


    eUno= abs(s-dUno)  #error método uno

    eDos = abs(s-dDos)  #error método dos

    #quitamos los errores de otros métodos

    eUno = np.delete(eUno,(n-3,n-2,n-1))

    eDos = np.delete(eDos,(0,1,n-2,n-1))


    return(dUno,dDos,eUno,eDos)
        


def main():
    a = np.array([-11/6, 3, -3/2, 1/3])
    b = np.array([1/12, -2/3, 2/3, -1/12 ])
    NArray = [10,50,100,500,1000,5000]  #añadiendo 10000,50000,100000,500000] para el plot
    l = np.size(NArray)
    HArray = np.divide(2*np.pi,NArray)
    eMaxUno = np.zeros(l)
    eMaxDos = np.zeros(l)
    for i in range (0,l):
        (dUno, dDos, eUno, eDos) = calculo(NArray[i],a,b)
        eMaxUno[i] = np.max(eUno)
        eMaxDos[i] = np.max(eDos)
    

    plt.subplot(2,1,1)
    plt.plot(np.log10(HArray),np.log10(eMaxUno))
    
    coefUno = np.polyfit(np.log10(HArray),np.log10(eMaxUno),1)
    print(coefUno)

    plt.subplot(2,1,2)
    plt.plot(np.log10(HArray),np.log10(eMaxDos))
    plt.show()

    coefDos = np.polyfit(np.log10(HArray),np.log10(eMaxDos),1)
    print(coefDos)



if __name__ == '__main__':
	main()

