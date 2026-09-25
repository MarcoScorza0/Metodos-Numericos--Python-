import numpy as np
import sympy as sp 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from own_numerical_methods import nm_functions as nm_f


x=sp.Symbol('x') ##defino la variable simbolica
funcion = x**3 - 2*x - 5  ##defino la funcion simbolica.
max_iteraciones = 1000  #defino numero maximo de iteraciones
tolerancia = 1e-10  #tolerancia
a = 2 #tope inferior del intervalo
b = 3 #tope superior del intervalo
x0 = 1.5 #valor inicial



print("\n\n\n\033[94m Criterio de Fourier para asegurar convergencia con x0 en [a,b]:"
    "\n\t#1 f(a)*f(b) < 0\n\t"
    "#2 f'(x) != 0 for all x on [a,b]\n\t"
    "#3 f''(x) != 0 for all x on [a,b]\n\t"
    "#4 f(x0)*f''(x0) > 0\n\n\n\033[0m")


print(f"\n\n\033[97mProbando Newton-Raphson con la función: {funcion} en el intervalo [{a},{b}]\033[0m\n\n")
    # Llamamos a la función



nm_f.newton_raphson(funcion, x0, tolerancia, max_iteraciones,a,b)


    
    




    





