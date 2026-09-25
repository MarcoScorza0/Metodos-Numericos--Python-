
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from own_numerical_methods import nm_functions as nm_f



###-----------------------------------------###
#Ingresar matriz principal
A = np.array([
    [10.0, -1.0,  2.0,  0.0,  1.0],
    [-1.0, 11.0, -1.0,  3.0,  0.0],
    [ 2.0, -1.0, 10.0, -1.0, -1.0],
    [ 0.0,  3.0, -1.0,  8.0,  1.0],
    [ 1.0,  0.0, -1.0,  1.0, 12.0]
])
#Ingresar vector de términos independientes
B = np.array([4.0, 25.0, -9.0, 13.0, -21.0]) 
#Ingresar vector inicial
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 
#Ingresar tolerancia buscada (Usa norma infinito)
tol=1e-10
#Ingresar numero máximo de iteraciones
max_iter=20000




Solucion_GS,Niteraciones_GS,Error_GS,Radio_espectral_GS=nm_f.gauss_seidel(A,B,x0,max_iter,tol)
Solucion_JAC,Niteraciones_JAC,Error_JAC,Radio_espectral_JAC=nm_f.jacobi(A,B,x0,max_iter,tol)
print(f"\nSOLUCION POR GAUSS-SEIDEL:\n {Solucion_GS}\n Iteraciones: {Niteraciones_GS}\n")
print(f"\n\n\nSOLUCION POR JACOBI:\n {Solucion_JAC}\n Iteraciones: {Niteraciones_JAC}\n")
