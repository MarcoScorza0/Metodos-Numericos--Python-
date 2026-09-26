import numpy as np
import sympy as sp 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from own_numerical_methods import nm_functions as nm_f


x = sp.Symbol('x') ##defines symbolic variable

# Circuit parameters
V_s = 303.0      # Source voltage (V)
R = 15.0     # Resistor (Ohms)
I_s = 1e-12    # Diode saturation current (A)
V_T = 0.02585  # Thermal voltage at room temp (~26mV)

# Kirchhoff's Voltage Law (KVL) + Shockley Diode Equation
# V_s = I_D * R + V_D  =>  f(V_D) = I_s * (e^(V_D/V_T) - 1) * R + V_D - V_s = 0
# Where 'x' represents the diode voltage V_D
function = I_s * R * (sp.exp(x / V_T) - 1) + x - V_s  

max_iterations = 100  #defines maximum number of iterations
tolerance = 1e-10  #tolerance
a = 0.4 #lower bound of the interval (0.4V)
b = 0.9 #upper bound of the interval (0.9V)
x0 = 0.6 #initial guess for a silicon diode



print("\n\n\n\033[94m Criterio de Fourier para asegurar convergencia con x0 en [a,b]:"
    "\n\t#1 f(a)*f(b) < 0\n\t"
    "#2 f'(x) != 0 for all x on [a,b]\n\t"
    "#3 f''(x) != 0 for all x on [a,b]\n\t"
    "#4 f(x0)*f''(x0) > 0\n\n\n\033[0m")
print(f"\n\n\033[97mTesting Newton-Raphson with function: {function} in interval [{a},{b}]\033[0m\n\n")



x_iterated, iterations, error, x_history = nm_f.graph_newton_raphson(function, x, x0, tolerance, max_iterations, a, b,1)







    





