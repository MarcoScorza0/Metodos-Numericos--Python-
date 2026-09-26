import numpy as np
import sympy as sp 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#VECTOR TO MATRIX FUNCTION.
def vector_to_matrix(data,rows,columns): #Receives an array and converts it into a matrix of n rows and m columns ("filas" and "columnas")
    vector=np.array(data,dtype=float)  #Defines a numpy array called "vector" and makes it a float array.
    if len(vector)!=rows*columns:  #Verifies the quantity of values inserted in the vector correlates with the dimensions of the matrix indicated
        raise ValueError(f"Error: The number of elements in the vector ({len(vector)}) does not match the order of the matrix ({rows*columns}).") #if doesnt match, it sends an error.
    else:
        matrix=vector.reshape((rows,columns)) #Gives the array the shape in correspondence with the matrix´s dimension.
        return matrix #Returns the matrix reshaped.


##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#MATRIX VALIDATION FUNCTION.
def matrix_validation(matrix):
    matrix=np.asarray(matrix,dtype=float)
    if matrix.ndim>2 or matrix.ndim==0 or len(matrix)==0:
        raise ValueError(f"Error: invalid matrix")
    else:
        return matrix


##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#SQUARE MATRIX VALIDATION
def sqr_matrix_validation(matrix):
    matrix=matrix_validation(matrix)
    if matrix.ndim==1:
        order=np.sqrt(len(matrix))
        if order!=int(order):
            raise ValueError(f"Error: The number of elements in the matrix must have a perfect square root. Must introduce a square matrix")
        else:
            order=int(order)
            matrix=vector_to_matrix(matrix,order,order)
    rows,columns=matrix.shape
    if rows!=columns:
        raise ValueError(f"Error: Matrix is not a square matrix.")
    else:
        return matrix


##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#SPECTRAL_RADIUS_Function.
def spectral_radius(matrix):
    matrix=sqr_matrix_validation(matrix)
    eigenvalues=np.linalg.eigvals(matrix)
    return np.max(np.abs(eigenvalues))


##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#JACOBI's method function
# NOTE: Inputs for this function can be NumPy arrays or Python lists. The "vector_to_matrix" function can be used for this purpose if necessary.
def jacobi(principal_matrix,independent_terms_matrix,initial_vector,max_iterations,tolerance):  #approximates a linear system Ax=b using Jacobi's iterative method.
  
    #1. Converts inputs to numpy arrays.
    principal_matrix = np.asarray(principal_matrix, dtype=float)  #Converts principal_matrix to a numpy array.
    independent_terms_matrix = np.asarray(independent_terms_matrix, dtype=float)  #Converts independent_terms_matrix to a numpy array.
    initial_vector = np.asarray(initial_vector, dtype=float)  #Converts initial_vector to a numpy array.

    #2. Converts 1D vectors to column vectors.
    if independent_terms_matrix.ndim == 1:
        independent_terms_matrix = independent_terms_matrix.reshape(-1, 1) #Converts 1D array to column array.
    if initial_vector.ndim == 1:
        initial_vector = initial_vector.reshape(-1, 1) #Converts 1D array to column array.

    #3. Verifies and converts (if necessary) 'principal_matrix' vector into matrix
    if principal_matrix.ndim==1: #Verifies if principal_matrix is a 1D array. In that case, must be converted into a nxn matrix.
        order=np.sqrt(len(principal_matrix)) #Calculates the root square for the number of elements in that array.
        if order!=int(order):  #Verifies
            raise ValueError(f"Error: The number of elements in 'principal_matrix' must have a perfect square root.")
        else:
            principal_matrix=vector_to_matrix(principal_matrix,int(order),int(order))
    
    #4. Verifies dimensions of every given matrix.
    rows_A, cols_A = principal_matrix.shape   #Gives the corresponding dimensions to matrix A.
    rows_b, cols_b = independent_terms_matrix.shape  #Same for matrix b.
    rows_x0, cols_x0 = initial_vector.shape #Same for initial_vector.
    if rows_A != cols_A or rows_b != rows_A or cols_b != 1 or rows_x0 != rows_A or cols_x0 != 1:  #Verifies dimensions of every given matrix.
        raise ValueError(f"Incorrect dimensions: principal_matrix must be of dimensions ({rows_A}, {rows_A}), and vectors must be of dimensions ({rows_A}, 1).") #Indicates error if dimensions don't match.
  
    #5. Verifies necessary conditions before applying Jacobi's method.
    if np.any(np.diag(principal_matrix)==0): #Method cant be used if the main diagonal contains zero in it.
        raise ValueError(f"Jacobi's method can't be used. The matrix D can't be inverted.") #Indicates an error if the previous verification fails.
    D=np.diag(np.diag(principal_matrix))  #Extracts diagonal matrix.
    L=-np.tril(principal_matrix)+D  #Extracts lower matrix.
    U=-np.triu(principal_matrix)+D  #Extracts upper matrix.
    D_inv=np.linalg.inv(D)  #Inverts the D matrix.
    
    #6. Proceeds with iterative method and verifications (including convergence criteria).
    TJ=D_inv @ (L+U)   #Calculates TJ matrix.
    CJ=D_inv @ independent_terms_matrix  #Same for CJ matrix.
    rho=spectral_radius(TJ) #Calculates spectral radius of TJ matrix    
    if rho>=1: #Checks spectral radius for ensuring convergence.
        raise ValueError(f"Error: The method will not converge because of spectral radius criteria. Can not use Jacobi's method.")
    matrix_k=initial_vector  #Defines first k-th matrix as the initial vector.
    k=0 #Sets iteration counter.
    infinite_norm=float('inf') #Initializes infinite_norm as float('inf') to ensure the norm condition is satisfied.
    while k<max_iterations and infinite_norm>tolerance: #Keeps iterating as the number of iterations and error is less than specified values.
        matrix_xnew=TJ @ matrix_k + CJ  #Calculates the K-th matrix of unknowns.
        k=k+1 #Adds 1 to counting variable.
        infinite_norm=np.max(np.abs(matrix_xnew - matrix_k)) #Calculates the infinite_norm for the current K-th unknown matrix.
        matrix_k=matrix_xnew #Sets the K matrix as the new matrix to calculate next iteration
        if np.isnan(infinite_norm) or np.isinf(infinite_norm): #Checks if new values turned infinite.
            print(f"Jacobi's method has DIVERGED (values turned infinite/nan in iteration {k})") #Indicates divergence error.
            return None,k,infinite_norm #Returns none, and extra data from last iteration. Also stops iterations.

    #7. Final check for different scenarios.
    if(k==max_iterations): #Checks if maximum number of iterations have been reached.
        print(f"Too many iterations. Jacobi's method did not converge. Maximum iterations have been reached ({k})") #Indicates error for max iterations reached.
        return None,k,infinite_norm #Returns none and extra data from last iteration.
    print(f"Jacobi's method CONVERGED successfully: \nx=\n{matrix_xnew}\nITERATIONS={k}\nERROR IN NORM=\n\t{infinite_norm}") #Indicates method converged successfully.
    
    return matrix_xnew,k,infinite_norm,rho  #Returns final k-th matrix of unknowns, and some extra data from last iteration.


##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## Gauss-Seidel's method function
def gauss_seidel(principal_matrix,independent_terms_matrix,initial_vector,max_iterations,tolerance):

    #1. Converts inputs to numpy arrays.
    principal_matrix = np.asarray(principal_matrix, dtype=float)  #Converts principal_matrix to a numpy array.
    independent_terms_matrix = np.asarray(independent_terms_matrix, dtype=float)  #Converts independent_terms_matrix to a numpy array.
    initial_vector = np.asarray(initial_vector, dtype=float)  #Converts initial_vector to a numpy array.

    #2. Converts 1D vectors to column vectors.
    if independent_terms_matrix.ndim == 1:
        independent_terms_matrix = independent_terms_matrix.reshape(-1, 1) #Converts 1D array to column array.
    if initial_vector.ndim == 1:
        initial_vector = initial_vector.reshape(-1, 1) #Converts 1D array to column array.

    #3. Verifies and converts (if necessary) 'principal_matrix' vector into matrix
    if principal_matrix.ndim==1: #Verifies if principal_matrix is a 1D array. In that case, must be converted into a nxn matrix.
        order=np.sqrt(len(principal_matrix)) #Calculates the root square for the number of elements in that array.
        if order!=int(order):  #Verifies perfect square.
            raise ValueError(f"Error: The number of elements in 'principal_matrix' must have a perfect square root.")
        else:
            principal_matrix=vector_to_matrix(principal_matrix,int(order),int(order)) 
    
    #4. Verifies dimensions of every given matrix.
    rows_A, cols_A = principal_matrix.shape   #Extracts the number of rows and columns of the principal matrix to the variables rows_A and cols_A
    rows_b, cols_b = independent_terms_matrix.shape  #Same for matrix b.
    rows_x0, cols_x0 = initial_vector.shape #Same for initial_vector.
    if rows_A != cols_A or rows_b != rows_A or cols_b != 1 or rows_x0 != rows_A or cols_x0 != 1:  #Verifies dimensions of every given matrix.
        raise ValueError(f"Incorrect dimensions on the matrix's system).") #Indicates error if dimensions don't match.
  
    #5. Verifies necessary conditions before applying Gauss-Seidel's method.
    if np.any(np.diag(principal_matrix)==0): #Method can't be used if the main diagonal contains zero in it.
        raise ValueError(f"Gauss-Seidel's method can't be used.") #Indicates an error if the previous verification fails.
    D=np.diag(np.diag(principal_matrix))  #Extracts diagonal matrix.
    L=-np.tril(principal_matrix,k=-1)  #Extracts lower matrix.
    U=-np.triu(principal_matrix,k=1)  #Extracts upper matrix.

    #6 Constructs Tgs to verify convergence with spectral radius criteria.
        # Given Tgs=(D-L)^(-1)U, to avoid inverting a matrix we can use:
            # (D-L)^(-1) * U = Tgs so U=(D-L)*Tgs and this linear sistem is easier to solve than inverting a matrix.
            #The solution to this system is gonna be Tgs matrix, the one we want to use for aplying convergence criteria of spectral radius
    Tgs = np.linalg.solve((D-L),U) #Solves the linear system for finding Tgs matrix.
    spec_radius = spectral_radius(Tgs) #Calculates the spectral radius of Tgs
    if spec_radius>=1: #Verifies criteria.
        raise ValueError(f"Error: Method will not converge. Spectral Radius of Tgs is {spec_radius} (must be <1 to converge)")

    #7 Starts iteration (Avoiding usage of matrix Tgs and Cgj, not for any utility reason but for educational ones)
    infinite_norm=float('inf') #Initialize infinite_norm to ensure next loop starts.
    k=0 #Initialize counter for loop.
    x=np.copy(initial_vector) #Copies unknowns initial vector for usage in first iteration.
    D_vector=np.diag(principal_matrix)  #Converts the diagonal of principal_matrix into an array for calculation purpose.
    while infinite_norm>tolerance and k<max_iterations: #Starts iteration.
        xold_copy=np.copy(x) #Copies x before writing it cause original vector is needed for calculating infinite norm before cycle.
        for i in range(rows_A): #Calculates every new unknown value using Gauss-Seidel's formula and writes upon the original unknowns vector while calculates every new x value.
            x[i,0]=(1/(D_vector[i]))*(independent_terms_matrix[i,0]-(sum(principal_matrix[i,j]*x[j,0] for j in range(rows_A) if j != i)))
        k=k+1
        infinite_norm=np.max(np.abs(x-xold_copy)) #Calculates error for stop criteria.
    return x,k,infinite_norm,spec_radius  #Returns vector of unknowns, number of iterations, error in norm and spectral radius

##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




##NEWTON-RAPHSON'S method
def newton_raphson(function,var,x0,tolerance,max_iterations,a,b):
    '''
    Introduce 'function' argument as a symbolic expression based on sympy library.

     Fourier criteria for ensuring convergence of continuous functions on [a,b]:
        #1 f(a)*f(b) < 0
        #2 f'(x) != 0 for all x on [a,b]
        #3 f''(x) != 0 for all x on [a,b]
        #4 f(x0)*f''(x0) > 0
    For this tests, it is necessary to discretize the interval [a,b] in order to apply the criteria.
    If the criteria is not met, cannot ensure convergence.

    '''

    f_def=sp.lambdify(var,function,'numpy') #converts the symbolic function to a numpy function.
    fprime_def=sp.lambdify(var,sp.diff(function,var,1),'numpy') #Finds the derivative of 'function'
    f_secprime_def=sp.lambdify(var,sp.diff(function,var,2),'numpy')#Same for second derivative (needed for convergence criteria).
    disc_interval=np.linspace(a,b,100000) #Discretize the interval [a,b]

    #1. First criterion from Fourier:
    if (f_def(a)*f_def(b))<0:
        criteria_1=1 # if 1, then it is satisfied
    else:
        if f_def(a)==0 or f_def(b)==0:
            raise ValueError ("One of the boundaries of the interval is indeed a root")
        else:
            criteria_1=0 # if 0, then it is not satisfied

    #2 Second criterion from Fourier:
    first_derivative_array=fprime_def(disc_interval) #calculates the derivative of every element of the discrete interval
    if np.any((first_derivative_array[:-1]*first_derivative_array[1:])<0):
        '''
        Multiplies every N-th element of the first-derivative discrete array for the (N+1)-th element.
        If any product is less than 0, this means that, by Bolzano, between both discrete elements there's a root.
        In this case, the criteria is not met.
        '''
        criteria_2=0
    else:
        criteria_2=1

    #3 Third criterion from Fourier:
        #Exactly the same procedure as step #2 applied on second derivative.
    second_derivative_array=f_secprime_def(disc_interval)
    if np.any((second_derivative_array[:-1]*second_derivative_array[1:]) < 0):
        criteria_3=0
    else:
        criteria_3=1

    #4 Fourth criterion from Fourier:
    if f_def(x0)*f_secprime_def(x0)>0:
        criteria_4=1
    else:
        criteria_4=0


    if criteria_1==1 and criteria_2==1 and criteria_3==1 and criteria_4==1:
        print("\033[96m\n\n\t\t\tFourier criteria are satisfied. Convergence is ensured\n\n\033[0m")
    else:
        print("\033[91m\n\n\t\t\tFourier criteria are not satisfied. Convergence is not ensured\n\n\033[0m")



    print("\n\n\n\033[32mStarting iteration:\n\n\033[0m")
    k=0
    xold=x0
    error=float('inf')
    x_history=[x0]
    while tolerance<error and k<max_iterations:
        xnew=xold - (f_def(xold)/fprime_def(xold))
        x_history.append(xnew)
        error=abs(xnew-xold)
        xold=xnew
        k=k+1
    if k==max_iterations:
        print("\033[31m\n\n\t\t\tMaximum number of iterations reached\n\n\033[0m")
    else:
        print("\033[32m\n\n\t\t\tIndicated tolerance has been reached\n\n\033[0m")
        print(f"\n\n\n\033[32m\t\t\tThe root is: {xnew}\033[0m")
        print(f"\n\n\n\033[32m\t\t\tNumber of iterations: {k}\033[0m")
        print(f"\n\n\n\033[32m\t\t\tError: {error}\033[0m")

    return xnew,k,error,x_history
    



# Graph newthon raphson function.
def graph_newton_raphson(function,var,x0,tolerance,max_iterations,a,b, damping):
    # We call the pure method just so it calculates and prints the real mathematical results to the console.
    real_x, real_k, real_error, real_history = newton_raphson(function, var, x0, tolerance, max_iterations, a, b)

    f_def = sp.lambdify(var, function, 'numpy')
    fprime_def = sp.lambdify(var, sp.diff(function, var, 1), 'numpy')
    
    k = 0
    xold = x0
    error = float('inf')
    _x_history = [x0]
    while tolerance < error and k < max_iterations:
        xnew = xold - damping * (f_def(xold) / fprime_def(xold))
        _x_history.append(xnew)
        error = abs(xnew - xold)
        xold = xnew
        k += 1
        
    _x_approximation = xnew
    _iterations = k
    _error = error
    
    #Plot graph config
    # 1. Configures window and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 2. Generates points for the function curve
    # Makes the range wider than [a;b] for better visualization
    margin = 1.0
    x_vals = np.linspace(a - margin, b + margin, 400)
    
    # Since 'function' is a SymPy symbolic expression, we use lambdify to convert it
    # into a numeric function that NumPy can evaluate quickly.
    # (Make sure 'x' is the global symbolic variable defined above)
    f_numeric = sp.lambdify(var, function, "numpy") #lambdifies 'function' argument.
    y_vals = f_numeric(x_vals) #gets the y values for every x value in the discretized range of x axis
    
    # 3. Sets visualization limits to avoid moving screen
    initial_xlim = (a - margin, b + margin)
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    # Sets a wider range in Y axis for better visualization
    range_y = y_max - y_min
    initial_ylim = (y_min - 0.2 * range_y, y_max + 0.2 * range_y)
    
    ax.set_xlim(*initial_xlim)
    ax.set_ylim(*initial_ylim)
    
    total_iters = len(_x_history)
    zoom_threshold = max(2, int(total_iters * 0.3)) # Waits 30% of iterations before zooming
    
    min_width = (initial_xlim[1] - initial_xlim[0]) * 0.15 # Maximum zoom limit (15% of original width)
    min_height = (initial_ylim[1] - initial_ylim[0]) * 0.15
    
    # 4. Titles, labels and grid
    ax.plot(x_vals, y_vals, label="f(x)", color="blue", linewidth=2)
    ax.axhline(0, color="black", linewidth=1.5)
    ax.set_title("Newton-Raphson Method", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    
    curve_dot, = ax.plot([], [], 'ro', markersize=8) 
    tangent_line, = ax.plot([], [], 'g-', linewidth=1.5) 
    root_dot, = ax.plot([], [], 'gx', markersize=8) 

    def update(i):
       
        actual_x = _x_history[i]
        actual_y = f_numeric(actual_x)
        curve_dot.set_data([actual_x], [actual_y])
        
        # --- DYNAMIC ZOOM LOGIC ---
        if i < zoom_threshold:
            ax.set_xlim(*initial_xlim)
            ax.set_ylim(*initial_ylim)
        else:
            x_final = _x_history[-1]
            dist_x = abs(actual_x - x_final)
            dist_y = abs(actual_y - 0)
            margin_x = max(dist_x * 0.5, min_width / 2)
            margin_y = max(dist_y * 0.5, min_height / 2)
            min_x, max_x = min(actual_x, x_final), max(actual_x, x_final)
            min_y, max_y = min(actual_y, 0), max(actual_y, 0)
            ax.set_xlim(min_x - margin_x, max_x + margin_x)
            ax.set_ylim(min_y - margin_y, max_y + margin_y)
        # -------------------------------
        
        if i < len(_x_history) - 1:
            x_next = _x_history[i + 1]
            
            # Extends the tangent line across the entire current view
            xlims = ax.get_xlim()
            slope = fprime_def(actual_x)
            y_start = slope * (xlims[0] - actual_x) + actual_y
            y_end = slope * (xlims[1] - actual_x) + actual_y
            
            tangent_line.set_data([xlims[0], xlims[1]], [y_start, y_end])
            
            # The cross should be exactly where the tangent line crosses y=0
            x_cross = actual_x - (actual_y / slope) if slope != 0 else x_next
            root_dot.set_data([x_cross], [0])
        else:
           
            tangent_line.set_data([], [])
            root_dot.set_data([], [])
            
        return curve_dot, tangent_line, root_dot

    ani = FuncAnimation(fig, update, frames=len(_x_history), interval=400, repeat=False)
    plt.show()


    return real_x, real_k, real_error, real_history


    
    
    