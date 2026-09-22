import numpy as np



#VECTOR TO MATRIX FUNCTION.
def vector_to_matrix(data,rows,columns): #Receives an array and converts it into a matrix of n rows and m columns ("filas" and "columnas")
    vector=np.array(data,dtype=float)  #Defines a numpy array called "vector" and makes it a float array.
    if len(vector)!=rows*columns:  #Verifies the quantity of values inserted in the vector correlates with the dimensions of the matrix indicated
        raise ValueError(f"Error: The number of elements in the vector ({len(vector)}) does not match the order of the matrix ({rows*columns}).") #if doesnt match, it sends an error.
    else:
        matrix=vector.reshape((rows,columns)) #Gives the array the shape in correspondence with the matrix´s dimension.
        return matrix #Returns the matrix reshaped.




#MATRIX VALIDATION FUNCTION.
def matrix_validation(matrix):
    matrix=np.asarray(matrix,dtype=float)
    if matrix.ndim>2 or matrix.ndim==0 or len(matrix)==0:
        raise ValueError(f"Error: invalid matrix")
    else:
        return matrix




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




#SPECTRAL_RADIUS_Function.
def spectral_radius(matrix):
    matrix=sqr_matrix_validation(matrix)
    eigenvalues=np.linalg.eigvals(matrix)
    return np.max(np.abs(eigenvalues))




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

