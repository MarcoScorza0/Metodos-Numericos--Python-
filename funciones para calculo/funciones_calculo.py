import numpy as np



#VECTOR TO MATRIX FUNCTION.
def vector_to_matrix(data,rows,columns): #Receives an array and converts it into a matrix of n rows and m columns ("filas" and "columnas")
    vector=np.array(data,dtype=float)  #Defines a numpy array called "vector" and makes it a float array.
    if len(vector)!=rows*columns:  #Verifies the quantity of values inserted in the vector correlates with the dimensions of the matrix indicated
        raise ValueError(f"Error: The number of elements in the vector ({len(vector)}) does not match the order of the matrix ({rows*columns}).") #if doesnt match, it sends an error.
    else:
        matrix=vector.reshape((rows,columns)) #Gives the array the shape in correspondence with the matrix´s dimension.
        return matrix #Returns the matrix reshaped.




#JACOBI's method function
def jacobi(principal_matrix,independent_terms_matrix,initial_vector,max_iterations,tolerance):
    if np.any(np.diag(principal_matrix)==0):
        raise ValueError(f"Jacobi's method can't be used. The matrix D can't be inverted.")
    D=np.diag(np.diag(principal_matrix))
    L=-np.tril(principal_matrix)+D
    U=-np.triu(principal_matrix)+D
    D_inv=np.linalg.inv(D)
    
    rows_A, cols_A = principal_matrix.shape
    rows_b, cols_b = independent_terms_matrix.shape
    rows_x0, cols_x0 = initial_vector.shape
    if rows_A != cols_A or rows_b != rows_A or cols_b != 1 or rows_x0 != rows_A or cols_x0 != 1:
        raise ValueError(f"Incorrect dimensions: principal_matrix must be of dimensions ({rows_A}, {rows_A}), and vectors must be of dimensions ({rows_A}, 1).")





    TJ=D_inv @ (L+U)
    CJ=D_inv @ independent_terms_matrix
    matrix_k=initial_vector
    k=0
    infinite_norm=float('inf')
    while k<max_iterations and infinite_norm>tolerance:
        matrix_xnew=TJ @ matrix_k + CJ
        k=k+1
        infinite_norm=np.max(np.abs(matrix_xnew - matrix_k))
        matrix_k=matrix_xnew
        if np.isnan(infinite_norm) or np.isinf(infinite_norm):
            print(f"Jacobi's method has DIVERGED (values turned infinite/nan in iteration {k})")
            return None,k,infinite_norm

    

    if(k==max_iterations):
        print(f"Too many iterations. Jacobi's method did not converge. Maximum iterations have been reached ({k})")
        return None,k,infinite_norm
    print(f"Jacobi's method CONVERGED successfully: \nx=\n{matrix_xnew}\nITERATIONS={k}\nERROR IN NORM=\n\t{infinite_norm}")




    return matrix_xnew,k,infinite_norm

