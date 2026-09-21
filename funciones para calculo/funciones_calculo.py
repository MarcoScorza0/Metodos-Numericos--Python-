import numpy as np



#VECTOR A MATRIZ
def vector_a_matriz(datos,filas,columnas):
    vector=np.array(datos,dtype=float)
    if len(vector)!=filas*columnas:
        raise ValueError(f"Error: La cantidad de datos del arreglo ({len(vector)}) no coincide con el orden indicado de la matriz ({filas*columnas})")
    else:
        matriz=vector.reshape((filas,columnas))
        return matriz




#METODO DE JACOBI
def jacobi(matriz_principal,matriz_term_independientes,matriz_inicial,max_iteraciones,tolerancia):
    if np.any(np.diag(matriz_principal)==0):
        raise ValueError(f"El metodo de jacobi no puede ser utilizado. La matriz D no es invertible.")
    D=np.diag(np.diag(matriz_principal))
    L=-np.tril(matriz_principal)+D
    U=-np.triu(matriz_principal)+D
    D_inv=np.linalg.inv(D)
    
    filas_A, cols_A = matriz_principal.shape
    filas_b, cols_b = matriz_term_independientes.shape
    filas_x0, cols_x0 = matriz_inicial.shape
    if filas_A != cols_A or filas_b != filas_A or cols_b != 1 or filas_x0 != filas_A or cols_x0 != 1:
        raise ValueError(f"Dimensiones incorrectas: matriz_principal debe ser ({filas_A}, {filas_A}), y los vectores ({filas_A}, 1).")





    TJ=D_inv @ (L+U)
    CJ=D_inv @ matriz_term_independientes
    matrizk=matriz_inicial
    k=0
    norma_infinito=float('inf')
    while k<max_iteraciones and norma_infinito>tolerancia:
        matriz_xnew=TJ @ matrizk + CJ
        k=k+1
        norma_infinito=np.max(np.abs(matriz_xnew - matrizk))
        matrizk=matriz_xnew
        if np.isnan(norma_infinito) or np.isinf(norma_infinito):
            print(f"El método de Jacobi DIVERGIÓ (los valores se fueron a infinito/nan en la iteración {k})")
            return None,k,norma_infinito

    

    if(k==max_iteraciones):
        print(f"El metodo de jacobi no convergio. Se alcanzo el maximo de iteraciones ({k})")
        return None,k,norma_infinito
    print(f"El metodo convergió: \nx=\n{matriz_xnew}\nITERACIONES={k}\nERROR EN NORMA=\n\t{norma_infinito}")




    return matriz_xnew,k,norma_infinito

