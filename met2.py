import numpy as np
import time

#Генерация рандомной матрицы и вектора
def matrix_and_vector_generation(n):
    A = np.random.rand(n, n) * 10  
    #Симметричность
    A = (A + A.T) / 2
    #Положительно определённая
    A += n * np.eye(n)
    
    b = np.random.rand(n) * 10
    
    return [A, b]

#Вывод матрицы на экран
def print_matrix(matrix):
    for line in matrix:
        print(' '.join(map(str, line)))

#Вывод вектора на экран
def print_vector(vector):
    for x in vector:
        print(x, ' ')    

#Метод минимальных невязок, возвращает вектор решения, время решения и погрешность
def method2(A, b, eps, x0 = None):    
    start_time = time.time()    
    
    if x0 is None:
        x = np.zeros(len(b))
    else:
        x = x0        
    
    r0 = np.dot(A, x) - b
    r = r0    

    while np.linalg.norm(r) / np.linalg.norm(r0) > eps:      
        Ar = np.dot(A, r)
        tau = np.dot(r, Ar) / np.dot(Ar, Ar)        
        x = x - tau * r         
        r = np.dot(A, x) - b

    error = np.linalg.norm(r) / np.linalg.norm(r0)
    
    end_time = time.time()
    solution_time = end_time - start_time
    
    return x, solution_time, error