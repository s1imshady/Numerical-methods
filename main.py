
import numpy as np
from scipy.sparse import linalg
import time
import random

#проверка на пол-опр
def is_pol(a):
    i = 0
    res = True
    while res and i < len(a):
        i += 1
        res = np.linalg.det(a[:i,:i]) > 0
    return res

#генерация рандомной системы
def create_system(n):
    
    L = np.random.randn(n, n) 
    L = np.tril(L)  
    A = np.dot(L, L.T)
    
    b = np.random.randn(n) 
    B = np.zeros((n, n))
    if (n > 2):
        for i in range(n):
            B[i, i] = A[i, i]
        
        for i in range(n-1):
            for j in range(n-1):
                if i == j: 
                    B[i, j+1] = A[i, j+1]
        for j in range(n-1):
            for i in range(n-1):
                if i == j: 
                    B[i+1, j] = A[i+1, j]
    else:
        for i in range(n):
            B[i,i] = A[i, i]
    
    if (is_pol(A)):
        
        return A, B, b
    
    else:
        
        print("Матрица A не является положительно-определенной")

# метод минимальных поправок (работает не всех вход данных хз)
def mmс_meth(A, B, b, x0, eps = 1e-6):
    
    xk = x0
    xk1 = b  
    
    start_time = time.time()
    
    while ((np.linalg.norm(xk1-xk))  > eps):
        
        xk = xk1
        rk = np.dot(A, xk) - b
        invB = np.linalg.inv(B)
        wk = np.dot(rk, invB)
        Awk = np.dot(A, wk)
        vk = np.dot(invB, Awk)
        tau = (np.dot(wk, Awk))/ (np.dot(vk, Awk))
        xk1 = xk - np.dot(tau, wk)
        
    err = (np.linalg.norm(xk1-xk))
    end_time = time.time()
    total_time = end_time - start_time
    
    return xk, err, time

#для вывода
def solve():
    
    n = random.randint(3, 5)
    
    A, B, b = create_system(n)
    
    x0 = np.zeros(n)
    x_res, time_res, err = mmс_meth(A, B, b, x0, 1e-6)
    x = linalg.cg(A, b, tol = 1e-6)
    
    print("Точное решение:")
    print(x)
    print("--------------------------------")
    
    print("Размерность матрицы/вектора: ", n)
    print("--------------------------------")
    print("Матрица А: ")
    print(A)
    print("--------------------------------")
    print("Вектор b: ")
    print(b)
    print("--------------------------------")
    print("Решение: ")
    print(x_res)
    print("Погрешность: ", err)
#     print(f"Время выполнения: {time_res} секунд")

    return
             


#main

print("Введите число:")
k = int(input())
for i in range(k):
    print()
    print("---------------- Решение ",  i + 1, ": -----------------")
    print()
    solve()




