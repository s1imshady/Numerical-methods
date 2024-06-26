# Метод наискорейшего спуска
# The steepest descent method
import random
import numpy as np
import time


def step_desc_met3(A3, b3, eps3, x03):
    """
    Calculates the vector 'x' of the system of the form 'A * x = b'
    by the steepest descent method.
    :param A3: matrix of dimension n * n
    :param b3: vector of dimension n
    :param x03: first approximation, vector of zeros of dimension n
    :param eps3: required accuracy
    :return: x - calculated vector;
             delta_r - accuracy of the method
             delta_time - time spent on calculation
    """
    start_time = time.time()

    x_prev = x03
    r0 = A3 @ x_prev - b3
    t = (r0 @ r0) / ((A3 @ r0) @ r0)
    x = x_prev - t * r0

    x_prev = x
    r = A3 @ x_prev - b3
    t = (r @ r) / ((A3 @ r) @ r)
    x = x_prev - t * r
    delta_r = np.linalg.norm(r) / np.linalg.norm(r0)

    while delta_r >= eps3:
        x_prev = x
        r = A3 @ x_prev - b3
        t = (r @ r) / ((A3 @ r) @ r)
        x = x_prev - t * r
        delta_r = np.linalg.norm(r) / np.linalg.norm(r0)

    end_time = time.time()
    delta_time = end_time - start_time
    return x, delta_time, delta_r


def random_system(dim):
    """
    Sets the random values for the vector 'b' with values of the
    elements in range[0, 100] and makes the symmetric positive
    definite matrix 'A' with values of the elements in range[0, 64].
    :param dim: dimension of 'A' and 'b'
    :return: _A - symmetric positive definite matrix A
             _b - vector b
    """
    _A = np.zeros((dim, dim))
    _b = np.zeros(dim)
    for i in range(dim):
        _b[i] = int(random.random() * 10000) / 100
        for j in range(dim):
            _A[i, j] = int(random.random() * 8000) / 1000
    _A = _A @ _A.T
    return _A, _b
