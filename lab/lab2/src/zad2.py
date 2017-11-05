import numpy as np
def lu(a):
    n = len(a)

    L = np.zeros((n, n))
    for i in range(0, n):
        L[i, i] = 1

    U = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            U[i, j] = a[i, j]

    for i in range(0, n):
        max_index = i
        pivot = a[i, i]
        for k in range(i,n):
            if abs(a[k, i]) > abs(pivot) :
                max_index, pivot = k, a[k, i]

        #swap
        if max_index != i:
            a[i, max_index] = a[max_index, i]

        #elimination
        for k in range(i + 1, n):
            elimination_ratio = U[k, i] / (U[i, i])
            L[k, i] = elimination_ratio
            for j in range(i, n):
                U[k, j] -= elimination_ratio * U[i, j]

        for k in range(i + 1, n):
            U[k, i] = 0

    return L, U
