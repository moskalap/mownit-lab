
def gauss_jordan(A,b):
    n = len(A)
    #make extended matrix
    for i in range(0,n):
        A[i].append(b[i])


    for i in range(0, n):

        pivot, pivot_row = abs(A[i][i]), i
        for k in range(i+1, n):
            if abs(A[k][i]) > pivot:
                pivot,pivot_row = abs(A[k][i]),k

        #swap
        for k in range(i, n+1):
            tmp = A[pivot_row][k]
            A[pivot_row][k] = A[i][k]
            A[i][k] = tmp

        #elimnation
        for k in range(i+1, n):
            elimination_ratio = -A[k][i]/A[i][i]
            for j in range(i, n+1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += elimination_ratio * A[i][j]

    #back substition
    x = [0 for i in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = A[i][n]/A[i][i]
        for k in range(i-1, -1, -1):
            A[k][n] -= A[k][i] * x[i]
    return x


if __name__ == "__main__":

    A = [[2,1,3],
         [2,6,8],
         [6,8,18]]

    b=[1,3,5]

    print(gauss_jordan(A,b))