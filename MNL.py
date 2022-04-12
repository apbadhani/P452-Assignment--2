import numpy as np
import sys
import math
import copy
#Creating a zero matrix of order m*n
def zeromatrix(m,n):
        p= [[0 for i in range(n)] for j in range(m)]
        return(p)
#Calculates the norm of a vector
def norm(x):
    total = 0
    for i in range(len(x)):
        total += x[i]**2

    return total**(1/2)

# Function for printing the matrix
def mat_print(a):
    for i in range(len(a)):
        print(a[i])

# Fucnction for vector subtraction
def vec_sub(a, b):
    if (len(a) != len(b)):
        exit()
    else:
        return [x1 - x2 for (x1, x2) in zip(a, b)]

# Function for reading a csv file
def read_csv(path):
    with open(path, 'r+') as file:
        results = []

        for line in file:
            line = line.rstrip('\n') # remove `\n` at the end of line
            items = line.split(',')
            results.append(list(items))

        # after for-loop
        return results
#Forward backward substitution
def forward_backward(U: list, L: list, b: list) -> list:
    y = [0 for i in range(len(b))]

    for i in range(len(b)):
        total = 0
        for j in range(i):
            total += L[i][j] * y[j]
        y[i] = b[i] - total

    x = [0 for i in range(len(b))]

    for i in reversed(range(len(b))):
        total = 0
        for j in range(i+1, len(b)):
            total += U[i][j] * x[j]
        x[i] = (y[i] - total)/U[i][i]

    return x
# Matrix multiplication
def matmul(a, b):
    product = [[sum(i*j for i,j in zip(a_row, b_col)) for b_col in zip(*b)] for
            a_row in a]

    return product
#function for matrix vector multiplication
def mat_vec_mult(A, B):
    n = len(B)
    if len(A[0]) == n:
        p = [0 for i in range(n)]
        for i in range(n):
            for j in range(n):
                p[i] = p[i] + (A[i][j] * B[j])
        return (p)
    else:
        print('This combination is not suitable for multiplication')


#Gauss-Jordan
def gau_jor(A: list, b: list) -> list:
    def partial_pivot(A: list, b: list):
        n = len(A)
        for i in range(n-1):
            if abs(A[i][i]) < 1e-10:
                for j in range(i+1,n):
                    if abs(A[j][i]) > abs(A[i][i]):
                        A[j], A[i] = A[i], A[j]  # interchange A[i] and A[j]
                        b[j], b[i] = b[i], b[j]  # interchange b[i] and b[j]

    n = len(A)
    partial_pivot(A, b)
    for i in range(n):
        pivot = A[i][i]
        b[i] = b[i] / pivot
        for c in range(i, n):
            A[i][c] = A[i][c] / pivot

        for k in range(n):
            if k != i and A[k][i] != 0:
                factor = A[k][i]
                b[k] = b[k] - factor*b[i]
                for j in range(i, n):
                    A[k][j] = A[k][j] - factor*A[i][j]

    x = b
    return x

#LU Decomposition
def lu_decomp(A: list, b: list) -> list:
    def partial_pivot(A: list, b: list):
        count = 0
        n = len(A)
        for i in range(n-1):
            if abs(A[i][i]) < 1e-10:
                for j in range(i+1,n):
                    if abs(A[j][i]) > abs(A[i][i]):
                        A[j], A[i] = A[i], A[j]  # interchange ith and jth rows of matrix 'A'
                        count += 1
                        b[j], b[i] = b[i], b[j]  # interchange ith and jth elements of vector 'b'

        return A, b,count
    def crout(A: list):
        U = [[0 for i in range(len(A))] for j in range(len(A))]
        L = [[0 for i in range(len(A))] for j in range(len(A))]

        for i in range(len(A)):
            L[i][i] = 1

        for j in range(len(A)):
            for i in range(len(A)):
                total = 0
                for k in range(i):
                    total += L[i][k] * U[k][j]

                if i == j:
                    U[i][j] = A[i][j] - total

                elif i > j:
                    L[i][j] = (A[i][j] - total)/U[j][j]

                else :
                    U[i][j] = A[i][j] - total

        return U, L

    partial_pivot(A, b)
    U, L = crout(A)
    x = forward_backward(U, L, b)
    return x

# Jacobi Method
def jacobi(A: list, b: list, tol: float) -> list:
    n = len(A)
    x = [1 for i in range(n)]     # define a dummy vector for storing solution vector
    xold = [0 for i in range(n)]
    iterations = []; residue = [];
    count = 0
    while norm(vec_sub(xold, x)) > tol:
        iterations.append(count)
        count += 1
        residue.append(norm(vec_sub(xold, x)))
        xold = x.copy()
        for i in range(n):
            total = 0
            for j in range(n):
                if i != j:
                    total += A[i][j] * x[j]

            x[i] = 1/A[i][i] * (b[i] - total)

    return x, iterations,residue

#Gauss Seidel
def gauss_seidel(A: list, b: list, tol: float) -> list:
    n = len(A)
    x = [0 for i in range(n)]
    xold = [1 for i in range(n)]
    iterations = []; residue = [];
    count = 0

    while norm(vec_sub(x, xold)) > tol:
        xold = x.copy()
        iterations.append(count)
        count += 1
        for i in range(n):
            d = b[i]
            for j in range(n):
                if j != i:
                    d -= A[i][j] * x[j]

            x[i] = d / A[i][i]

        residue.append(norm(vec_sub(x, xold)))

    return x, iterations,residue
#Conjugate Gradient
def conjgrad(A: list, b: list, tol: float) -> list:
    n = len(b)
    x = [1 for i in range(n)]
    r = vec_sub(b, vecmul(A, x))
    d = r.copy()
    rprevdot = dotprod(r, r)
    iterations = []; residue = [];
    count = 0       # counts the number of iterations

    # convergence in n steps
    for i in range(n):
        iterations.append(count)
        Ad = vecmul(A, d)
        alpha = rprevdot / dotprod(d, Ad)
        for j in range(n):
            x[j] += alpha*d[j]
            r[j] -= alpha*Ad[j]
        rnextdot = dotprod(r, r)
        residue.append(sqrt(rnextdot))
        count += 1

        if sqrt(rnextdot) < tol:
            return x, iterations, residue

        else:
            beta = rnextdot / rprevdot
            for j in range(n):
                d[j] = r[j] + beta*d[j]
            rprevdot = rnextdot
#Givans method
def crossprod(A,B):
    if len(A[0]) == len(B):
        crossprod = [[0 for i in range(len(B[0]))]for j in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for m in range(len(A)):
                    crossprod[i][j] = crossprod[i][j] + A[i][m]*B[m][j]
        return crossprod
    else:
        print("Matrices cannot be multiplied")
# crossprod is used in the function gaussgivan
def maxoff(A):
    maxtemp = A[0][1]
    k = 0
    l = 1
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            if abs(A[i][j]) > abs(maxtemp):
                maxtemp = A[i][j]
                k = i
                l = j
    return maxtemp, k, l


def gaussgivan(A, ep):
    max, i, j = maxoff(A)
    while abs(max) >= ep:
        #calculating theta
        if A[i][i] - A[j][j] == 0:
            theta = math.pi / 4
        else:
            theta = math.atan((2 * A[i][j]) / (A[i][i] - A[j][j])) / 2
        #Identity matrix
        P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        #Making P matix tridiagonal
        P[i][i] = P[j][j] = math.cos(theta)
        P[i][j] = -1 * math.sin(theta)
        P[j][i] = math.sin(theta)
        AP = crossprod(A, P)
        #making P an array so to use transpose function
        P = np.array(P)
        #Transpose of P
        PT = P.T.tolist()
        #getting back the matrix in tridiagonal form
        A = crossprod(PT, AP)
        #checking the offset in the matrix obtained
        max, i, j = maxoff(A)
    return A

#frobenius norm
def frob_norm(A):
    sum = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            sum = sum + (A[i][j] ** 2)
    return math.sqrt(sum)
#gives the norm of A
def pow_norm(A):
    max = 0
    for i in range(len(A)):
        if max <= A[i][0]:
            max = A[i][0]
    normA = scaler_matrix_division(max, A)
    return normA

# Power method
def pow_method(A, x0=[[1], [1], [1]], eps=1.0e-4):
    i = 0
    lam0 = 1
    lam1 = 0
    while abs(lam1 - lam0) >= eps:
        # print("error=",abs(lam1-lam0))
        if i != 0:
            lam0 = lam1

        Ax0 = mat_mult(A, x0)
        AAx0 = mat_mult(A, Ax0)
        # print("Ax0=",Ax0)
        # print("AAx0=",AAx0)
        dotU = inner_product(AAx0, Ax0)
        dotL = inner_product(Ax0, Ax0)
        # print("U=",dotU)
        # print("L=",dotL)
        lam1 = dotU / dotL

        x0 = Ax0
        i = i + 1
        # print("i=",i)

        # print("eigenvalue=",lam1)
        ev = pow_norm(x0)
        # print ("eigenvector=",ev)
    return lam1, ev  # returns lam1=largest eigen value and ev = coressponding eigen vec
#gives mean
def Mean(A):
    n = len(A)
    sum = 0
    mean = 0
    for i in range(n):
        sum = sum + A[i]
    return sum/n
#gives variance
def Variance(A):
    n = len(A)
    mean = Mean(A)
    sum = 0
    for i in range(n):
        sum = sum + (A[i]-mean)**2
    return sum/n
#solves equation
def solveeqn(m, qw):
    m = Invert(m)

    X = []
    X.append(m[0][0]*qw[0] + m[0][1]*qw[1])
    X.append(m[1][0]*qw[0] + m[1][1]*qw[1])
    return(X)
def sum1(X, n):
    n = n + 1
    suMatrix = []
    j = 0
    while j<2*n:
        sum = 0
        i = 0
        while i< len(X):
            sum = sum + (X[i])**j
            i = i + 1
        suMatrix.append(sum)
        j = j+1
    return suMatrix
#makes a new matrix
def makemat(suMatrix, n):
    n = n + 1
    m = [[0 for i in range(n)]for j in range(n)]
    i = 0
    while i<n:
        j = 0
        while j<n:
            m[i][j] = suMatrix[j+i]
            j = j+1
        i = i + 1
    return m

def sum2(X, Y, n):
    n = n+1
    suMatrix = []
    j = 0
    while j<n:
        sum = 0
        i = 0
        while i< len(X):
            sum = sum + ((X[i])**j)*Y[i]
            i = i + 1
        suMatrix.append(sum)
        j = j+1
    return suMatrix

#chi square fit function
def fit(X,Y):
    k = sum1(X, 1)         #taking all the sigma_x
    m = makemat(k, 1)      #sigma_x**i matrix

    qw = sum2(X, Y, 1)     #sigma_x**i*y matrix

    X = solveeqn(m, qw)
    return X[0],X[1]

# Bootstrap method
def bootstrap(A,b):
    mean = []
    vari = []
    for i in range(b):
        #making bootstrap dataset
        resample = random.choices(A,k=len(A))
        #calculating mean of the resampled data
        m = Mean(resample)
        mean.append(m)
        var = Variance(resample)
        vari.append(var)
    #to get confidence levels we calculate Standard deviation of this distribution
    x = (Mean(mean))
    y = (Mean(var))
    #plotting the mean values as a histogram
    plt.hist(mean)
    return x,y

#jackknife method
def jkknife(A):
    n = len(A)
    yi = []
    for i in range(n):
        B = A.copy()
        del(B[i])
        #calculating mean excluding one element
        mean = Mean(B)
        #MAking a new y vector, stores all means
        yi.append(mean)
    #mean of the new formed set
    yibar = Mean(yi)
    sum = 0
    for i in range(n):
        sum = sum + (yi[i] - yibar)**2
    #calculating error
    err = ((n-1)/n)*sum
    return yibar,err
#Defining Chebyshev function
def chebyshev(x: float, order: int) -> float:
    if order == 0:
        return 1
    elif order == 1:
        return 2*x - 1
    elif order == 2:
        return 8*x**2 - 8*x + 1
    elif order == 3:
        return 32*x**3 - 48*x**2 + 18*x - 1

#Defining the function for chebyshev fit
def fitw_cheby(xvals: np.array, yvals: np.array, degree: int):
    n = len(xvals)
    para = degree + 1
    A = np.zeros((para, para))
    b = np.zeros(para)

    for i in range(para):
        for j in range(para):
            total = 0
            for k in range(n):
                total += chebyshev(xvals[k], j) * chebyshev(xvals[k], i)

            A[i, j] = total

    for i in range(para):
        total = 0
        for k in range(n):
            total += chebyshev(xvals[k], i) * yvals[k]

        b[i] = total

    para = lu_decomp(A, b)
    return para,A
# Function for Pseudo random number generator
def mu_li_co_ge(seed: float, a: float, m: float, num: int) -> list:
    x = seed
    rands = []
    for i in range(num):
        x = (a*x) % m
        rands.append(x)

    return rands