def chisquar_fit(x, y, pow):
    n = len(x)
    M = [[0 for i in range(pow + 1)] for j in range(pow + 1)]
    a = [[0] for i in range(pow + 1)]
    for i in range(n):
        for j in range(degree + 1):
            a[j][0] += (x[i] ** j) * y[i]
            for k in range(degree + 1):
                A[j][k] += (x[i] ** (j + k))
    A1 = deepcopy(M)
    coeff = Solve_LE_LU_Decomposition(M, b)
    return coeff, varmat, A1

#data is read here
X, Y, std = read_data('assign2fit.txt')
coeff A1 = chisquar_fit(X, Y, std, 3)
print('Coefficients are given as: ', coeff)

#defining phi(x)
def phi(x):
    return [1, 2 * x - 1, 8 * (x ** 2) - 8 * x + 1, 32 * (x ** 3) - 48 * (x ** 2) + 18 * x - 1]


def fitwbasis(x, y, phi):
    pow = len(phi(0)) - 1
    n = len(x)
    M = [[0 for i in range(pow + 1)] for j in range(pow + 1)]
    b = [[0] for i in range(pow + 1)]
    for i in range(n):
        for j in range(pow + 1):
            b[j][0] += (phi(x[i])[j]) * y[i]
            for k in range(degree + 1):
                M[j][k] += (phi(x[i])[j]) * (phi(x[i])[k])
    A1 = deepcopy(A)
    coeff = Solve_LE_LU_Decomposition(A, b)
    A = deepcopy(A1)
    return coeff, A
