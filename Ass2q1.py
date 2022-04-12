from MNL import read_csv, lu_decomp,chebyshev,fitw_cheby
import numpy as np
import matplotlib.pyplot as plt
# Defining function for polysquare fit
def polysq_fit(x, y, pow):
    n = len(x)
    para = pow + 1
    M = np.zeros((para,para))
    a = np.zeros(para)
    for i in range(para):
         for j in range(para):
            total = 0
            for k in range(n):
                total += x[k] ** (i + j)

            M[i, j] = total

    for i in range(para):
        total = 0
        for k in range(n):
            total += x[k] ** i * y[k]

        a[i] = total

    coef = lu_decomp(M, a)
    return coef, M

# Data is read here :
file = read_csv("assign2fit.txt")
u = [sub[0] for sub in file]
v = [i.split('\t',1) for i in u]
x = []
y = []
for i in range(len(v)):
    for j in range(1):
        x.append(v[i][0])
        y.append(v[i][1])

val_x = list(map(float,x))
val_y = list(map(float,y))
coeff, A1 = polysq_fit(val_x, val_y, 3)
print('Coefficients in original basis are given as: ', coeff)

# Condition number of matrix = 4.79553
cpara, A2 = fitw_cheby(val_x, val_y, 3)

c0, c1, c2, c3 = cpara[0], cpara[1], cpara[2], cpara[3]
print("Coefficients in modified Chebyshev basis: ",cpara)

x = np.linspace(0, 1, 100)
y = coeff[0] + coeff[1] * x + coeff[2] * x**2 + coeff[3] * x**3
plt.xlabel("$x$")
plt.ylabel("$y = f(x)$")
plt.title("Cubic least square fit")
plt.scatter(val_x, val_y, s=5, label="Data")
plt.plot(x, y, "g", label="Line fit")
plt.legend()
plt.show()
