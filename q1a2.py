from MNL import read_csv, lu_decomp,chebyshev,fitw_cheby
import matplotlib.pyplot as plt
import numpy as np

#-----Polysquare fit
def sq_fit_poly(x, y, pow):
    n = len(x)
    u = n + 1
    M = np.zeros((u,u))
    a = np.zeros(u)
    for i in range(u):
         for j in range(u):
            total = 0
            for k in range(n):
                total += x[k] ** (i + j)
            M[i, j] = total
    for i in range(u):
        total = 0
        for k in range(n):
            total += x[k] ** i * y[k]
        a[i] = total
    coef = lu_decomp(M, a)
    return coef, M

#----Data reading
file = read_csv("assign2fit.txt")
u = [sub[0] for sub in file]
v = [i.split('\t',1) for i in u]
x = []
y = []
for i in range(len(v)):
    for j in range(1):
        x.append(v[i][0])
        y.append(v[i][1])


x = list(map(float,x))
y = list(map(float,y))
coeff, A1 = sq_fit_poly(x,y, 3)
print('Coefficients in original basis are given as: ', coeff)

#-------chebyshev coefficients
cpara, A2 = fitw_cheby(x,y, 3)

c0, c1, c2, c3 = cpara[0], cpara[1], cpara[2], cpara[3]
print("Coefficients in modified Chebyshev basis: ",cpara)


####################################################
#Coefficients in original basis are given as:  [0.5746586674195995, 4.725861442142078, -11.128217777643616, 7.6686776229096685]
#Coefficients in modified Chebyshev basis:  [1.1609694790335525, 0.39351446798815237, 0.04684983209010658, 0.23964617571596986]



######
#As a rule of thumb, if the condition number $\kappa (A)=10^{k}$ , then you may lose up to k digits of accuracy.
#Here from our analysis we can see that k ~ 4 for original chebyshev and k ~ 1 for modified basis.
#Modified chebyshev basis are better choice.