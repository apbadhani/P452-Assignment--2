import math
import numpy as np
from MNL import *

N = 10000
#--------Circle
def cir(x,y):
    return x**2 + y**2
#--------Integration function
def f(x):
    f = 4*(math.sqrt(1 - x ** 2))
    return f

#------Pi using MC Integration
def _using_MC(s,f,N,a,m):
    val = 0
    x = np.array(mu_li_co_ge(s, a, m, N))/m
    for i in range(N):
        val = val + f(x[i])
    ans = val/N
    return ans
#Seed Value = 1
print('pi value using Monte Carlo integration for case a) : ',_using_MC(1,f,N,65,1021))
print('pi value using Monte Carlo integration for case b) : ',_using_MC(1,f,N,572,16381))
#--------Pi using throws
def _using_thro(s1,s2,a,m,N):
    x = np.array(mu_li_co_ge(s1, a, m, N)) / m
    y = np.array(mu_li_co_ge(s2, a, m, N)) / m
    # It is basically a probability that points lie in the desired circles
    pro = 0
    res = 0
    for i in range(N):
        if (x[i]**2 + y[i]**2) <= 1:
            pro+=1
    res = (1/N) * pro
    return res
#Seed values for case a = 1,26
#Seed values for case b = 2,78
print('pi value by throwing points for case a) :',4*_using_thro(1,26,65, 1021, 10000))
print('pi value by throwing points for case b) :',4*_using_thro(2,78,572, 16381, 10000))

###################################################
#pi value using Monte Carlo integration for case a) :  3.142055466949416
#pi value using Monte Carlo integration for case b) :  3.141720546369245
#pi value by throwing points for case a) : 3.1416
#pi value by throwing points for case b) : 3.1488