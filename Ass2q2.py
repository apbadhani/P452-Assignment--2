import math
import numpy as np
from MNL import mu_li_co_ge

def f(x):
    return math.sqrt(1 - x ** 2)

def circle(x,y):
    return x**2 + y**2 - 1

#MC Integration
def pi_using_MC(f, seed, a, m, N):
    # Generate list of N random points between lims
    rand_x = mu_li_co_ge(seed, a, m, N)
    rand_x = np.array(rand_x) / m

    sum = 0
    for i in range(N):
        sum += f(rand_x[i])

    tot = 1/float(N) * sum

    return tot
#Pi using throws
def pi_w_thro(seed1,seed2,a, m, N):
    X = mu_li_co_ge(seed1,a,m,N)
    Y = mu_li_co_ge(seed2,a,m,N)

    rand_x = np.array(X)/m
    rand_y = np.array(Y)/m

    h = 0
    for i in range(N):
        if circle(rand_x[i],rand_y[i])<=0:
            h+=1
    res = h/N
    return res
int_pi1 = pi_using_MC(f,5.12,65,1021,10000)
int_pi2 = pi_using_MC(f,5.12,572,16381,10000)
pi_hit1 = pi_w_thro(65.4,7.68,65, 1021, 10000)
pi_hit2 = pi_w_thro(65.4,7.68,572, 16381, 10000)
print('pi value using MC integration for case a) : ',4*int_pi1)
print('pi value using MC integration for case b) : ',4*int_pi2)
print('pi value by throwing points for case a) :',4*pi_hit1)
print('pi value by throwing points for case b) :',4*pi_hit2)