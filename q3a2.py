# Steinmetz solid is the intersection of perpendicular cylinders
#Using a,m from paert (ii) of Q.2
#Radius of the cylinders is 1 unit
from MNL import *
N = 1000
m = 16381
a = 572
#------------Random number array
mlcg = np.array(mu_li_co_ge(1,a,m,N))/m
#-------Monte Carlo
def _using_MC(s,f,N,a,m):
    val = 0
    x = np.array(mu_li_co_ge(s, a, m, N))/m
    for i in range(N):
        val = val + f(x[i])
    ans = val/N
    return ans
#-------Enclosed volume
def vol_stez(x):
    return 4*(1 - x**2)

#------Calculated volume
vol = _using_MC(1,vol_stez,N,a,m)
print("Volume of the Steinmetz solid is : ",2*vol)

##############################################
#Volume of the Steinmetz solid is :  5.366198272359293