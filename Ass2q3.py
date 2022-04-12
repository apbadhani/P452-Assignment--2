# Assuming unit radius for the two cylinders
from MNL import *

# Function to calculate the common enclosed volume
def steimz(x):
    return 4 * (1 - x**2)

def MC(func, N, lims):
    # Generate list of N random points between limits
    xrand = mu_li_co_ge(234.34, 65, 1, N)

    summation = 0
    for i in range(N):
        summation += func(xrand[i])

    total = 1/float(N) * summation

    return total

vol = MC(steimz, 10000, (0,1))
print("Volume of the Steinmetz solid is : {} ".format(2*vol))