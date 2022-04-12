import math
def f(x):
    return math.sqrt(1 - x ** 2)

#Pi using throws
def piwthro(a, m, N):
    X = LCG(a, 0, m, 0.3, N)
    Y = LCG(a, 0, m, 0.5, N)

    Points_in_Circle = 0
    #Pi using Monte Carlo
    piwMC = 0
    for x in X:
        piwMC += f(x)
        for y in Y:
            if math.sqrt(x ** 2 + y ** 2) < 0.5:
                Points_in_Circle += 1

    piwMC = 4 * piwMC / N
    pi_by_throws = 4 * Points_in_Circle / (N ** 2)
    return piwthro, piwMC


a1, b1 = piwthro(65, 1021, 1000)
a2, b2 = piwthro(572, 16381, 1000)

print(a1,b1,a2,b2)