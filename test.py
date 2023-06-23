import Znumbers
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds


a1 = np.array([0,0,0.1,0.2])
a2 = np.array([0.1,0.2,0.2,0.3])
# a3 = np.array([ ])

b1 = np.array([0.7,0.8,0.9])
b2 = np.array([0.5,0.6,0.7])

z1 = Znumbers.Znumbers(a1,b1)
z2 = Znumbers.Znumbers(a2,b2)

#print(Znumbers.sv(z1,z2))



def objective(x):
    return (x[0]*np.log(x[0])+x[1]*np.log(x[1])+x[2]*np.log(x[2])+x[3]*np.log(x[3])+x[4]*np.log(x[4]))
Aa = np.ones(shape =(2,5))

for i in range(3):
    Aa[1][i]= i/2
    Aa[1][4-i]=i/2
l_constarints = LinearConstraint(A=Aa, lb= [1, 0.7],ub= [1, 0.7])
lim = [0.25, 0.30, 0.75, 0.8]

bounds = Bounds([0, 0, 0, 0, 0],[1, 1, 1, 1, 1])  

x0 = [0.2, 0.2, 0.2, 0.2, 0.2]

solution = minimize(objective, x0, constraints=l_constarints,bounds=bounds)
ans =np.array(solution.x)
print(solution)

