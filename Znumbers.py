import math
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds

epsilon = 1e-15

class Znumbers :
    def __init__(self,A,B):
        self.A = A
        self.B = B
        if A.shape[0] != 4:
            print(f"Error A: {A.shape}")
        if B.shape[0] != 3:
            print(f"Error B: {B.shape}")

def h(z1,z2):
    return max(abs(z1.B[0]-z2.B[0]),abs(z1.B[1]-z2.B[1]),abs(z1.B[2]-z1.B[2]))

def slc(z1,z2):
    # slc =1-sl, where sl: similarity of reliability measures
    return abs(z1.B[0]+z1.B[2]-z2.B[0]-z2.B[2])/2
    #return 0


def zrsm(z1,z2):
    return 1-(h(z1,z2) + slc(z1,z2))/2

def objective(x):
    return (x[0]*np.log(x[0]+epsilon)+x[1]*np.log(x[1]+epsilon)+x[2]*np.log(x[2]+epsilon)+x[3]*np.log(x[3]+epsilon)+x[4]*np.log(x[4]+epsilon))

def memTrfn(A,a):
    #membership function for trapezoidal fuzzy number
    # returns the membership value at a
    if(A[0]<a<=A[1]): 
        return  (a-A[0])/(A[1]-A[0])
    elif(A[1]<a<=A[2]):
        return 1
    elif(A[2]<a<=A[3]):
        return (A[3]-a)/(A[3]-A[2])
    else:
        return 0
    
def memTfn(B,b):
    if(B[0]<b<=B[1]): 
        return  (b-B[0])/(B[1]-B[0])
    elif(B[1]<b<=B[2]):
        return (B[2]-b)/(B[2]-B[1])
    else:
        return 0

def pdf(A,b,n_parts):
    arr_A= np.ones(shape=(2,n_parts))
    for i in range(n_parts//2):
        arr_A[1][i]= i/(n_parts//2)
        arr_A[1][n_parts-i-1]=i/(n_parts//2)

    l_constraints = LinearConstraint(A=arr_A, lb= [1, b],ub= [1, b])
    #lim = [0.25, 0.30, 0.75, 0.8]

    bounds = Bounds([0, 0, 0, 0, 0],[1, 1, 1, 1, 1])  

    x0 = [0.2, 0.2, 0.2, 0.2, 0.2]

    solution = minimize(objective, x0, constraints=l_constraints,bounds=bounds)
    return (solution.x)
    



def sv(z1,z2,n_i=5,n_j=5):
    # gives the variational similarity between z1 and z2
    # n_i is the no.of probaility ditributions i.e, no.of parts B components are divided into 
    # n_j is the no.of parts A components are divided into
    ans =0
    
    for i in range(n_i):
        p1 = pdf(z1.A,z1.B[0]+ i*(z1.B[2]-z1.B[0])/(n_i-1),n_j)
        p2 = pdf(z2.A,z2.B[0]+ i*(z2.B[2]-z2.B[0])/(n_i-1),n_j)

        # print(i)
        # print(p1)
        # print(p2)
        
        p1 = p1-p2
        p1 = np.abs(p1)

        ans = max(ans,np.sum(p1))
        #print(ans)

    return 1-ans

def COG(z1,z2):
    if z1.A[0]==z1.A[3] :
        rhoA1 = 0.5
    else :
        rhoA1 = (z1.A[2] - z1.A[1])/(z1.A[3]-z1.A[0])
    piA1 = (rhoA1*(z1.A[2] + z1.A[1])+(1-rhoA1)*(z1.A[3]+z1.A[0]))/2
    
    if z2.A[0]==z2.A[3] :
        rhoA2 = 0.5
    else :
        rhoA2 = (z2.A[2] - z2.A[1])/(z2.A[3]-z2.A[0])
    piA2 = (rhoA2*(z2.A[2] + z2.A[1])+(1-rhoA2)*(z2.A[3]+z2.A[0]))/2

    return math.sqrt(((piA1-piA2)*(piA1-piA2)+(rhoA1-rhoA2)*(rhoA1-rhoA2))/1.25)

def area(z):
    return 0.5*(z.A[3]+z.A[2]-z.A[1]-z.A[0])

def perimeter(z):
    peri = z.A[3]+z.A[2]-z.A[1]-z.A[0]
    if(z.A[1]==z.A[0]):
        peri += 1
    else:
        peri+= math.sqrt(1+(z.A[1]-z.A[0])*(z.A[1]-z.A[0]))

    if(z.A[3]==z.A[2]):
        peri += 1
    else:
        peri+= math.sqrt(1+(z.A[3]-z.A[2])*(z.A[3]-z.A[2]))

    return peri
    

def sz(z1,z2):
    A1 = area(z1)
    A2 = area(z2)
    P1 = perimeter(z1)
    P2 = perimeter(z2)

    APD = 1- (abs(A1-A2)+abs(P1-P2)/max(P1,P2))/3

    return (1-(abs(z1.A[0]-z2.A[0])+abs(z1.A[1]-z2.A[1])+abs(z1.A[2]-z2.A[2])+abs(z1.A[3]-z2.A[3]))*COG(z1,z2)/4)*APD

def zpsm(z1,z2):
    #return 0.5(sz(z1,z2)+sv(z1,z2))
    return 0.8*sz(z1,z2)+0.2*sv(z1,z2)


def distanceZ(z1,z2,w=0.1):
    return (1-(w*zrsm(z1,z2)+ (1-w)*zpsm(z1,z2)))

