"""
@copyright: 2020--2030
@author: Haoxi Zhang

Testing the iLQR applied to the robot leg.

"""

import numpy as np
from iLQRSolver import *
from CostFunctionOneLeg import *
from oneLegDynamicModel import *
import matplotlib.pyplot as pl
from random import uniform
import time

Xinit = np.array([[1.57], [1.57], [0.0], [0.0]])
Xdes = np.array([[2.12],[2.0]])

FinalX = np.array([[-9999.0], [-99999.0], [-99999.0], [-999999.0]])

# number of actions in the control sequnce
# u = [a1, a2, ... aN]
N = 350

"""Debug"""
traj = False
if (traj):
    M = 5
else:
    M = 1
trajList = list([Xdes,Xdes,Xdes,Xdes,Xdes])

dt = 0.01

model = oneLegDynamicModel(dt)
costFunction = CostFunctionOneLeg()

initTime = time.time()
solver = iLQRSolver(model, dt, costFunction)

for i in range(M):
    Xdes = trajList[i]
    XList, UList = solver.trajectoryOptimizer(Xinit, Xdes, N, 200)
    endTime = time.time() - initTime
    FinalX = XList[-1]

print ("------- new ----------")
"""print ("xlist:")
print (XList)
print ("ulist:")
print (UList)"""

print ('time:')
print (endTime)
print ("\n --------")
print ("---- X init ----")
print (Xinit)
print ("---- X destination ----")
print (Xdes)
print ("---- X final ----")
print (FinalX)
print ("---- U final ----")
print (UList[-1])