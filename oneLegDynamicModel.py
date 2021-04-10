"""
@Copyright (C) 2020--2030
@Author: Haoxi Zhang

dynamic model class for moving a simply leg

- usage:
    a single 2-link leg dynamics

- inputs:
    x = [q1, q2, q1_dot, q2_dot]    # State vector: joint angles (2); joint velocities (2)
    u = [F1, F2]                    # Control vector: torques for servo_1 & servo_2
    
- returns:
    derivatives
    nextState from currentState and chosen Action

"""
import numpy as np
from random import uniform
from CostFunctionOneLeg import *
import time

class oneLegDynamicModel:
    
    def __init__(self, dt):
        self.XList = []
        self.Xdim = 4     # dimention of state vector: 4 -- 2 joint angles + 2 joint velocities
        self.Udim = 2     # dimention of control vector: 2 -- 2-link arm [f1, f2]
        self.flgFix = 0          # clear large-fix global flag
        self.maxValue = 1E+10    # upper bound on states and costs (Inf: none)
        self.index = 0
        self.terminal = False
        self.dt = dt
        self.Xinit = np.array([1.57, 1.57, 0.0, 0.0])
        self.costFunction = CostFunctionOneLeg()

        # leg model parameters
        self.m1_ = 1.4     # segment mass
        self.m2_ = 1.1
        self.l1_ = 0.3    # segment length
        self.l2_ = 0.33
        self.s1_ = 0.11    # segment center of mass
        self.s2_ = 0.16
        self.i1_ = 0.025   # segment moment of inertia
        self.i2_ = 0.045
        self.b11_ = 0.5    # joint friction
        self.b22_ = 0.5 
        self.b12_ = 0.1
        self.b21_ = 0.1

        self.EPS = 1E-5    # finite difference epsilon

        self.mls = self.m2_*self.l1_*self.s2_
        self.iml = self.i1_ + self.i2_ + self.m2_*(self.l1_**2 )  
        self.dd = self.i2_*self.iml - self.i2_**2   

    def reset(self):
        self.index = 0
        self.terminal = False
        return self.index       
    
    def computeX_dot(self, X, U):
        """
        Dynamics model function
        Args:
            X:  current state vector [[q1, q2, q1_dot, q2_dot]]
            U:  Control vector [f1, f2]

        Returns:
            x_dot: the velocity of X
        """
        #---- compute inertia I and extra torque H ----
        #sy = np.sin(X[1])
        cy = np.cos(X[1])
        #---- inertia I ----
        I_11 = self.iml + 2*self.mls*cy
        I_12 = self.i2_ + self.mls*cy
        I_22 = self.i2_*np.ones(cy.shape)
        #---- determinant ----
        determinant = self.dd - self.mls**2*cy**2
        #---- inverse inertia I1 ----
        I1_11 = np.array(self.i2_/determinant)                 
        I1_12 = np.array((-self.i2_-self.mls*cy)/determinant)   
        I1_22 = np.array((self.iml+2*self.mls*cy)/determinant)  
        #---- temp vars ----
        sw = np.sin(X[1])
        #cw = np.cos(X[1])
        y = X[2]
        z = X[3]
        #---- extra torque H (Coriolis, centripetal, friction) ----
        H = np.array([- self.mls * (2*y + z)*z*sw + self.b11_*y + self.b12_*z,
            self.mls*(y**2)*sw + self.b22_*z + self.b12_*y])
        '''
        H = [-mls*(2*y+z).*z.*sw + b11_*y + b12_*z;...
                mls*y.^2.*sw + b22_*z + b12_*y];
        '''
        #----- compute xdot = inv(I) * (torque - H) -----
        torque = U - H
        x_dot = np.array( [ X[2], 
                            X[3],
                            I1_11*torque[0] + I1_12*torque[1],
                            I1_12*torque[0] + I1_22*torque[1]])

        return x_dot

    def step(self, x, u):
        """
        Dynamics model function
        Args:
            dt: time step
            x:  current state vector [q1, q2, q1_dot, q2_dot]
            U:  Control vector [f1, f2]

        Returns:
            Next state vector nextX.
        """
        #self.terminal = False
        nextX = x + self.dt * self.computeX_dot(x, u)
        nextX = self.fixBig(nextX, self.maxValue)	

        return nextX

    def fixBig(self, x, maxValue):
        #  Limit numbers that have become suspiciously large
        ibad = np.array(np.abs(x) > maxValue)
        x[ibad] = maxValue
        if ibad.any():
            print( "warning ilqg_det had to limit large numbers, results may be inaccurate")

        return x

    def getBatch(self, batch_size):
        uList = np.zeros((batch_size, self.Udim))
        xList = np.zeros((batch_size+1, self.Xdim))
        xList[0] = self.Xinit
        for i in range(batch_size):
            x1 = np.random.randint(-1,1)
            x2 = np.random.randint(-1,1)
            x = np.array([x1, x2])
            y = np.random.rand(2)
            r = x + y
            uList[i] = r
            xList[i+1] = self.step(xList[i], r)

        return xList, uList

    def doForwardLoop(self, X, uList, xdes):
        """
        Simulate controlled system, compute trajectory and cost
        Args:
            X:      initial state vector [q1_0, q2_0, q1_0_dot, q2_0_dot]
            uList:  control vector sequece [[f0_1, f0_2], [f1_1, f1_2], ..., [fn_1, fn_2]]
            xdes:   [2 x 1] destination (final) state vector [q1_n, q2_n]
        Returns:
            xList:  all states from state_0 to state_n [x0, x1, ..., xn] | x = [q1, q2, q1_dot, q2_dot]
            cost:   the cost of this control
        """
        # get sizes 
        #szX = x0.shape[0] # 4
        #szU = uList.shape[0] # 2
        #n = uList.shape[0]   # time-step
        # initialize simulation
        #xList = np.zeros((n, szX))
        #xList[0] = x0
        self.XList = [X]
        cost = 0.0
        # run simulation with substeps
        for i in range(uList.shape[0]):
            nextX = self.step(self.XList[i], uList[i])
            self.XList.append(nextX)
            cost += self.dt * self.costFunction.computeCost(self.XList[i], uList[i], xdes, False) # False = running cost
        cost +=  self.costFunction.computeCost(self.XList[-1], uList[-1], xdes, True)             # True = final state cost
        print ("--- xxxx c -----")
        print (cost)
        return self.XList, cost


    def computeModelDerivatives(self, x, u):
        """
        Compute model derivatives
        Args:
            x:       state vector [q1, q2, q1_dot, q2_dot]
            u:       Control vector [f1, f2]
        Returns:
            x_dot_x: derivative of x wrt x in shape [4, 4]:
                     [[  0.        ,    0.        ,    1.        ,    0.        ],
                     [   0.        ,    0.        ,    0.        ,    1.        ],
                     [   0.        ,  348.32079592,  -10.70037987,   11.45107693],
                     [   0.        , -834.01189623,   23.46962095,  -35.87991989]]

            x_dot_u: derivative of x wrt u in shape [4, 2]:
                     [[0.,  0.],
                     [ 0.,  0.],
                     [ 1.,  0.],
                     [ 0.,  1.]]
        """
        '''
        # Method_1:
        x1 = np.tile(x, (1,4)) + np.eye(4)*self.EPS
        x2 = np.tile(x, (1,4)) - np.eye(4)*self.EPS
        uu = np.tile(u,(1,4))

        f1 = self.computeX_dot(x1, uu) 
        f2 = self.computeX_dot(x2, uu)
        x_dot_x = (f1-f2)/2/self.EPS
    
        x_dot_u = np.zeros((4,2))
        x_dot_u[2:4,:] = np.eye(2)
        # End of Method_1
        '''

        # Not very sure:
        # Method 1 & 2 both work  
        
        # Method_2:
        fx = np.matrix([[0.0, 0.0, 1.0, 0.0],
			            [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]])

        fu = np.matrix([[ 0.,  0.],
                        [ 0.,  0.],
                        [ 1.,  0.],
                        [ 0.,  1.]])

        x_dot_x = fx
        x_dot_u = fu

        return x_dot_x, x_dot_u
