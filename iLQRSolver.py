"""
@Copyright (C) 2020--2030
@Author: Haoxi Zhang

Iterative LQG program for nonlinear plants with bounded controls
based on the algorithm described in:
      Todorov, E. and Li, W. (2005) A generalized iterative LQG method for
      locally-optimal feedback control of constrained nonlinear stochastic
      systems.

- DYNAMICS:  X(t+1) = X(t) + dt * DynamicModel(X(t), U(t))
                    where X(0) = x0, t = 1:n-1

- COST:      fnCost(X(n)) + sum(dt * fnCost(x(t),u(t),t))

- CONTROL:   u(t) = u_(t) + L(:,:,t) * (x(t) - x_(t))

"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import time

# Algorithms from iterative Linear Quadratic Gaussian(iLQG) trajectory optimizer, variant of the classic DDP
###############################################################################################################################
# Steps for solving Trajectory Optimizer
#
# 0.   Initiallization x, u, l(cost function), f, Q, V
# 1.   Derivatives
#      Given a nominal sequence (x, u, i) computes first and second derivatives of l and f, which will be each, Jacobian and Hessian
# 2-1. BackWard Pass
#      Iterating equations related to Q, K and V for decreasing i = N-1, ... 1.
# 2-2. Condition Hold
#      If non-PD(Positive Definite) Q_uu is encountered, increase mu, and restart the backward pass or, decrease mu if successful.
# 3-1. Forward Pass
#      set alpha = 1. Iterate the new controller u and the new nominal sequence (x, u, i)
# 3-2. Convergence
#      A condition for convergence is satisfied, done, or decrease alpha and restart the forward pass.
###############################################################################################################################


class iLQRSolver:
    def __init__(self, model, dt, costFunction):
        # Initializations 
        # model, costFunction, X, U, F, derivatives 
        self.model = model
        self.costFunction = costFunction
        self.Xinit = np.zeros((model.Xdim, 1))
        self.Xdes = np.zeros((2, 1))
        self.szX = self.model.Xdim     # size of state vector:   4
        self.szU = self.model.Udim     # size of control vector: 2
        self.X = np.zeros((self.szX, 1)) 
        self.U = np.zeros((self.szU, 1))
        self.n = 50
        self.dt = dt
        self.iterMax = 200
        self.uMin = np.tile(-np.inf, (self.szU, 1))
        self.uMax = np.tile( np.inf, (self.szU, 1))

        # ---- user-adjustable parameters ----
        self.lambdaInit = 0.1       # initial value of Levenberg-Marquardt lambda
        self.lambdaFactor = 10      # factor for multiplying or dividing lambda
        self.lambdaMax = 1000       # exit if lambda exceeds threshold
        self.epsConverge = 0.001    # exit if relative improvement below threshold
        self.maxIter = 100          # exit if number of iterations reaches threshold
        self.flgPrint = 1           # show cost- 0:never, 1:every iter, 2:final
        self.maxValue = 1E+10       # upper bound on states and costs (Inf: none)
        self.flgFix = 0             # clear large-fix global flag
        # ---- zero functions for initialization ----
        self.zerosState = np.zeros((self.model.Xdim, 1))
        self.zerosCommand = np.zeros((self.model.Udim, 1))
    
    def trajectoryOptimizer(self, Xinit, Xdes, n, iterMax = 20):
        #initialization
        self.Xinit = Xinit
        self.Xdes = Xdes
        self.n = n
        self.iterMax = iterMax
        # ---- iLQG parameters ----
        # 1: derivatives from cost
        self.s0= np.zeros(self.n)   # shape (1) -- l
        self.s = np.zeros((self.n, self.szX, 1))  # shape (n, 4, 1) -- l_x
        self.S = np.zeros((self.n, self.szX, self.szX))  # shape(4,4) -- l_xx

        self.q0= np.zeros(self.n)   # shape (1) -- l
        self.q = np.zeros((self.n, self.szX, 1))  # shape (n, 4, 1) -- l_x
        self.Q = np.zeros((self.n, self.szX, self.szX))  # shape(4,4) -- l_xx

        self.r = np.zeros((self.n, self.szU, 1))    # shape (n, 2, 1) -- l_u
        self.R = np.zeros((self.n, self.szU, self.szU))  # shape(n, 2, 2) -- l_uu
        self.P = np.zeros((self.n, self.szU, self.szX))  # shape(n, 2, 4) -- l_ux --> maybe change to (4,2) from CostFntOneLeg

        # 2: derivatives from DynamicModel
        self.A = np.zeros((self.n, self.szX, self.szX))  # shape(n, 4, 4) -- x_dot_x
        self.B = np.zeros((self.n, self.szX, self.szU))  # shape(n, 4, 2) -- x_dot_u

        self.l = np.zeros((self.n-1, self.szU, 1)) 
        self.L = np.zeros((self.n-1, self.szU, self.szX))# init feedback gains, shape(n, 2, 4)
        # ---- end iLQG parameters-------------------
        self.initTrajectory()
        self.XList, self.cost = self.model.doForwardLoop(Xinit, self.UList, self.Xdes)
        print ("initial cost")
        print (self.cost)
        #---- optimization loop ----
        lambda_ = self.lambdaInit
        flgChange = 1

        for iter in range(iterMax):
            #---- STEP 1: backwardloop - approximate dynamics and cost along new trajectory -----
            if flgChange:
                self.s0[n-1], self.s[n-1,:], self.S[n-1,:] = self.costFunction.computeFinalCostDerivative(self.XList[n-1], self.Xdes)

                for k in reversed(range(n-1)): # k = n-2 to 0
                    # quadratize cost, adjust for dt
                    l0, l_x, l_xx, l_u, l_uu, l_ux = self.costFunction.computeCostDerivatives(self.UList[k])

                    self.q0[k] = self.dt * l0
                    self.q[k,:]= self.dt * l_x
                    self.Q[k,:]= self.dt * l_xx
                    self.r[k,:]= self.dt * l_u
                    self.R[k,:]= self.dt * l_uu
                    self.P[k,:]= self.dt * l_ux

                    # linearize dynamics, adjust for dt
                    f_x, f_u = self.model.computeModelDerivatives(self.XList[0], self.UList[0])

                    self.A[k,:] = np.eye(self.szX) + self.dt * f_x
                    self.B[k,:] = self.dt * f_u

                flgChange = 0

            #---- STEP 2: compute optimal control law and cost-to-go ----
            for k in reversed(range(n-1)): # k = n-2 to 0
                # compute shortcuts g,G,H
                g = self.r[k] + self.B[k,:].conj().T @self.s[k+1]
                G = self.P[k,:] + self.B[k,:].conj().T @self.S[k+1,:] @self.A[k,:]
                H = self.R[k,:] + self.B[k,:].conj().T @self.S[k+1,:] @self.B[k,:]
               
                # find control law
                self.l[k], self.L[k,:] = self.uOptimal(g,G,H,self.UList[k], self.uMin, self.uMax, lambda_)
                
                # update cost-to-go approximation
                self.S[k,:] = self.Q[k,:] + self.A[k,:].conj().T @self.S[k+1,:] @self.A[k,:] + \
                                self.L[k,:].conj().T @H @self.L[k,:] + self.L[k,:].conj().T @G + G.conj().T @self.L[k,:]
                
                self.s[k] = self.q[k,:] + self.A[k,:].conj().T @self.s[k+1,:] + \
                                self.L[k,:].conj().T @H @self.l[k,:]  + self.L[k,:].conj().T @g + G.conj().T @self.l[k,:]

                self.s0[k] = self.q0[k] + self.s0[k+1] + \
                                self.l[k,:].conj().T @H @self.l[k,:]/2 + self.l[k,:].conj().T @g
                
                # HACK USED TO PREVENT OCCASIONAL DIVERGENCE
                if np.isfinite(self.maxValue):
                    self.S[k,:] = self.fixBig(self.S[k,:], self.maxValue)
                    self.s[k] = self.fixBig(self.s[k,:], self.maxValue)
                    self.s0[k] = np.minimum(self.s0[k], self.maxValue)

            # ---- STEP 3: new control sequence, trajectory, cost ----
            # simulate linearized system to compute new control
            dx = np.zeros((self.szX, 1))
            for k in range(n-1):
                du = self.l[k,:] + self.L[k,:] @dx
                du = np.minimum(np.maximum(du + self.UList[k], self.uMin), self.uMax) - self.UList[k]
                dx = self.A[k,:] @dx + self.B[k,:] @du 
                self.UListnew[k] =  self.UList[k] + du 
            # simulate system to compute new trajectory and cost
            xnew, costnew = self.model.doForwardLoop(Xinit, self.UListnew, self.Xdes)

            #---- STEP 4: Levenberg-Marquardt method ----
            if costnew < self.cost :
                # decrease lambda (get closer to Newton method)
                lambda_ = lambda_ / self.lambdaFactor
        
                # accept changes, flag changes
                self.UList = self.UListnew
                self.XList = xnew
                
                flgChange = 1

                if self.flgPrint==1:
                    print("Iteration = %d; Cost = %.4f; logLambda = %.1f\n"% (iter, costnew, np.log10(lambda_)))

                #if iter>1 & (abs(costnew - self.cost)/self.cost < self.epsConverge):
                #   self.cost = costnew
                #    break # improvement too small: EXIT

                self.cost = costnew

            else:
                # increase lambda (get closer to gradient descent)
                lambda_ = lambda_ * self.lambdaFactor
                if lambda_ > self.lambdaMax:
                    break # lambda_ is too large: EXIT

            # print final result if necessary
            if self.flgPrint==2:
                print("Iterations = %d;  Cost = %.4f\n"% (iter, self.cost))
            if self.flgFix:
                print( "warning ilqg_det had to limit large numbers, results may be inaccurate")
                self.flgFix = 0

        return self.XList, self.UList
    
    def initTrajectory(self):
        #self.XList = np.array([self.zerosState for i in range(self.n)])   # [n, 4, 1]
        self.UList = np.array([self.zerosCommand for i in range(self.n)]) # [n, 2, 1]
        self.UListnew = np.array([self.zerosCommand for i in range(self.n)]) # [n, 2, 1]
        return 0
    
    def uOptimal(self, g, G, H, u, uMin, uMax, lambda_):
        # Compute optimal control law
        # eigenvalue decomposition, modify eigenvalues
        D,V = np.linalg.eig(H)  # np.linalg.eigvals  or np.linalg.eig
        d = np.diag(D)
        d[d<0] = 0
        d = d + lambda_
        # inverse modified Hessian, unconstrained control law
        H1 = V * np.diag(1/d) * V.conj().T
        l = - H1 @ g
        L = - H1 @ G
        # enforce constraints
        l = np.minimum(np.maximum(l+u,uMin),uMax) - u
        b_index = np.logical_or((l+u==uMin),(l+u==uMax)).reshape(2)
        # modify L to reflect active constraints
        L[ b_index,: ] = 0

        return l, L

    def fixBig(self, x, maxValue):
        #  Limit numbers that have become suspiciously large
        ibad = np.array(np.abs(x) > maxValue)
        x[ibad] = maxValue
        if ibad.any():
            self.flgFix = 1

        return x