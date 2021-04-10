"""
@copyright: 2020--2030
@author: Haoxi Zhang

The cost function of the robot leg.

"""
import numpy as np

class CostFunctionOneLeg:
    def __init__(self):
        self.wp = 1E+5      # terminal position cost weight
        self.wv = 1E+5      # terminal velocity cost weight

    def computeCost(self, X, U, Xdes, t):
        """
        Compute cost
        Args:
            X:      state vector [q1, q2, q1_dot, q2_dot]
            U:      Control vector [f1, f2]
            Xdes:   [2 x 1] destination (final) state vector [q1_n, q2_n] 
            t:      boolean terminal flag

        Returns:
            l:   the cost of the control under the given state
        """
        l = 0.0
        if (t):  # terminal==True --- final state cost
            l = self.wp*np.sum((X[0:2] - Xdes)**2) + self.wv*np.sum(X[2:4]**2)
        else:     # running cost
            l = np.sum(U**2)

        return l

    def computeCostDerivatives(self, U):
        """
        Compute partial derivatives of cost
        Args:
            U:  Control vector in shape:
                [[f1],
                 [f2]]
        Returns:
            cost and derivatives 
        """
        l = np.sum(U**2)
        l_x = np.zeros((4,1))
        l_xx = np.zeros((4,4))
        l_u = 2*U
        l_uu = 2*np.eye(2) 
        l_ux = np.zeros((2,4)) 

        return l, l_x, l_xx, l_u, l_uu, l_ux

    def computeFinalCostDerivative(self, X, Xdes):
        l_x = np.zeros((4, 1))
        l_x[0:2] = 2*self.wp*(X[0:2]-Xdes)
        l_x[2:4] = 2*self.wv*X[2:4]  

        l_xx = 2 * np.diag([self.wp, self.wp, self.wv, self.wv])
        """
                l_xx :  
                [[ wp,  0,  0,  0 ],
                    0,  wp,  0,  0],
                    0,   0, wv,  0],
                    0,   0,  0, wv]
                ] 
        """
        # computer final state cost
        l = self.wp*np.sum((X[0:2] - Xdes)**2) + self.wv*np.sum(X[2:4]**2)

        return l, l_x, l_xx
