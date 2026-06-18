import numpy as np

from math import pi


class FK:

    def __init__(self):

        pass


    def forward(self, q):

        """

        INPUT:

        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]


        OUTPUTS:

        jointPositions - 8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector

                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.

                  The base of the robot is located at [0,0,0].

        T0e       - a 4 x 4 homogeneous transformation matrix,

                  representing the end effector frame expressed in the

                  world frame

        """

        c, s = np.cos, np.sin


        T0_1 = np.array([[c(q[0]), 0, -s(q[0]), 0],

                         [s(q[0]), 0, c(q[0]), 0],

                         [0,       -1, 0,       0.333],

                         [0,       0, 0,       1]], dtype=float)


        T1_2 = np.array([[c(q[1]), 0, s(q[1]), 0],

                         [s(q[1]), 0,  -c(q[1]), 0],

                         [0,      1,  0,       0],

                         [0,       0,  0,       1]], dtype=float)


        T2_3 = np.array([[c(q[2]), 0,  s(q[2]), 0.0825*c(q[2])],

                         [s(q[2]), 0, -c(q[2]), 0.0825*s(q[2])],

                         [0,       1,  0,       0.316],

                         [0,       0,  0,       1]], dtype=float)


        T3_4 = np.array([[c(q[3]), 0, -s(q[3]), -0.0825*c(q[3])],

                         [s(q[3]), 0,  c(q[3]), -0.0825*s(q[3])],

                         [0,      -1,  0,       0],

                         [0,       0,  0,       1]], dtype=float)


        T4_5 = np.array([[c(q[4]), 0,  s(q[4]), 0],

                         [s(q[4]), 0, -c(q[4]), 0],

                         [0,       1,  0,       0.384],

                         [0,       0,  0,       1]], dtype=float)


        T5_6 = np.array([[c(q[5]), 0,  s(q[5]), 0.088*c(q[5])],

                         [s(q[5]), 0, -c(q[5]), 0.088*s(q[5])],

                         [0,       1,  0,       0],

                         [0,       0,  0,       1]], dtype=float)


        T6_7 = np.array([[c(q[6]-np.pi/4), -s(q[6]-np.pi/4), 0, 0],

                         [s(q[6]-np.pi/4),  c(q[6]-np.pi/4), 0, 0],

                         [0, 0, 1, 0.210],

                         [0, 0, 0, 1]], dtype=float)


        

        H0 = np.eye(4, dtype=float)

        H1 = H0 @ T0_1

        H2 = H1 @ T1_2

        H3 = H2 @ T2_3

        H4 = H3 @ T3_4

        H5 = H4 @ T4_5

        H6 = H5 @ T5_6

        H7 = H6 @ T6_7   

        T0e = H7


      


        ez = np.array([0.0, 0.0, 1.0])


        p1 = np.array([0.0, 0.0, 0.141])                               

        p2 = H1[:3, 3]                                                 

        p3 = H2[:3, 3] + H2[:3, :3] @ (0.195 * ez)                      

        p4 = H3[:3, 3]                                                  

        p5 = H4[:3, 3] + H4[:3, :3] @ (0.125 * ez)                      

        p6 = H5[:3, 3] + H5[:3, :3] @ (-0.015 * ez)                                           

        p7 = H6[:3, 3] + H6[:3, :3] @ (0.051 * ez)                     

        p8 = T0e[:3, 3]                                                  


        jointPositions = np.vstack([p1, p2, p3, p4, p5, p6, p7, p8])



        return jointPositions, T0e




    # feel free to define additional helper methods to modularize your solution for lab 1

    
    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        H = self.compute_Ai(q)
        axes = np.zeros((3, 7), dtype=float)
        for i in range(1, 8):
            R_previous = H[i-1][:3, :3]
            axes[:, i-1] = R_previous[:, 2]   
        return axes
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        c, s = np.cos, np.sin
  
        T0_1 = np.array([[c(q[0]), 0, -s(q[0]), 0],
                        [s(q[0]), 0,  c(q[0]), 0],
                        [0,       -1,        0, 0.333],
                        [0,        0,        0, 1]], dtype=float)

        T1_2 = np.array([[c(q[1]), 0,  s(q[1]), 0],
                        [s(q[1]), 0, -c(q[1]), 0],
                        [0,       1,        0, 0],
                        [0,       0,        0, 1]], dtype=float)

        T2_3 = np.array([[c(q[2]), 0,  s(q[2]), 0.0825*c(q[2])],
                        [s(q[2]), 0, -c(q[2]), 0.0825*s(q[2])],
                        [0,       1,        0, 0.316],
                        [0,       0,        0, 1]], dtype=float)

        T3_4 = np.array([[c(q[3]), 0, -s(q[3]), -0.0825*c(q[3])],
                        [s(q[3]), 0,  c(q[3]), -0.0825*s(q[3])],
                        [0,      -1,        0, 0],
                        [0,       0,        0, 1]], dtype=float)

        T4_5 = np.array([[c(q[4]), 0,  s(q[4]), 0],
                        [s(q[4]), 0, -c(q[4]), 0],
                        [0,       1,        0, 0.384],
                        [0,       0,        0, 1]], dtype=float)

        T5_6 = np.array([[c(q[5]), 0,  s(q[5]), 0.088*c(q[5])],
                        [s(q[5]), 0, -c(q[5]), 0.088*s(q[5])],
                        [0,       1,        0, 0],
                        [0,       0,        0, 1]], dtype=float)

        T6_7 = np.array([[c(q[6]-np.pi/4), -s(q[6]-np.pi/4), 0, 0],
                        [s(q[6]-np.pi/4),  c(q[6]-np.pi/4), 0, 0],
                        [0,                0,               1, 0.210],
                        [0,                0,               0, 1]], dtype=float)

        H = [np.eye(4, dtype=float)]
        H.append(H[-1] @ T0_1)  # H1
        H.append(H[-1] @ T1_2)  # H2
        H.append(H[-1] @ T2_3)  # H3
        H.append(H[-1] @ T3_4)  # H4
        H.append(H[-1] @ T4_5)  # H5
        H.append(H[-1] @ T5_6)  # H6
        H.append(H[-1] @ T6_7)  # H7 (T0e)


        return H
    
if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0, -pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
