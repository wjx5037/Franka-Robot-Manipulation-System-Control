import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """
    fk = FK()
    H = fk.compute_Ai(q_in)         
    z = fk.get_axis_of_rotation(q_in) 
    eep = H[7][:3, 3]       
    Jv = np.zeros((3, 7))
    Jw = np.zeros((3, 7))

    for i in range(7):
        p_previous = H[i][:3, 3]
        z_previous = z[:, i]
        Jw[:, i] = z_previous
        Jv[:, i] = np.cross(z_previous, eep - p_previous)

    J = np.vstack([Jv, Jw])
    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))

