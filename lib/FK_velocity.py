import numpy as np 
from lib.calcJacobian import calcJacobian

def FK_velocity(q_in, dq):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param dq: 1 x 7 vector corresponding to the joint velocities.
    :return:
    velocity - 6 x 1 vector corresponding to the end effector velocities.    
    """
    J = calcJacobian(q_in)              
    dq = np.asarray(dq, dtype=float).reshape(-1)
    dq_col = dq.reshape(7, 1)           
    velocity = J @ dq_col              

    return velocity
