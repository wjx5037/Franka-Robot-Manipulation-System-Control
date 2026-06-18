import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """
    # make sure the data size
    q = np.asarray(q_in, float).reshape(-1) # (7,)
    v = np.asarray(v_in, float).reshape(-1) # (3,)
    w = np.asarray(omega_in, float).reshape(-1) # (3,)
    b = np.asarray(b, float).reshape(7, 1) # (7,1)
    
    vh = np.hstack([v, w]) # (6,)
    mask = ~np.isnan(vh)  # set mask if it is NAN
    
    dq_primary = IK_velocity(q, v, w).reshape(7, 1) # (7,1)
    J = calcJacobian(q)                      
    J = np.asarray(J, float)

    if not np.any(mask):
        dq = b.reshape(1, 7)
        return dq

    J_task = J[mask, :] # (m,7)  m<=6
    J_pinv_task = np.linalg.pinv(J_task) # (7,m)
    P = np.eye(7) - J_pinv_task @ J_task # (7,7)

    dq_null = P @ b # (7,1)
    null = (dq_null).reshape(1, 7) # (1.7)
    dq = (dq_primary).reshape(1, 7) # (1,7)

    return dq + null

