import numpy as np 
from lib.calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """
    J = calcJacobian(np.asarray(q_in, float))   
    v_in = np.asarray(v_in, float).reshape(-1)         
    omega_in = np.asarray(omega_in, float).reshape(-1)  
    vh = np.hstack([v_in, omega_in])                    

    mask = ~np.isnan(vh)              
    J_1 = J[mask, :]                
    vh_1 = vh[mask]                 

    qdot, *_ = np.linalg.lstsq(J_1, vh_1, rcond=None)
    dq = qdot.reshape(1, 7)
    
    return dq
