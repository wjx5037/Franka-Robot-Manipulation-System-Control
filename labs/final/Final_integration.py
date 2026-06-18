import sys
import numpy as np
from copy import deepcopy
from math import pi
import math
import rospy
from lib.calculateFK import FK
from lib.IK_position_null import IK
from core.interfaces import ArmController
from core.interfaces import ObjectDetector
from core.utils import time_in_seconds


# ===================== tool function =====================

def get_H_world_ee(arm, side_sign):

    q = np.array(arm.get_positions()).flatten()
    joint_positions, H_base_ee = fk_solver.forward(q)

    H_world_base = np.eye(4)
    H_world_base[0, 3] = 0.0
    H_world_base[1, 3] = side_sign * 0.99
    H_world_base[2, 3] = 0.0

    H_world_ee = H_world_base @ H_base_ee
    return H_world_ee


def wrap_to_pi(angle):
    
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def pose_to_xyz(pose):
    
    if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
        return pose[0, 3], pose[1, 3], pose[2, 3]
    else:
        raise ValueError(
            f"[ERROR] pose format incorrect: expected 4×4 numpy array, "
            f"got {type(pose)} with shape {getattr(pose,'shape','N/A')}"
        )


def detect_dynamic_blocks_world(arm, detector, H_ee_camera, turntable_center, side_sign):
    
    blocks = []

    # world -> EE
    H_world_ee = get_H_world_ee(arm, side_sign)
    # world -> camera
    H_world_cam = H_world_ee @ H_ee_camera

    cx, cy, cz = turntable_center

    for (name, pose_c) in detector.get_detections():
        if "dynamic" not in name.lower():
            continue

        x_c, y_c, z_c = pose_to_xyz(pose_c)
        p_c = np.array([x_c, y_c, z_c, 1.0])

        # camera -> world
        p_w = H_world_cam @ p_c
        x_w, y_w, z_w = p_w[0], p_w[1], p_w[2]

        MIN_BLOCK_CENTER_Z = 0.225   
        z_w = max(z_w, MIN_BLOCK_CENTER_Z)

        x_rel = x_w - cx
        y_rel = y_w - cy
        r = math.sqrt(x_rel**2 + y_rel**2)
        theta = math.atan2(y_rel, x_rel)

        blocks.append({
            "name":      name,
            "xyz":       np.array([x_w, y_w, z_w]),
            "r":         r,
            "theta":     theta
        })

    return blocks


def polar_to_cartesian(r, theta, z):
    
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return np.array([x, y, z])


def solve_IK_position(arm, target_pos, side_sign,
                      method='J_pseudo', alpha=0.5, q_seed=None, yaw_world=None):
    
    # 1) seed
    if q_seed is None:
        q_seed = np.array(arm.get_positions()).flatten()
    else:
        q_seed = np.array(q_seed).flatten()

    # 2) world -> base
    H_world_base = np.eye(4)
    H_world_base[0, 3] = 0.0
    H_world_base[1, 3] = side_sign * 0.99
    H_world_base[2, 3] = 0.0

    p_w = np.hstack([target_pos, 1.0])  # (4,)
    p_b = np.linalg.inv(H_world_base) @ p_w
    target_pos_base = p_b[:3]

    # 3) 
    if side_sign == -1:
        R_down = np.array([
            [-1.0, 0.0,  0.0],
            [ 0.0, 1.0,  0.0],
            [ 0.0, 0.0, -1.0],
        ])
    else:
        R_down = np.array([
            [ 1.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0, -1.0],
        ])

    if Grasp_mode == 0:
        R_down = np.array([
            [ 1.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0, -1.0],
        ])
    # 4) 
    if yaw_world is not None:
        
        gx = R_down[0, 1]   # x of y axis  
        gy = R_down[1, 1]   # y of y axis  
        yaw_current = math.atan2(gy, gx)

        delta_yaw = wrap_to_pi(yaw_world - yaw_current)

        cz = math.cos(delta_yaw)
        sz = math.sin(delta_yaw)
        Rz = np.array([
            [cz, -sz, 0.0],
            [sz,  cz, 0.0],
            [0.0, 0.0, 1.0],
        ])

        R_down = Rz @ R_down

    T_target = np.eye(4)
    T_target[:3, :3] = R_down
    T_target[:3, 3] = target_pos_base

    q_sol, rollout, success, msg = ik_solver.inverse(
        target=T_target,
        seed=q_seed,
        method=method,
        alpha=alpha
    )

    if not success:
        print("[WARN] IK not strictly converged:", msg)
        if q_sol is None:
            return None

    return q_sol

# ===================== dynamic grasp =====================

def dynamic_grasp_once(arm, detector,
                       TURNTABLE_CENTER,
                       THETA_GRASP,
                       R_MIN, R_MAX,
                       GRASP_DROP_Z,
                       MAX_WAIT_TIME,
                       SIDE_SIGN,
                       q_dynamic,
                       THETA_TOL=0.02): 
    
    print("\n[INFO] ====== Dynamic grasp (window-based) start ======")
    ########################################################################################################################################################
    if SIDE_SIGN < 0:
        THETA_LEAD = 0.30
    else:
        THETA_LEAD = 0.29
    # SIMULATION: 0.30 for blue, 0.32 for red  
    # HARDWARE: 0.45 for hardware RED1 
    # The parameter of simulation is useless. 
    # Collect the parameters for hardware is our job!!!!
    ########################################################################################################################################################
    # EE -> camera
    H_ee_camera = detector.get_H_ee_camera()

    t_global_start = time_in_seconds()
    NO_DYNAMIC_SEEN_TIMEOUT = 110.0
    t_last_seen_dynamic = time_in_seconds()

    target_name = None
    target_radius = None

    # -------- Step 1 --------

    while target_name is None and (time_in_seconds() - t_global_start) < MAX_WAIT_TIME:
        blocks = detect_dynamic_blocks_world(
            arm, detector, H_ee_camera, TURNTABLE_CENTER, SIDE_SIGN
        )

        if not blocks:
            if time_in_seconds() - t_last_seen_dynamic >= NO_DYNAMIC_SEEN_TIMEOUT:
                print(f"[MISSION] No dynamic blocks seen for {NO_DYNAMIC_SEEN_TIMEOUT:.0f}s (selection phase).")
                return "NO_DYNAMIC_TIMEOUT"
            rospy.sleep(0.05)
            continue
        else:
            t_last_seen_dynamic = time_in_seconds()

        angle_candidates = []

        for b in blocks:
            # 1) 
            r_b = b["r"]
            if not (R_MIN <= r_b <= R_MAX):
                continue

            # 2) 
            theta_b = b["theta"]
            delta0 = wrap_to_pi(theta_b - THETA_GRASP)  

            if delta0 < -(THETA_LEAD + THETA_TOL):
                angle_candidates.append((abs(delta0), b))

        if not angle_candidates:
            rospy.sleep(0.05)
            continue

        _, best = min(angle_candidates, key=lambda x: x[0])
        target_name = best["name"]
        target_radius = best["r"]

        print(f"[INFO] Choose target block {target_name}, "
              f"r={target_radius:.3f}, theta={best['theta']:.3f}")

        x_x, y_y, z_z = best["xyz"]
        print(f"[INFO] Target block world XYZ = "
            f"({x_x:.3f}, {y_y:.3f}, {z_z:.3f})")

    if target_name is None:
        print("[WARN] No suitable dynamic block selected within MAX_WAIT_TIME (selection phase).")
        return False

    # -------- Step 2 --------
    print("[INFO] Waiting for target block to enter grasp window...")

    cx, cy, cz = TURNTABLE_CENTER

    while (time_in_seconds() - t_global_start) < MAX_WAIT_TIME:
        blocks = detect_dynamic_blocks_world(
            arm, detector, H_ee_camera, TURNTABLE_CENTER, SIDE_SIGN
        )

        current = None
        for b in blocks:
            if b["name"] == target_name:
                current = b
                break

        if current is None:
            rospy.sleep(0.01)
            continue

        theta_now = current["theta"]
        delta = wrap_to_pi(theta_now - THETA_GRASP)

        lower_bound = -THETA_LEAD - THETA_TOL
        upper_bound = -THETA_LEAD + THETA_TOL

        print(f"[DEBUG] {target_name}: theta={theta_now:.3f}, "
            f"delta={delta:.3f}, r={current['r']:.3f}, "
            f"window=[{lower_bound:.3f}, {upper_bound:.3f}]")

        if delta >= -THETA_LEAD: #if lower_bound <= delta <= upper_bound:
            print("[INFO] Block in *early* window, start grabbing motion.")

            r_use = current["r"]

            cx, cy, cz = TURNTABLE_CENTER
            x_rel, y_rel, _ = polar_to_cartesian(r_use, THETA_GRASP, 0.0)
            x = cx + x_rel
            y = cy + y_rel # -0.01 for red, +0.01 for blue Hareware

            Z_APPROACH = GRASP_DROP_Z + 0.05   
            target_up = np.array([x, y, Z_APPROACH])

            q_up = solve_IK_position(arm, target_up, SIDE_SIGN, q_seed=q_dynamic)
            if q_up is None:
                print("[WARN] IK failed for approach (up) pose, RE-selecting!! ")
                return dynamic_grasp_once(
                    arm, detector,
                    TURNTABLE_CENTER,
                    THETA_GRASP,
                    R_MIN, R_MAX,
                    GRASP_DROP_Z,
                    MAX_WAIT_TIME,
                    SIDE_SIGN,
                    q_dynamic,
                )
            print(f"[INFO] q_up = {q_up}")
           
            target_down = np.array([x, y, GRASP_DROP_Z])
            q_drop = solve_IK_position(arm, target_down, SIDE_SIGN, q_seed=q_up)

            if q_drop is None:
                print("[WARN] IK failed for drop position, RE-selecting!!")
                arm.safe_move_to_position(q_dynamic)
                return dynamic_grasp_once(
                    arm, detector,
                    TURNTABLE_CENTER,
                    THETA_GRASP,
                    R_MIN, R_MAX,
                    GRASP_DROP_Z,
                    MAX_WAIT_TIME,
                    SIDE_SIGN,
                    q_dynamic,
                ) 
            print(f"[INFO] q_drop = {q_drop}")

            arm.safe_move_to_position(q_up)
            arm.safe_move_to_position(q_drop)

            # ------ close gripper ------
            BLOCK_WIDTH = 0.05
            GRIP_MARGIN = 0.005
            GRIP_WIDTH  = BLOCK_WIDTH - GRIP_MARGIN
            GRIP_FORCE  = 100.0

            ok = arm.exec_gripper_cmd(GRIP_WIDTH, GRIP_FORCE)
            if not ok:
                print("[WARN] exec_gripper_cmd failed (no gripper?).")
                return False

            arm.safe_move_to_position(q_dynamic)

            state = arm.get_gripper_state()
            if not state:
                print("[WARN] No gripper state available.")
                return False

            width_now = float(np.mean(state['position']))
            force_now = float(np.mean(state['force']))
            gap_now = 2.0 * width_now

            MIN_GAP_WITH_BLOCK = 0.04 
            MAX_GAP_WITH_BLOCK = 0.073
            FORCE_THRESH = 5.0

            if MIN_GAP_WITH_BLOCK <= gap_now <= MAX_GAP_WITH_BLOCK or force_now >= FORCE_THRESH: 
                print(f"[INFO] Grasp OK: gap_now={gap_now:.4f}, "
                      f"force={force_now:.2f}")
                print("[INFO] ====== Dynamic grasp finished (SUCCESS) ======")
                return True
            else:
                print(f"[WARN] Grasp maybe failed: gap_now={gap_now:.4f}, "
                      f"force={force_now:.2f}")
                print("[INFO] ====== Dynamic grasp finished (FAILED) ======")
                return False

        rospy.sleep(0.01)

    print("[WARN] Waiting for grasp window timed out.")
    print("[INFO] ====== Dynamic grasp finished (TIMEOUT) ======")
    return False

# ===================== static grasp =====================

def detect_static_blocks_world(arm, detector, H_ee_camera, side_sign):
    
    blocks = []

    # world -> EE
    H_world_ee = get_H_world_ee(arm, side_sign)
    # world -> camera
    H_world_cam = H_world_ee @ H_ee_camera

    base_x = 0.0
    base_y = side_sign * 0.99

    for (name, pose_c) in detector.get_detections():
        if "static" not in name.lower():
            continue

        x_c, y_c, z_c = pose_to_xyz(pose_c)
        p_c = np.array([x_c, y_c, z_c, 1.0])

        # camera -> world
        p_w = H_world_cam @ p_c
        x_w, y_w, z_w = p_w[0], p_w[1], p_w[2]
        if side_sign < 0:
            x_w += 0.01
            y_w -= 0.025
        else:
            y_w += 0.02
       
        MIN_BLOCK_CENTER_Z = 0.23
        z_w = max(z_w, MIN_BLOCK_CENTER_Z)

        dist = math.hypot(x_w - base_x, y_w - base_y)

        # ========= cacualte grip_yaw =========
        H_cam_block = pose_c
        H_world_block = H_world_cam @ H_cam_block
        R_world_block = H_world_block[:3, :3]

        vx0, vy0, vz0 = R_world_block[0, 0], R_world_block[1, 0], R_world_block[2, 0]
        vx1, vy1, vz1 = R_world_block[0, 1], R_world_block[1, 1], R_world_block[2, 1]

        if abs(vz0) <= abs(vz1):
            ex, ey = vx0, vy0  
        else:
            ex, ey = vx1, vy1   

        norm_e = math.hypot(ex, ey)
        if norm_e < 1e-6:
            ex, ey = 1.0, 0.0
            norm_e = 1.0
        ex /= norm_e
        ey /= norm_e

        edge_yaw = math.atan2(ey, ex)

        grip_dir_x = -ey
        grip_dir_y =  ex
        grip_yaw = math.atan2(grip_dir_y, grip_dir_x)
        if grip_yaw is not None:
            grip_yaw = wrap_to_pi(grip_yaw)
            if side_sign < 0:
                if grip_yaw > 0:
                    if grip_yaw > pi/2:
                        grip_yaw -= pi
                    else:
                        grip_yaw -= pi/2
                if grip_yaw < 0:
                    if grip_yaw < -pi/2:
                        grip_yaw += pi/2
                    else:
                        grip_yaw = grip_yaw
            else:
                if grip_yaw > 0:
                    grip_yaw -= pi/2
                if grip_yaw < 0:
                    grip_yaw += pi/2

        blocks.append({
            "name": name,
            "xyz":  np.array([x_w, y_w, z_w]),
            "dist": dist,
            "edge_yaw":  edge_yaw,
            "grip_yaw": grip_yaw
        })

    return blocks


def static_grasp_once(arm, detector,
                      side_sign,
                      q_static_obs,
                      GRASP_DROP_Z_STATIC,
                      block_info,
                      PRE_GRASP_DZ=0.05):
   
    print("\n[INFO] ====== Static grasp start ======")

    target_name   = block_info["name"]
    target_center = block_info["xyz"].copy()
    grip_yaw      = block_info.get("grip_yaw", None)  

    align_deg = math.degrees(grip_yaw) if grip_yaw is not None else 0.0
    print(f"[INFO] Using cached static block {target_name}, "
          f"world xyz = ({target_center[0]:.3f}, {target_center[1]:.3f}, {target_center[2]:.3f}), "
          f"grip_yaw = {grip_yaw if grip_yaw is not None else 0.0:.3f} rad ({align_deg:.1f} deg)")

    # pre-grasp & drop
    target_up = target_center.copy()
    target_up[2] += PRE_GRASP_DZ

    target_down = target_center.copy()
    target_down[2] = GRASP_DROP_Z_STATIC

    # ---- Step 1 ----
    q_up_plain = solve_IK_position(
        arm, target_up, side_sign,
        q_seed=q_static_obs,
        yaw_world=None
    )
    if q_up_plain is None:
        print("[WARN] IK failed for static approach (plain up).")
        return False

    print(f"[INFO] static q_up_plain = {q_up_plain}")
    arm.safe_move_to_position(q_up_plain)

    # ---- Step 2 ----
    if grip_yaw is not None:
        q_up = solve_IK_position(
            arm, target_up, side_sign,
            q_seed=q_up_plain, # q_up_plain
            yaw_world=grip_yaw
        )
        if q_up is None:
            print("[WARN] IK with yaw for static up failed, fallback to plain up.")
            q_up = q_up_plain
        else:
            print(f"[INFO] static q_up_yaw = {q_up}")
            arm.safe_move_to_position(q_up)
    else:
        q_up = q_up_plain
    # ---- Step 3 ----
    q_drop = solve_IK_position(
        arm, target_down, side_sign,
        q_seed=q_up,
        yaw_world=grip_yaw
    )
    if q_drop is None:
        print("[WARN] IK with yaw for static drop failed.")
        arm.safe_move_to_position(q_static_obs)
        return False

    print(f"[INFO] static q_drop = {q_drop}")

    # ---- Step 4 ----
    arm.safe_move_to_position(q_up)
    arm.safe_move_to_position(q_drop)

    BLOCK_WIDTH = 0.05
    GRIP_MARGIN = 0.005
    GRIP_WIDTH  = BLOCK_WIDTH - GRIP_MARGIN
    GRIP_FORCE  = 50.0

    ok = arm.exec_gripper_cmd(GRIP_WIDTH, GRIP_FORCE)
    if not ok:
        print("[WARN] exec_gripper_cmd failed (no gripper?).")
        arm.safe_move_to_position(q_static_obs)
        return False

    arm.safe_move_to_position(q_up)

    # ---- Step 5 ----
    state = arm.get_gripper_state()
    if not state:
        print("[WARN] No gripper state available.")
        arm.safe_move_to_position(q_static_obs)
        return False

    width_now = float(np.mean(state['position']))
    force_now = float(np.mean(state['force']))
    gap_now   = 2.0 * width_now

    MIN_GAP_WITH_BLOCK = 0.02
    MAX_GAP_WITH_BLOCK = 0.073
    FORCE_THRESH       = 5.0

    if MIN_GAP_WITH_BLOCK <= gap_now <= MAX_GAP_WITH_BLOCK or force_now >= FORCE_THRESH:
        print(f"[INFO] Static grasp OK at q_up: gap_now={gap_now:.4f}, force={force_now:.2f}")
        print("[INFO] ====== Static grasp finished (SUCCESS) ======")
        return True
    else:
        print(f"[WARN] Static grasp maybe failed at q_up: gap_now={gap_now:.4f}, force={force_now:.2f}")
        print("[INFO] ====== Static grasp finished (FAILED) ======")
        arm.safe_move_to_position(q_static_obs)
        return False


def get_stack_target_world(stack_level, GOAL_PLATFORM_CENTER):
    
    cx, cy, cz = GOAL_PLATFORM_CENTER
    z = 0.01*(stack_level + 1) + cz + (BLOCK_SIZE / 2.0) + stack_level * BLOCK_SIZE # every block 1cm offset
    return np.array([cx, cy, z])

def place_on_goal_stack(arm, side_sign, stack_level,
                        GOAL_PLATFORM_CENTER,
                        open_width=0.09,
                        open_force=30.0):
    
    target_center = get_stack_target_world(stack_level, GOAL_PLATFORM_CENTER)

    target_up = target_center.copy()
    target_up[2] += 0.10

    # 1) 
    if Grasp_mode == 0:
        q_place = q_static_obs
    else:
        q_place = q_dynamic

    q_up = solve_IK_position(arm, target_up, side_sign, q_seed=q_place)
    if q_up is None:
        print("[WARN] IK failed for goal up pose.")
        return False, stack_level
    print(f"[INFO] pre tower level position = {q_up} ")
    arm.safe_move_to_position(q_up)

    # 2) 
    q_down = solve_IK_position(arm, target_center, side_sign, q_seed=q_up)
    if q_down is None:
        print("[WARN] IK failed for goal down pose.")
        return False, stack_level
    
    print(f"[INFO] tower level position = {q_down} ")
    arm.safe_move_to_position(q_down)

    # 3) 
    ok = arm.exec_gripper_cmd(open_width, open_force)
    if not ok:
        print("[WARN] Failed to open gripper at goal.")
        return False, stack_level

    rospy.sleep(0.2)   

    # 4) 
    arm.safe_move_to_position(q_up)

    new_level = stack_level + 1
    print(f"[INFO] Placed block at stack_level={stack_level}. New level={new_level}")
    return True, new_level


# ===================== main =====================

if __name__ == "__main__":
    try:
        team = rospy.get_param("team")  # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        sys.exit(1)

    rospy.init_node("team_script")

    arm = ArmController()
    detector = ObjectDetector()
    fk_solver = FK()
    ik_solver = IK()
    arm.set_arm_speed(0.2)

    # ---------- initial position ----------
    start_position = np.array([
        -0.01779206, -0.76012354,  0.01978261,
        -2.34205014, 0.02984053,   1.54119353 + pi/2,
        0.75344866
    ])
    arm.safe_move_to_position(start_position)

    arm.open_gripper()
    print("[INFO] Gripper opened (using built-in open_gripper()).")


    # ---------- team info ----------
    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n")
    print("Go!\n")

    # ---------- side_sign ----------
    if team == 'red':
        side_sign = -1.0   # red team in -y
    else:
        side_sign = +1.0   # blue team in +y

    print(f"[INFO] team = {team}, side_sign = {side_sign}")

    if side_sign < 0:
        q_dynamic = np.array([
            (1.51904649 + 0.12 ),
            0.77707742 - 0.3,
            0.08158916,
            -1.13651913 - 0.1, #0.2
            -0.0606659,
            1.91135068 - 0.02 , #0.1
            2.38296356 - pi
        ])
    else:
        q_dynamic = np.array([
            (-1.51904649 * side_sign),
            0.77707742 - 0.3,
            0.08158916,
            -1.13651913 - 0.1, #0.2
            -0.0606659,
            1.91135068 , #0.1
            2.38296356 - pi
        ])
   
    # Grasp_mode = 0
    # arm.safe_move_to_position(arm.neutral_position())
    if side_sign < 0:
      
        q_static_obs = np.array([-0.14915829, 0.03110731, -0.15634994, -1.58882114, 0.00484883, 1.61954868 -0.01, 0.47972805])

    else:

        q_static_obs = np.array([0.16648511, 0.03102723, 0.13864984, -1.58882169, -0.00429258, 1.6195509 -0.01, 1.09067643])

    # kkk = np.array([0.562, side_sign * 0.831 , 0.6])
    # q_static_obs = solve_IK_position(arm, kkk, side_sign, q_seed=arm.neutral_position())
    # print(f"[INFO] = {q_static_obs}")
    
    # qkkk= np.array([ 0.24009831, 0.22494166, 0.04549249, -1.06045231, -0.01057205, 1.28518411, 1.06686567])
    # arm.safe_move_to_position(qkkk)
    # ---------- turn table ----------
    TURNTABLE_CENTER = np.array([0.0, 0.0, 0.200])

    # ---------- grasp range ----------
    R_MIN = 0.10
    R_MAX = 0.36
    
    # ---- Block & goal platform params ----
    BLOCK_SIZE = 0.05        # 50 mm
    PLATFORM_Z = 0.2      # goal platform surface height (world z)

    GOAL_PLATFORM_CENTER = np.array([
        0.562,
        side_sign * 0.831,
        PLATFORM_Z
    ])

    # ---------- grasp angle（red -90°, blue +90°） ----------
    if side_sign > 0:
        THETA_GRASP = +pi/2   # blue：+90°
    else:
        THETA_GRASP = -pi/2   # red：-90°

    print(f"[INFO] THETA_GRASP = {THETA_GRASP:.3f} rad")

    # ---------- grasp angle ----------
    GRASP_DROP_Z = 0.23    
    GRASP_DROP_Z_STATIC = 0.23  # Hardware + 0.01 

    # ---------- max wait time  ----------
    MAX_WAIT_TIME = 120.0  

    stack_level = 0
    MAX_STACK_LEVEL = 9 
    Grasp_mode = 1 # 0 means doing static, 1 means doing dynamic

    MATCH_TIME = 400
    t_match_start = time_in_seconds()

    # ========= DYNAMIC loop =========
    arm.safe_move_to_position(q_dynamic)
    print(f"[INFO] q_dynamic = {q_dynamic}")
    print("\n========== Dynamic Grasp Loop ==========")


    while True:
        Grasp_mode = 1
        t_now = time_in_seconds()
        if t_now - t_match_start > MATCH_TIME:
            print("[MISSION] Time up, stop dynamic loop.")
            break
        if stack_level >= MAX_STACK_LEVEL:
            print("[MISSION] Stack is full (level = {}).".format(stack_level))
            break

        arm.safe_move_to_position(q_dynamic)

        print("\n[LOOP] Start one dynamic grasp, current stack_level =", stack_level)

        ret = dynamic_grasp_once(
            arm, detector,
            TURNTABLE_CENTER,
            THETA_GRASP,
            R_MIN, R_MAX,
            GRASP_DROP_Z,
            MAX_WAIT_TIME,
            side_sign,
            q_dynamic,
        )

        if ret == "NO_DYNAMIC_TIMEOUT":
            print("[MISSION] Switch to static: no dynamic blocks seen for 108s.")
            break

        if not ret:
            print("[WARN] dynamic_grasp_once failed, try again next loop.")
            rospy.sleep(0.2)
            arm.open_gripper()
            continue

        print("[INFO] One dynamic grasp succeeded, go place on goal stack.")

        place_ok, stack_level = place_on_goal_stack(
            arm, side_sign, stack_level, GOAL_PLATFORM_CENTER,
            open_width=0.09, open_force=30.0
        )
        if not place_ok:
            print("[WARN] place_on_goal_stack failed, stop loop for safety.")
            break

        # arm.safe_move_to_position(q_dynamic)

    # ========= STATIC loop =========
    arm.safe_move_to_position(q_static_obs)
    print(f"[INFO] q_static_obs = {q_static_obs}")

    H_ee_camera = detector.get_H_ee_camera()
    static_blocks = detect_static_blocks_world(arm, detector, H_ee_camera, side_sign)

    if not static_blocks:
        print("[WARN] No static blocks detected at init!")
    else:
        static_blocks.sort(key=lambda b: b["dist"])
        for i, b in enumerate(static_blocks):
            print(f"[INIT] static[{i}]: {b['name']}, xyz=({b['xyz'][0]:.3f}, "
                f"{b['xyz'][1]:.3f}, {b['xyz'][2]:.3f}), dist={b['dist']:.3f}")

    static_index = 0
    MAX_STATIC = min(4, len(static_blocks))

    static_attempt = 0
    while True:
        Grasp_mode = 0
        t_now = time_in_seconds()
        if t_now - t_match_start > MATCH_TIME:
            print("[MISSION] Time up during static phase.")
            break
        if stack_level >= MAX_STACK_LEVEL:
            print("[MISSION] Stack is full during static phase (level = {}).".format(stack_level))
            break
        if static_index >= MAX_STATIC or static_index >= len(static_blocks):
            print("[MISSION] No more cached static blocks to grab.")
            break

        static_attempt += 1
        print(f"\n[STATIC LOOP] Start static grasp #{static_attempt}, "
            f"current stack_level = {stack_level}, static_index = {static_index}")

        block_info = static_blocks[static_index]

        success_grasp = static_grasp_once(
            arm, detector, side_sign,
            q_static_obs, GRASP_DROP_Z_STATIC, block_info
        )

        if not success_grasp:
            print("[WARN] static_grasp_once returned False, try again next loop.")
            arm.open_gripper()
            arm.safe_move_to_position(q_static_obs)
            continue

        static_index += 1
        print("[INFO] One static grasp succeeded, go place on goal stack.")

        place_ok, stack_level = place_on_goal_stack(
            arm, side_sign, stack_level, GOAL_PLATFORM_CENTER,
            open_width=0.09, open_force=30.0
        )
        if not place_ok:
            print("[WARN] place_on_goal_stack failed after static grasp, stop static phase.")
            break

    # ---------- all finish ----------
    arm.safe_move_to_position(arm.neutral_position())
    print("[MISSION] Match finished, final stack_level =", stack_level)
