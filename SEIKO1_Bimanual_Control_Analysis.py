import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand
from std_msgs.msg import Float32  # Added for rqt_plot

import numpy as np
import os
import pinocchio as pin
import threading
import csv
import time

# OFFICIAL LIBRARY
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from seiko_stacking.solver import SeikoSolver

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
BASE_SEPARATION = 0.60
HALF_DIST = BASE_SEPARATION / 2.0

OBJ_X = 0.40
OBJ_Y_L =  HALF_DIST  
OBJ_Y_R = -HALF_DIST  
OBJ_Z = 0.10          

PLACE_X = 0.30
PLACE_Y_L =  HALF_DIST
PLACE_Y_R = -HALF_DIST
PLACE_Z = 0.15        

# SEIKO Tuning
FILTER_GAIN = 0.2
VEL_LIMIT = 1.0
DRIVER_DOF = 6
LOG_FILE = os.path.expanduser("~/seiko_data.csv")
# ==========================================

class LowPassFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha; self.prev = None
    def update(self, val):
        if self.prev is None: self.prev = val
        self.prev = self.alpha * val + (1.0 - self.alpha) * self.prev
        return self.prev

class SeikoBimanualSync(Node):
    def __init__(self):
        super().__init__('seiko_bimanual_sync')
        self.logdebug = self.get_logger().debug
        self.loginfo = self.get_logger().info
        self.logwarn = self.get_logger().warning
        self.logerror = self.get_logger().error

        # 1. Initialize Interbotix Interfaces
        self.arm_L = InterbotixManipulatorXS(robot_model='vx300s', group_name='arm', gripper_name='gripper', robot_name='arm_1', node=self)
        self.arm_R = InterbotixManipulatorXS(robot_model='vx300s', group_name='arm', gripper_name='gripper', robot_name='arm_2', node=self)

        # 2. Setup SEIKO Solvers
        urdf_path = os.path.expanduser("~/interbotix_ws/src/seiko_stacking/urdf/vx300s_pure.urdf")
        self.solver_L = SeikoSolver(urdf_path)
        self.solver_R = SeikoSolver(urdf_path)
        self.solver_L.velocity_limit = VEL_LIMIT; self.solver_L.step_size = 0.5
        self.solver_R.velocity_limit = VEL_LIMIT; self.solver_R.step_size = 0.5
        self.filter_L = LowPassFilter(FILTER_GAIN); self.filter_R = LowPassFilter(FILTER_GAIN)
        self.q_neutral = pin.neutral(self.solver_L.model)
        
        # 3. State Tracking
        self.q_L = None; self.raw_L = None; self.ready_L = False
        self.q_R = None; self.raw_R = None; self.ready_R = False
        self.last_grip_state = "NONE"

        # 4. ROS Setup (Added Metric Publishers for rqt_plot)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_L = self.create_publisher(JointGroupCommand, '/arm_1/commands/joint_group', 10)
        self.pub_R = self.create_publisher(JointGroupCommand, '/arm_2/commands/joint_group', 10)
        self.sub_L = self.create_subscription(JointState, '/arm_1/joint_states', self.cb_left, qos)
        self.sub_R = self.create_subscription(JointState, '/arm_2/joint_states', self.cb_right, qos)

        # Metrics Publishers
        self.pub_err_L = self.create_publisher(Float32, '/seiko/left/error', 10)
        self.pub_err_R = self.create_publisher(Float32, '/seiko/right/error', 10)
        self.pub_eff_L = self.create_publisher(Float32, '/seiko/left/efficiency', 10)
        self.pub_eff_R = self.create_publisher(Float32, '/seiko/right/efficiency', 10)
        self.pub_man_L = self.create_publisher(Float32, '/seiko/left/manipulability', 10)
        self.pub_man_R = self.create_publisher(Float32, '/seiko/right/manipulability', 10)

        # 5. CSV Setup
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(['time', 'err_L', 'err_R', 'manip_L', 'manip_R', 'eff_L', 'eff_R'])

        self.timer = self.create_timer(0.05, self.control_loop) 
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        print(" INITIALIZING: Grippers Opening...")
        self.open_grippers()
        time.sleep(1.0)
        print(f" SEIKO SYNC READY. \n   > Live Plot: rqt_plot /seiko/left/error\n   > Data File: {LOG_FILE}")

    # --- CALLBACKS ---
    def cb_left(self, msg):
        n = self.solver_L.nv; raw = np.zeros(n)
        if len(msg.position) >= 6: raw[:6] = msg.position[:6]
        self.q_L = pin.integrate(self.solver_L.model, self.q_neutral, raw)
        self.raw_L = raw; self.ready_L = True

    def cb_right(self, msg):
        n = self.solver_R.nv; raw = np.zeros(n)
        if len(msg.position) >= 6: raw[:6] = msg.position[:6]
        self.q_R = pin.integrate(self.solver_R.model, self.q_neutral, raw)
        self.raw_R = raw; self.ready_R = True

    # --- GRIPPERS ---
    def open_grippers(self):
        t1 = threading.Thread(target=self.arm_L.gripper.release)
        t2 = threading.Thread(target=self.arm_R.gripper.release)
        t1.start(); t2.start()

    def close_grippers(self):
        t1 = threading.Thread(target=self.arm_L.gripper.grasp)
        t2 = threading.Thread(target=self.arm_R.gripper.grasp)
        t1.start(); t2.start()

    # --- MATH ---
    def calculate_metrics(self, solver, q, dq, target):
        pin.forwardKinematics(solver.model, solver.data, q)
        pin.updateFramePlacements(solver.model, solver.data)
        
        J = pin.getFrameJacobian(solver.model, solver.data, solver.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        manip = np.sqrt(np.linalg.det(J @ J.T))
        
        curr_pos = solver.data.oMf[solver.ee_frame_id].translation
        error = np.linalg.norm(target - curr_pos)
        
        v_task = np.linalg.norm((J @ dq)[:3])
        v_joint = np.linalg.norm(dq[:DRIVER_DOF])
        eff = 0.0 if v_joint < 1e-4 else min((v_task / v_joint) * 100.0, 100.0)
        return error, manip, eff

    def solve_frame(self, solver, q, raw, target, pub, filt):
        dq = solver.solve(q, target)
        if dq is None or np.isnan(dq).any(): return 0.0, 0.0, 0.0
        
        vmax = np.max(np.abs(dq))
        if vmax > 1.0: dq *= 1.0 / vmax
        dq = filt.update(dq)
        
        err, manip, eff = self.calculate_metrics(solver, q, dq, target)
        
        cmd = (raw + dq)[:DRIVER_DOF]
        msg = JointGroupCommand(); msg.name = 'arm'; msg.cmd = cmd.tolist()
        pub.publish(msg)
        return err, manip, eff

    # --- LOOP ---
    def control_loop(self):
        t = (self.get_clock().now().nanoseconds / 1e9) - self.start_time
        CYCLE = 15.0; phase = t % CYCLE
        
        SAFE_BUFFER = 0.02
        PICK_L = np.array([OBJ_X, OBJ_Y_L - SAFE_BUFFER, OBJ_Z])
        PICK_R = np.array([OBJ_X, OBJ_Y_R + SAFE_BUFFER, OBJ_Z])
        PLACE_L = np.array([PLACE_X, PLACE_Y_L - SAFE_BUFFER, PLACE_Z])
        PLACE_R = np.array([PLACE_X, PLACE_Y_R + SAFE_BUFFER, PLACE_Z])
        HOME_L = np.array([0.3, 0.20, 0.25]); HOME_R = np.array([0.3, -0.20, 0.25])

        tgt_L = HOME_L; tgt_R = HOME_R
        grip_cmd = "OPEN"; state_name = "WAIT"

        if phase < 1.0: state_name = "INIT"
        elif phase < 4.0: state_name = "APPROACH"; tgt_L = PICK_L + [0, 0, 0.10]; tgt_R = PICK_R + [0, 0, 0.10]
        elif phase < 6.0: state_name = "DIVE"; tgt_L = PICK_L; tgt_R = PICK_R
        elif phase < 7.0: state_name = "GRASP"; tgt_L = PICK_L; tgt_R = PICK_R; grip_cmd = "CLOSE"
        elif phase < 9.0: state_name = "LIFT"; tgt_L = PICK_L + [0, 0, 0.15]; tgt_R = PICK_R + [0, 0, 0.15]; grip_cmd = "CLOSE"
        elif phase < 11.0: state_name = "MOVE"; tgt_L = PLACE_L + [0, 0, 0.15]; tgt_R = PLACE_R + [0, 0, 0.15]; grip_cmd = "CLOSE"
        elif phase < 13.0: state_name = "LOWER"; tgt_L = PLACE_L; tgt_R = PLACE_R; grip_cmd = "CLOSE"
        elif phase < 14.0: state_name = "RELEASE"; tgt_L = PLACE_L; tgt_R = PLACE_R; grip_cmd = "OPEN"
        else: state_name = "RETRACT"; tgt_L = HOME_L; tgt_R = HOME_R

        if grip_cmd != self.last_grip_state:
            if grip_cmd == "OPEN": self.open_grippers()
            else: self.close_grippers()
            self.last_grip_state = grip_cmd

        stats_L = (0,0,0); stats_R = (0,0,0) # err, manip, eff
        
        if self.ready_L: stats_L = self.solve_frame(self.solver_L, self.q_L, self.raw_L, tgt_L, self.pub_L, self.filter_L)
        if self.ready_R: stats_R = self.solve_frame(self.solver_R, self.q_R, self.raw_R, tgt_R, self.pub_R, self.filter_R)

        # 1. WRITE CSV (Legacy)
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([phase, stats_L[0], stats_R[0], stats_L[1], stats_R[1], stats_L[2], stats_R[2]])
            
        # 2. PUBLISH LIVE (For rqt_plot)
        self.pub_err_L.publish(Float32(data=float(stats_L[0])))
        self.pub_err_R.publish(Float32(data=float(stats_R[0])))
        self.pub_man_L.publish(Float32(data=float(stats_L[1])))
        self.pub_man_R.publish(Float32(data=float(stats_R[1])))
        self.pub_eff_L.publish(Float32(data=float(stats_L[2])))
        self.pub_eff_R.publish(Float32(data=float(stats_R[2])))

        if int(t * 100) % 10 == 0:
            print(f"[{state_name}] T:{phase:.1f}s | Err: {stats_L[0]*1000:.1f}mm | Manip: {stats_L[1]:.3f}")

def main():
    rclpy.init()
    try: rclpy.spin(SeikoBimanualSync())
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__':
    main()
