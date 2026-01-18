import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from interbotix_xs_msgs.srv import TorqueEnable # Import Torque Service
import numpy as np
import os
import pinocchio as pin
from seiko_stacking.solver import SeikoSolver 

# ==========================================
# 🔧 USER CONFIGURATION
# ==========================================
DRIVER_DOF = 6          
STACK_INC = 0.06        
BASE_DIST = 0.60        

# 🔧 Z-TRIM REFINEMENT
# Was -0.020. Moved to -0.025 to lower Left Arm further.
LEFT_Z_TRIM = -0.025 

# 🔧 GRIPPER MAX SETTINGS
# 0.08 is roughly the physical max for ViperX grippers.
GRIP_OPEN_VAL = 0.08    
GRIP_CLOSE_VAL = 0.026  
# ==========================================

class LowPassFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha; self.prev = None
    def update(self, val):
        if self.prev is None: self.prev = val
        self.prev = self.alpha * val + (1.0 - self.alpha) * self.prev
        return self.prev

class BimanualPickNode(Node):
    def __init__(self):
        super().__init__('seiko_bimanual_pick')
        self.ns_L = 'arm_1'; self.ns_R = 'arm_2'
        
        # --- SOLVERS ---
        urdf_path = os.path.expanduser("~/interbotix_ws/src/seiko_stacking/urdf/vx300s_pure.urdf")
        self.solver_L = SeikoSolver(urdf_path)
        self.solver_R = SeikoSolver(urdf_path)
        
        self.solver_L.velocity_limit = 1.2; self.solver_L.step_size = 0.5
        self.solver_R.velocity_limit = 1.2; self.solver_R.step_size = 0.5
        self.filter_L = LowPassFilter(0.3); self.filter_R = LowPassFilter(0.3)
        
        self.q_neutral = pin.neutral(self.solver_L.model)
        self.q_L = None; self.raw_L = None; self.q_R = None; self.raw_R = None
        self.ready_L = False; self.ready_R = False

        # --- COMMUNICATIONS ---
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_L = self.create_publisher(JointGroupCommand, f'/{self.ns_L}/commands/joint_group', 10)
        self.pub_R = self.create_publisher(JointGroupCommand, f'/{self.ns_R}/commands/joint_group', 10)
        
        # Gripper Publishers
        self.pub_L_grip = self.create_publisher(JointSingleCommand, f'/{self.ns_L}/commands/joint_single', 10)
        self.pub_R_grip = self.create_publisher(JointSingleCommand, f'/{self.ns_R}/commands/joint_single', 10)
        
        self.sub_L = self.create_subscription(JointState, f'/{self.ns_L}/joint_states', self.cb_left, qos)
        self.sub_R = self.create_subscription(JointState, f'/{self.ns_R}/joint_states', self.cb_right, qos)
        
        # Torque Clients (To wake up grippers)
        self.cli_L_torque = self.create_client(TorqueEnable, f'/{self.ns_L}/torque_enable')
        self.cli_R_torque = self.create_client(TorqueEnable, f'/{self.ns_R}/torque_enable')

        self.timer = self.create_timer(0.05, self.control_loop) 
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        # --- COORDINATES ---
        self.Y_OFFSET_L = (BASE_DIST / 2.0)   
        self.Y_OFFSET_R = -1 * (BASE_DIST / 2.0) 
        
        self.OBJ_X = 0.50 
        self.OBJ_Z = 0.08 
        
        self.stack_count = 0
        self.current_stack_z = 0.0
        
        # Force Wakeup
        self.enable_grippers()
        
        self.get_logger().info(f"🏭 SEIKO ADJUSTED. Left Z-Trim: {LEFT_Z_TRIM}m")

    def enable_grippers(self):
        # Asynchronously enable torque on grippers to ensure they move
        req = TorqueEnable.Request()
        req.cmd_type = 'single'
        req.name = 'gripper'
        req.enable = True
        if self.cli_L_torque.service_is_ready():
            self.cli_L_torque.call_async(req)
        if self.cli_R_torque.service_is_ready():
            self.cli_R_torque.call_async(req)

    def cb_left(self, msg): 
        n_ros = len(msg.position); n_sol = self.solver_L.nv
        raw = np.zeros(n_sol)
        if n_ros >= n_sol: raw = np.array(msg.position[:n_sol])
        else: raw[:n_ros] = msg.position
        self.q_L = pin.integrate(self.solver_L.model, self.q_neutral, raw)
        self.raw_L = raw
        self.ready_L = True

    def cb_right(self, msg): 
        n_ros = len(msg.position); n_sol = self.solver_R.nv
        raw = np.zeros(n_sol)
        if n_ros >= n_sol: raw = np.array(msg.position[:n_sol])
        else: raw[:n_ros] = msg.position
        self.q_R = pin.integrate(self.solver_R.model, self.q_neutral, raw)
        self.raw_R = raw
        self.ready_R = True

    def send_gripper(self, side, value):
        # Aggressive Mode: Send every cycle. No filtering.
        if side == 'L': pub = self.pub_L_grip
        else: pub = self.pub_R_grip
            
        msg = JointSingleCommand()
        msg.name = 'gripper'
        msg.cmd = float(value)
        pub.publish(msg)

    def solve_move(self, solver, q, raw, target, pub, filt):
        dq = solver.solve(q, target)
        if dq is None or np.isnan(dq).any(): return 0.0, 0.0
        
        max_v = np.max(np.abs(dq))
        if max_v > 1.0: dq = dq * (1.0/max_v)
        dq = filt.update(dq)
        
        cmd = (raw + dq)[:DRIVER_DOF]
        msg = JointGroupCommand(); msg.name = 'arm'; msg.cmd = cmd.tolist()
        pub.publish(msg)
        
        J = pin.getFrameJacobian(solver.model, solver.data, solver.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        v_task = np.linalg.norm((J @ dq)[:3]); v_joint = np.linalg.norm(dq)
        return 0.0 if v_joint < 1e-4 else min((v_task/v_joint)*100, 100)

    def control_loop(self):
        t = (self.get_clock().now().nanoseconds / 1e9) - self.start_time
        CYCLE_TIME = 14.0
        cycle = t % CYCLE_TIME
        
        cycle_idx = int(t / CYCLE_TIME)
        if cycle_idx > self.stack_count:
            self.stack_count = cycle_idx
            self.current_stack_z = self.stack_count * STACK_INC
            self.get_logger().info(f"📚 STACK LEVEL {self.stack_count}")

        # --- TARGETS ---
        
        # APPLY Z-TRIM TO LEFT ARM ONLY (-0.025m)
        TGT_L_PICK = np.array([self.OBJ_X, self.Y_OFFSET_L, self.OBJ_Z + LEFT_Z_TRIM])
        TGT_R_PICK = np.array([self.OBJ_X, self.Y_OFFSET_R, self.OBJ_Z])
        
        TGT_L_PLACE = np.array([self.OBJ_X, self.Y_OFFSET_L, self.OBJ_Z + self.current_stack_z + LEFT_Z_TRIM])
        TGT_R_PLACE = np.array([self.OBJ_X, self.Y_OFFSET_R, self.OBJ_Z + self.current_stack_z])
        
        HOME_L = np.array([0.3, 0.20, 0.25]) 
        HOME_R = np.array([0.3, -0.20, 0.25])

        tgt_L = HOME_L; tgt_R = HOME_R
        grp_val = GRIP_OPEN_VAL
        status = "WAIT"

        # --- STATE MACHINE ---
        if cycle < 1.0:
            status = "SYNC-INIT"; grp_val = GRIP_OPEN_VAL
        elif cycle < 3.0:
            status = "APPROACH"; 
            tgt_L = TGT_L_PICK + [0, 0, 0.10]
            tgt_R = TGT_R_PICK + [0, 0, 0.10]
            grp_val = GRIP_OPEN_VAL # Sending 0.08m
        elif cycle < 4.5:
            status = "DIVE"; 
            tgt_L = TGT_L_PICK
            tgt_R = TGT_R_PICK
            grp_val = GRIP_OPEN_VAL
        elif cycle < 5.5:
            status = "GRIPPING"; 
            tgt_L = TGT_L_PICK
            tgt_R = TGT_R_PICK
            grp_val = GRIP_CLOSE_VAL # Snap shut
        elif cycle < 7.5:
            status = "LIFT"; 
            tgt_L = TGT_L_PICK + [0, 0, 0.15]
            tgt_R = TGT_R_PICK + [0, 0, 0.15]
            grp_val = GRIP_CLOSE_VAL
        elif cycle < 9.5:
            status = "STACK-MOVE"; 
            tgt_L = TGT_L_PLACE + [0, 0, 0.05]
            tgt_R = TGT_R_PLACE + [0, 0, 0.05]
            grp_val = GRIP_CLOSE_VAL
        elif cycle < 11.0:
            status = "PLACE-DOWN"; 
            tgt_L = TGT_L_PLACE
            tgt_R = TGT_R_PLACE
            grp_val = GRIP_CLOSE_VAL
        elif cycle < 12.0:
            status = "RELEASE"; 
            tgt_L = TGT_L_PLACE
            tgt_R = TGT_R_PLACE
            grp_val = GRIP_OPEN_VAL # Release
        else:
            status = "RETRACT"; 
            tgt_L = HOME_L
            tgt_R = HOME_R
            grp_val = GRIP_OPEN_VAL
            
        # Execute
        eff_L = 0; eff_R = 0
        if self.ready_L:
            eff_L = self.solve_move(self.solver_L, self.q_L, self.raw_L, tgt_L, self.pub_L, self.filter_L)
            self.send_gripper('L', grp_val)
            
        if self.ready_R:
            eff_R = self.solve_move(self.solver_R, self.q_R, self.raw_R, tgt_R, self.pub_R, self.filter_R)
            self.send_gripper('R', grp_val)

        if int(t*100)%50==0: self.get_logger().info(f"{status} | Stack: {self.stack_count} | L:{eff_L:.0f}% R:{eff_R:.0f}%")

def main():
    rclpy.init()
    rclpy.spin(BimanualPickNode())
    rclpy.shutdown()
