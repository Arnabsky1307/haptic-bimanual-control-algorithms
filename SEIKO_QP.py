import numpy as np
import modern_robotics as mr
import time
import rclpy
from scipy.sparse import csc_matrix
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import create_interbotix_global_node
from qpsolvers import solve_qp

class SEIKOMonitoringLoop:
    def __init__(self, d_base=0.4):
        # 1. Initialize ROS 2 and create the SINGLE shared global node
        if not rclpy.ok():
            rclpy.init()
        
        # Shared node is essential for bimanual control to avoid InterbotixException
        self.node = create_interbotix_global_node()

        # 2. Integrate the SHARED node for both arm instances
        # This prevents the 'Tried to create an Interbotix global node but one already exists' error
        self.bot_l = InterbotixManipulatorXS(
            robot_model="vx300s", 
            robot_name="left_arm", 
            node=self.node
        )
        self.bot_r = InterbotixManipulatorXS(
            robot_model="vx300s", 
            robot_name="right_arm", 
            node=self.node
        )
        
        # 3. Kinematic Constants (VX300s)
        self.L1, self.L2, self.L3, self.L4 = 0.11035, 0.300, 0.300, 0.17415
        self.Slist = np.array([
            [0, 0, 1, 0, 0, 0], [0, 1, 0, -self.L1, 0, 0],
            [0, 1, 0, -(self.L1 + self.L2), 0, 0], [1, 0, 0, 0, (self.L1 + self.L2), 0],
            [0, 1, 0, -(self.L1 + self.L2 + self.L3), 0, 0], [1, 0, 0, 0, (self.L1 + self.L2 + self.L3), 0]
        ]).T

        # World-frame base offsets
        self.T_w_Lbase = mr.RpToTrans(np.eye(3), [0,  d_base/2, 0])
        self.T_w_Rbase = mr.RpToTrans(np.eye(3), [0, -d_base/2, 0])

        # 4. Constraints and Weights
        self.q_min = np.array([-3.1, -1.8, -1.8, -3.1, -1.8, -3.1] * 2)
        self.q_max = np.array([3.1, 1.8, 1.8, 3.1, 1.8, 3.1] * 2)
        self.dt = 0.05
        self.weight_task = 100.0
        self.weight_reg = 0.1

    def execute_and_monitor(self):
        print("Moving to Home...")
        self.bot_l.arm.go_to_home_pose()
        self.bot_r.arm.go_to_home_pose()
        time.sleep(2.0)

        print("\n--- Starting Real-Time SEIKO Full Jacobian Monitor ---")
        for i in range(100):
            # A. Get Live Hardware Data
            ql = np.array(self.bot_l.arm.get_joint_positions())
            qr = np.array(self.bot_r.arm.get_joint_positions())
            q_curr = np.concatenate([ql, qr])

            # B. Compute Unified Jacobian in World Frame
            Jw_L = mr.Adjoint(self.T_w_Lbase) @ mr.JacobianSpace(self.Slist, ql)
            Jw_R = mr.Adjoint(self.T_w_Rbase) @ mr.JacobianSpace(self.Slist, qr)

            # REAL-TIME MONITOR: Print both 6x6 matrices for every iteration
            print(f"\n--- Iteration {i} ---")
            print("Left Arm 6x6 Jacobian:\n", np.round(Jw_L, 3))
            print("Right Arm 6x6 Jacobian:\n", np.round(Jw_R, 3))

            # C. SEIKO QP Formulation
            # Form Absolute and Relative Tasks
            J_abs = 0.5 * (np.hstack([Jw_L, Jw_R])) 
            J_rel = np.hstack([Jw_L, -Jw_R])
            A_eq_stack = np.vstack([J_abs, J_rel])
            
            b_eq = np.zeros(12)
            b_eq[5] = 0.03 # Target: 3cm/s upwards

            # Reformulate into Quadratic Cost: Min 1/2*x^T*P*x + c^T*x
            P = csc_matrix((self.weight_reg * np.eye(12)) + (self.weight_task * (A_eq_stack.T @ A_eq_stack)))
            c = -(b_eq.T @ (self.weight_task * A_eq_stack))
            
            # Inequality: Gx <= h (Joint Limits)
            G = csc_matrix(np.vstack([np.eye(12), -np.eye(12)]))
            h = np.concatenate([(self.q_max - q_curr)/self.dt, (q_curr - self.q_min)/self.dt])

            # D. Solve using OSQP
            dq = solve_qp(P, c, G, h, solver="osqp")
            
            if dq is not None:
                q_next = q_curr + dq * self.dt
                # Non-blocking simultaneous commands
                self.bot_l.arm.set_joint_positions(q_next[0:6].tolist(), blocking=False)
                self.bot_r.arm.set_joint_positions(q_next[6:12].tolist(), blocking=False)
            
            time.sleep(self.dt)

        self.bot_l.arm.go_to_sleep_pose()
        self.bot_r.arm.go_to_sleep_pose()
        rclpy.shutdown()

if __name__ == "__main__":
    try:
        ctrl = SEIKOMonitoringLoop()
        ctrl.execute_and_monitor()
    except KeyboardInterrupt:
        if rclpy.ok():
            rclpy.shutdown()
