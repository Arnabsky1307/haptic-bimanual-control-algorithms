#!/usr/bin/env python3

import numpy as np
import rclpy
import math
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from omni_msgs.msg import OmniButtonEvent, OmniFeedback
from std_msgs.msg import Float64MultiArray
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from scipy.spatial.transform import Rotation as R

# --- WRITING CONFIGURATION ---
ROBOT_MODEL = 'vx300s'

# SCALING (1.2 means 1cm hand move = 1.2cm robot move)
SCALE_POS = 1.2      

# TIMING
LOOP_RATE = 0.025    # 40Hz
MOVING_TIME = 0.15   
ACCEL_TIME = 0.05    

# HAPTICS
STIFFNESS = 20.0     
MAX_FORCE = 3.0      
FORCE_DEADBAND = 0.015 

# WORKSPACE OFFSET (LOWERED Z to 0.10 so it can reach the desk)
# X=0.40 (Forward), Y=0.0 (Center), Z=0.10 (Lower Height)
T_OFFSET = np.array([0.40, 0.0, 0.10])

# --- 1€ FILTER CLASS ---
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = np.zeros_like(x0)
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev
        
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class WritingTeleop(Node):
    def __init__(self, bot):
        super().__init__('writing_teleop')
        self.bot = bot
        
        self.force_pub = self.create_publisher(OmniFeedback, '/phantom/force_feedback', 10)
        self.create_subscription(PoseStamped, '/phantom/pose', self.phantom_pose_cb, 10)
        self.create_subscription(OmniButtonEvent, '/phantom/button', self.button_cb, 10)

        # State Variables
        self.target_pos = T_OFFSET.copy()
        self.target_rpy = np.zeros(3)
        self.actual_pos = np.zeros(3)
        self.gripper_closed = True 
        
        # Debouncing for Button
        self.last_toggle_time = 0.0
        self.white_btn_prev = 0

        # Filter Init
        t0 = self.get_clock().now().nanoseconds / 1e9
        self.pos_filter = OneEuroFilter(t0, self.target_pos, min_cutoff=0.8, beta=0.02)
        
        self.timer = self.create_timer(LOOP_RATE, self.control_loop)
        self.get_logger().info("✍️ TELE-WRITING ACTIVE. Z-Height Lowered. Button Fixed.")

    def button_cb(self, msg: OmniButtonEvent):
        # Current Time
        now = self.get_clock().now().nanoseconds / 1e9

        # --- WHITE BUTTON: TOGGLE GRIPPER (With Debounce) ---
        # Only toggle if button is pressed (1), was previously released (0),
        # AND 0.5 seconds have passed since last toggle.
        if msg.white_button == 1 and self.white_btn_prev == 0:
            if (now - self.last_toggle_time) > 0.5:
                if self.gripper_closed:
                    self.bot.gripper.release(delay=0.0)
                    self.gripper_closed = False
                    self.get_logger().info("Gripper OPEN")
                else:
                    self.bot.gripper.grasp(delay=0.0)
                    self.gripper_closed = True
                    self.get_logger().info("Gripper CLOSED")
                
                self.last_toggle_time = now # Reset cooldown timer
        
        self.white_btn_prev = msg.white_button
        
        # --- GREY BUTTON: RE-CENTER ---
        if msg.grey_button == 1:
            self.bot.arm.go_to_home_pose()
            t_now = self.get_clock().now().nanoseconds / 1e9
            self.pos_filter.x_prev = T_OFFSET.copy()

    def phantom_pose_cb(self, msg: PoseStamped):
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z

        rob_x = (-pz * SCALE_POS) + T_OFFSET[0]
        rob_y = (-px * SCALE_POS) + T_OFFSET[1]
        rob_z = (py * SCALE_POS)  + T_OFFSET[2]
        
        self.target_pos = np.array([rob_x, rob_y, rob_z])

        q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        r = R.from_quat(q)
        rpy = r.as_euler('xyz', degrees=False)
        
        # Allow Pen Pitch (Tilt)
        self.target_rpy = np.array([0.0, rpy[1], 0.0])

    def control_loop(self):
        t_now = self.get_clock().now().nanoseconds / 1e9
        
        # 1. Filter Targets
        smooth_pos = self.pos_filter(t_now, self.target_pos)

        # 2. Get Actual Position
        T_rob = self.bot.arm.get_ee_pose()
        self.actual_pos[0] = T_rob[0, 3]
        self.actual_pos[1] = T_rob[1, 3]
        self.actual_pos[2] = T_rob[2, 3]
        
        # 3. Send Command
        self.bot.arm.set_ee_pose_components(
            x=smooth_pos[0],
            y=smooth_pos[1],
            z=smooth_pos[2],
            roll=0.0,
            pitch=self.target_rpy[1],
            yaw=0.0,
            moving_time=MOVING_TIME,
            accel_time=ACCEL_TIME,
            blocking=False
        )

        # 4. Haptic Feedback
        error_vec = self.actual_pos - smooth_pos
        dist = np.linalg.norm(error_vec)
        fx, fy, fz = 0.0, 0.0, 0.0

        if dist > FORCE_DEADBAND:
            effective_error = error_vec * ((dist - FORCE_DEADBAND) / dist)
            fx = -effective_error[1] * STIFFNESS
            fy = effective_error[2] * STIFFNESS
            fz = -effective_error[0] * STIFFNESS
            
            total = np.sqrt(fx**2 + fy**2 + fz**2)
            if total > MAX_FORCE:
                s = MAX_FORCE / total
                fx *= s; fy *= s; fz *= s
            
            # Print Force Feedback
            if total > 0.1:
                print(f"✍️  CONTACT! Force: {total:.2f} N | Z-Push: {abs(fy):.2f} N", end='\r')

        fb_msg = OmniFeedback()
        fb_msg.force.x, fb_msg.force.y, fb_msg.force.z = float(fx), float(fy), float(fz)
        self.force_pub.publish(fb_msg)

def main(args=None):
    rclpy.init(args=args)
    bot = InterbotixManipulatorXS(robot_model=ROBOT_MODEL, group_name='arm', gripper_name='gripper')
    
    # Start neutral but LOWER (0.15 instead of 0.35) so it's ready to write
    bot.arm.set_ee_pose_components(x=T_OFFSET[0], y=0, z=0.15, pitch=0, roll=0, moving_time=2.0)
    
    node = WritingTeleop(bot)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            robot_shutdown()
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()
