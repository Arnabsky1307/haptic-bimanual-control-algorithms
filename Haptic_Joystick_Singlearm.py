import numpy as np
import rclpy
import math
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from omni_msgs.msg import OmniButtonEvent, OmniFeedback
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from scipy.spatial.transform import Rotation as R

# --- ADVANCED CONFIGURATION ---
ROBOT_MODEL = 'vx300s'

# SCALING (Sensitivity)
SCALE_POS = 1.8      # XYZ Motion Scaling
SCALE_ROT = 1.0      # Rotation Scaling (1:1 is best for orientation)

# TIMING (The Heartbeat of Smoothness)
LOOP_RATE = 0.025    # 40Hz Control Loop (25ms)
MOVING_TIME = 0.15   # Robot move time (slightly larger than loop to smooth buffer)
ACCEL_TIME = 0.05    

# HAPTICS
STIFFNESS = 18.0     
MAX_FORCE = 3.0      
FORCE_DEADBAND = 0.02 

# WORKSPACE OFFSET (Center 40cm forward)
T_OFFSET = np.array([0.40, 0.0, 0.25])

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

class UltimateTeleop(Node):
    def __init__(self, bot):
        super().__init__('ultimate_teleop')
        self.bot = bot
        
        # Publishers for Real-Time Plotting
        self.pub_telemetry = self.create_publisher(Float64MultiArray, '/teleop/telemetry', 10)
        self.force_pub = self.create_publisher(OmniFeedback, '/phantom/force_feedback', 10)
        
        # Subscribers
        self.create_subscription(PoseStamped, '/phantom/pose', self.phantom_pose_cb, 10)
        self.create_subscription(OmniButtonEvent, '/phantom/button', self.button_cb, 10)

        # State Variables
        self.target_pos = T_OFFSET.copy()
        self.target_rpy = np.zeros(3)
        self.actual_pos = np.zeros(3)
        self.gripper_closed = False
        self.start_time = time.time()

        # Initialize Filters (Pos + Rot)
        t0 = self.get_clock().now().nanoseconds / 1e9
        self.pos_filter = OneEuroFilter(t0, self.target_pos, min_cutoff=0.5, beta=0.05)
        self.rot_filter = OneEuroFilter(t0, self.target_rpy, min_cutoff=1.0, beta=0.01)

        # Timer
        self.timer = self.create_timer(LOOP_RATE, self.control_loop)
        self.get_logger().info("ULTIMATE 6-DOF CONTROL ACTIVE. Open rqt_plot for graphs.")

    def button_cb(self, msg: OmniButtonEvent):
        # Grey: Gripper
        if msg.grey_button == 1:
            if not self.gripper_closed:
                self.bot.gripper.grasp(delay=0.0)
                self.gripper_closed = True
        else:
            if self.gripper_closed:
                self.bot.gripper.release(delay=0.0)
                self.gripper_closed = False
        
        # White: Re-center
        if msg.white_button == 1:
            self.bot.arm.go_to_home_pose()
            t_now = self.get_clock().now().nanoseconds / 1e9
            self.pos_filter.x_prev = T_OFFSET.copy()
            self.rot_filter.x_prev = np.zeros(3)

    def phantom_pose_cb(self, msg: PoseStamped):
        # 1. Position Mapping (Geomagic -> Robot)
        # Geomagic X (Right) -> Robot -Y
        # Geomagic Y (Up)    -> Robot Z
        # Geomagic Z (Pull)  -> Robot -X
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z

        rob_x = (-pz * SCALE_POS) + T_OFFSET[0]
        rob_y = (-px * SCALE_POS) + T_OFFSET[1]
        rob_z = (py * SCALE_POS)  + T_OFFSET[2]
        
        self.target_pos = np.array([rob_x, rob_y, rob_z])

        # 2. Orientation Mapping (Quaternion -> Euler RPY)
        q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        
        # Convert to Euler
        r = R.from_quat(q)
        rpy = r.as_euler('xyz', degrees=False)
        
        # Mapping Orientation Frames is tricky. 
        # For simplicity, we map Stylus Pitch to Robot Pitch directly
        # and Stylus Roll to Robot Roll. Yaw is often weird on stylus.
        # We apply a coordinate rotation to match the "Forward" facing robot.
        
        # Stylus Roll (Twist) -> Robot Roll (Wrist Rotate)
        # Stylus Pitch (Up/Down) -> Robot Pitch (Wrist Angle)
        # Stylus Yaw (Left/Right) -> Robot Yaw (Waist?) - usually better to ignore yaw for picking
        
        # Experimentally determined safe mapping:
        target_roll = rpy[0] 
        target_pitch = rpy[1] + 0.0 # Offset if needed
        target_yaw = 0.0 # Locking Yaw keeps gripper straight for picking
        
        self.target_rpy = np.array([target_roll, target_pitch, target_yaw])

    def control_loop(self):
        t_now = self.get_clock().now().nanoseconds / 1e9
        
        # 1. Filter Targets
        smooth_pos = self.pos_filter(t_now, self.target_pos)
        smooth_rpy = self.rot_filter(t_now, self.target_rpy)

        # 2. Get Actual Robot State
        T_rob = self.bot.arm.get_ee_pose()
        self.actual_pos[0] = T_rob[0, 3]
        self.actual_pos[1] = T_rob[1, 3]
        self.actual_pos[2] = T_rob[2, 3]
        
        # Get Joint Velocities (for plotting)
        # The Interbotix API stores current joint states in self.bot.arm.core.joint_states
        # We assume the user wants the magnitude or specific joints.
        # Let's verify velocities are available, otherwise use previous pos to calc.
        
        # 3. Send 6-DOF Command
        self.bot.arm.set_ee_pose_components(
            x=smooth_pos[0],
            y=smooth_pos[1],
            z=smooth_pos[2],
            roll=smooth_rpy[0],
            pitch=smooth_rpy[1],
            yaw=smooth_rpy[2],
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

        # Publish Force
        fb_msg = OmniFeedback()
        fb_msg.force.x, fb_msg.force.y, fb_msg.force.z = float(fx), float(fy), float(fz)
        self.force_pub.publish(fb_msg)

        # 5. Publish Telemetry for RQT Plot
        # Format: [TargetX, ActualX, TargetY, ActualY, TargetZ, ActualZ, ErrorMagnitude]
        telem_msg = Float64MultiArray()
        telem_msg.data = [
            smooth_pos[0], self.actual_pos[0], # X
            smooth_pos[1], self.actual_pos[1], # Y
            smooth_pos[2], self.actual_pos[2], # Z
            dist # Total Error
        ]
        self.pub_telemetry.publish(telem_msg)

def main(args=None):
    rclpy.init(args=args)
    bot = InterbotixManipulatorXS(robot_model=ROBOT_MODEL, group_name='arm', gripper_name='gripper')
    
    # Start Neutral
    bot.arm.set_ee_pose_components(x=T_OFFSET[0], y=0, z=T_OFFSET[2], pitch=0, roll=0, moving_time=2.0)
    
    node = UltimateTeleop(bot)
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
