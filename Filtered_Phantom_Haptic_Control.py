import numpy as np
import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from omni_msgs.msg import OmniButtonEvent, OmniFeedback
from sensor_msgs.msg import JointState
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

# --- CONFIGURATION ---
ROBOT_MODEL = 'vx300s'

# SCALING (Sensitivity)
# Higher Y helps the shoulder rotate more easily
SCALE_X = 1.8   
SCALE_Y = 2.4   
SCALE_Z = 1.8   

# TIMING (The Secret to Smoothness)
# We update the loop at 50Hz (0.02s). 
# We tell the robot to finish the move in 0.1s. 
# This slight overlap creates a buttery smooth buffer.
LOOP_RATE = 0.02
MOVING_TIME = 0.1
ACCEL_TIME = 0.05

# HAPTICS
STIFFNESS = 18.0        
MAX_FORCE = 3.0         
FORCE_DEADBAND = 0.02   # 2cm deadband

# WORKSPACE CENTER
X_OFFSET = 0.40  
Y_OFFSET = 0.0
Z_OFFSET = 0.25

# --- 1€ FILTER CLASS (Industry Standard Smoothing) ---
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
        if t_e <= 0: return self.x_prev # Prevent divide by zero if time didn't change

        # Calculate speed (dx)
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # Calculate adaptive cutoff frequency
        # The faster you move, the higher the cutoff (less filtering, less lag)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        
        # Filter the position
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Save for next frame
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class PrecisionTeleopNode(Node):
    def __init__(self, bot):
        super().__init__('precision_teleop_node')
        self.bot = bot
        
        self.force_pub = self.create_publisher(OmniFeedback, '/phantom/force_feedback', 10)
        self.create_subscription(PoseStamped, '/phantom/pose', self.phantom_pose_cb, 10)
        self.create_subscription(OmniButtonEvent, '/phantom/button', self.button_cb, 10)

        # State Variables
        self.raw_target = np.array([X_OFFSET, Y_OFFSET, Z_OFFSET])
        self.actual_pos = np.zeros(3)
        self.gripper_closed = False

        # Initialize One Euro Filter
        # min_cutoff: Lower = smoother slow movements (try 0.1 to 1.0)
        # beta: Higher = less lag during fast movements (try 0.001 to 0.1)
        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.filter = OneEuroFilter(now_sec, self.raw_target, min_cutoff=0.5, beta=0.05)

        # Control Loop
        self.timer = self.create_timer(LOOP_RATE, self.control_loop)
        self.get_logger().info("PRECISION CONTROL ACTIVE (1 Euro Filter + Synced Timing)")

    def button_cb(self, msg: OmniButtonEvent):
        if msg.grey_button == 1:
            if not self.gripper_closed:
                self.bot.gripper.grasp(delay=0.0)
                self.gripper_closed = True
        else:
            if self.gripper_closed:
                self.bot.gripper.release(delay=0.0)
                self.gripper_closed = False
        
        if msg.white_button == 1:
            self.bot.arm.go_to_home_pose()
            # Reset filter target
            now_sec = self.get_clock().now().nanoseconds / 1e9
            self.filter.x_prev = np.array([X_OFFSET, Y_OFFSET, Z_OFFSET])

    def phantom_pose_cb(self, msg: PoseStamped):
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z

        # Mapping & Scaling
        rob_x = (-pz * SCALE_X) + X_OFFSET
        rob_y = (-px * SCALE_Y) + Y_OFFSET
        rob_z = (py * SCALE_Z)  + Z_OFFSET

        self.raw_target = np.array([rob_x, rob_y, rob_z])

    def control_loop(self):
        # 1. Update Filter
        now_sec = self.get_clock().now().nanoseconds / 1e9
        filtered_pos = self.filter(now_sec, self.raw_target)

        # 2. Get Actual Robot Position
        T_rob = self.bot.arm.get_ee_pose()
        self.actual_pos[0] = T_rob[0, 3]
        self.actual_pos[1] = T_rob[1, 3]
        self.actual_pos[2] = T_rob[2, 3]

        # 3. Send Synced Command
        # CRITICAL: We set moving_time to force the robot to keep up.
        self.bot.arm.set_ee_pose_components(
            x=filtered_pos[0],
            y=filtered_pos[1],
            z=filtered_pos[2],
            roll=0.0,
            pitch=0.0,
            moving_time=MOVING_TIME,  # <--- The secret to reducing lag
            accel_time=ACCEL_TIME,
            blocking=False
        )

        # 4. Haptic Feedback
        error_vec = self.actual_pos - filtered_pos
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
        
        msg = OmniFeedback()
        msg.force.x = float(fx)
        msg.force.y = float(fy)
        msg.force.z = float(fz)
        self.force_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    bot = InterbotixManipulatorXS(robot_model=ROBOT_MODEL, group_name='arm', gripper_name='gripper')
    
    bot.arm.set_ee_pose_components(x=X_OFFSET, y=0, z=Z_OFFSET, pitch=0, roll=0, moving_time=1.5)
    
    node = PrecisionTeleopNode(bot)
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
