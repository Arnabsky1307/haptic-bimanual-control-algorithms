import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand

class HapticTeleop(Node):
    def __init__(self):
        super().__init__('haptic_teleop_node')
        
        # Scaling factors
        self.HAPTIC_GAIN = 0.5  # Adjust this to feel more or less resistance
        self.EFFORT_THRESHOLD = 500.0 # 500g threshold (tuning required)
        
        # Publishers
        self.slave_pos_pub = self.create_publisher(JointGroupCommand, '/slave/commands/joint_group', 10)
        self.master_eff_pub = self.create_publisher(JointGroupCommand, '/master/commands/joint_group', 10)
        self.slave_grip_pub = self.create_publisher(JointSingleCommand, '/slave/commands/joint_single', 10)

        # Subscribers
        self.create_subscription(JointState, '/master/joint_states', self.master_cb, 10)
        self.create_subscription(JointState, '/slave/joint_states', self.slave_cb, 10)

        self.last_slave_effort = [0.0] * 6

    def slave_cb(self, msg):
        # Store effort from the VX300s (lifting joints: shoulder, elbow)
        # Note: Indexing depends on your specific Dynamixel ID setup
        self.last_slave_effort = msg.effort
        
        # Logic for 500g Error
        total_lift_effort = abs(msg.effort[1]) + abs(msg.effort[2])
        if total_lift_effort > self.EFFORT_THRESHOLD:
            self.get_logger().error(f"OVERLOAD: {total_lift_effort:.2f} effort. Object exceeds 500g limit!")

    def master_cb(self, msg):
        # 1. SEND POSITION TO SLAVE
        pos_cmd = JointGroupCommand()
        pos_cmd.name = 'arm'
        pos_cmd.cmd = msg.position[:6]
        self.slave_pos_pub.publish(pos_cmd)

        # 2. SEND GRIPPER POSITION
        grip_cmd = JointSingleCommand(name='gripper', cmd=msg.position[6])
        self.slave_grip_pub.publish(grip_cmd)

        # 3. HAPTIC FEEDBACK: SEND EFFORT BACK TO MASTER
        # We take the effort the slave is feeling and command the master 
        # motors to apply that same torque in the opposite direction.
        haptic_cmd = JointGroupCommand()
        haptic_cmd.name = 'arm'
        # Reverse the effort to create resistance
        haptic_cmd.cmd = [-eff * self.HAPTIC_GAIN for eff in self.last_slave_effort[:6]]
        self.master_eff_pub.publish(haptic_cmd)

def main():
    rclpy.init()
    node = HapticTeleop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
