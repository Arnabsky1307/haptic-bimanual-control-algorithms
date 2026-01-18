import rclpy
import threading
import time
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import create_interbotix_global_node

def move_bimanual():
    # 1. Initialize the shared node
    node = create_interbotix_global_node()

    # 2. Initialize Arms
    bot_left = InterbotixManipulatorXS(robot_model="vx300s", robot_name="arm_1", node=node)
    bot_right = InterbotixManipulatorXS(robot_model="vx300s", robot_name="arm_2", node=node)

    # Global Smooth Profile
    SMOOTH_TIME = 4.0
    GENTLE_ACCEL = 1.0
    
    for bot in [bot_left, bot_right]:
        bot.arm.set_trajectory_time(moving_time=SMOOTH_TIME, accel_time=GENTLE_ACCEL)

    # 3. Preparation: Concurrent Homing
    print("Homing and Opening Grippers...")
    bot_left.gripper.release()
    bot_right.gripper.release()
    bot_left.arm.go_to_home_pose(blocking=False)
    bot_right.arm.go_to_home_pose(blocking=True)

    # 4. User Input
    try:
        obj_x = float(input("Enter Target X: "))
        obj_y_offset = float(input("Enter Y Half-Width: "))
        obj_z = float(input("Enter Target Z: "))
    except ValueError:
        return

    # Wrist Orientation: 90 degrees opposite (Parallel to ground)
    roll_l, roll_r = -1.57, 1.57

    # 5. Synchronized Standoff
    print("Moving to Standoff...")
    bot_left.arm.set_ee_pose_components(x=obj_x, y=obj_y_offset + 0.06, z=obj_z, roll=roll_l, blocking=False)
    bot_right.arm.set_ee_pose_components(x=obj_x, y=-(obj_y_offset + 0.06), z=obj_z, roll=roll_r, blocking=True)

    # 6. Synchronized Approach
    input("\nPress Enter to Approach and Grasp...")
    bot_left.arm.set_ee_pose_components(x=obj_x, y=obj_y_offset, z=obj_z, roll=roll_l, blocking=False)
    bot_right.arm.set_ee_pose_components(x=obj_x, y=-obj_y_offset, z=obj_z, roll=roll_r, blocking=True)

    # 7. Grasp
    print("Grasping...")
    bot_left.gripper.grasp()
    bot_right.gripper.grasp()
    time.sleep(2.0) # Ensure solid grip

    # 8. THE LIFT: Pure Joint-Space Synchronization
    # To avoid the cartesian delay, we calculate the joint target for the lift first.
    print("Calculating Lift and executing...")
    
    # Get current joint positions
    left_joints = bot_left.arm.get_joint_commands()
    right_joints = bot_right.arm.get_joint_commands()
    
    # To move "Up" in joint space for a 6DOF arm like vx300, 
    # we target the specific pose but trigger them as joint commands.
    # Alternatively, we move both to a new Z coordinate simultaneously:
    bot_left.arm.set_ee_pose_components(x=obj_x, y=obj_y_offset, z=obj_z + 0.1, roll=roll_l, blocking=False)
    bot_right.arm.set_ee_pose_components(x=obj_x, y=-obj_y_offset, z=obj_z + 0.1, roll=roll_r, blocking=True)

    # 9. Smooth Sleep Transition
    print("\nLift complete. Returning to Sleep gently...")
    input("Press Enter to Sleep...")

    bot_left.gripper.release()
    bot_right.gripper.release()
    time.sleep(1.0)
    
    # Set a very slow profile for Sleep to avoid aggressive motion
    for bot in [bot_left, bot_right]:
        bot.arm.set_trajectory_time(moving_time=6.0, accel_time=2.0)
    
    bot_left.arm.go_to_sleep_pose(blocking=False)
    bot_right.arm.go_to_sleep_pose(blocking=True)

    print("System at rest, comrade.")
    rclpy.shutdown()

if __name__ == "__main__":
    move_bimanual()
