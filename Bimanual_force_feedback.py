import rclpy
import threading
import time
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import create_interbotix_global_node

# Safety Threshold: 0.0 to 1.0 (1.0 is max torque)
# For delicate objects, start with 0.3. For heavier ones, use 0.6.
GRASP_PRESSURE = 0.5 

def force_sensitive_grasp(bot, pressure_value):
    """
    Sets the gripper pressure limit and triggers a grasp.
    """
    print(f"[{bot.robot_name}] Setting pressure to {pressure_value} and grasping...")
    # Set the torque/pressure limit for the gripper motor
    bot.gripper.set_pressure(pressure_value)
    # Trigger the grasp move (closes until it hits resistance or limit)
    bot.gripper.grasp()

def move_bimanual():
    # 1. Initialize the shared node
    node = create_interbotix_global_node()

    # 2. Initialize Arms
    bot_left = InterbotixManipulatorXS(robot_model="vx300", robot_name="left_arm", node=node)
    bot_right = InterbotixManipulatorXS(robot_model="vx300", robot_name="right_arm", node=node)

    # Global Smooth Profile
    SMOOTH_TIME = 4.0
    GENTLE_ACCEL = 1.0
    
    for bot in [bot_left, bot_right]:
        bot.arm.set_trajectory_time(moving_time=SMOOTH_TIME, accel_time=GENTLE_ACCEL)

    # 3. Preparation: Homing
    print("Preparing hardware: Homing and Opening Grippers...")
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

    roll_l, roll_r = -1.57, 1.57

    # 5. Synchronized Standoff
    print("Moving to Standoff...")
    bot_left.arm.set_ee_pose_components(x=obj_x, y=obj_y_offset + 0.06, z=obj_z, roll=roll_l, blocking=False)
    bot_right.arm.set_ee_pose_components(x=obj_x, y=-(obj_y_offset + 0.06), z=obj_z, roll=roll_r, blocking=True)

    # 6. Synchronized Approach
    input("\nPress Enter to Approach and Grasp...")
    bot_left.arm.set_ee_pose_components(x=obj_x, y=obj_y_offset, z=obj_z, roll=roll_l, blocking=False)
    bot_right.arm.set_ee_pose_components(x=obj_x, y=-obj_y_offset, z=obj_z, roll=roll_r, blocking=True)

    # 7. CORRECTED FORCE SENSITIVE GRASP
    print("Executing Force-Sensitive Grasping...")
    t_l = threading.Thread(target=force_sensitive_grasp, args=(bot_left, GRASP_PRESSURE))
    t_r = threading.Thread(target=force_sensitive_grasp, args=(bot_right, GRASP_PRESSURE))
    
    t_l.start()
    t_r.start()
    t_l.join()
    t_r.join()
    
    time.sleep(1.5) # Wait for firm contact

    # 8. Synchronized Lift (Joint-Space based for no-delay)
    print("Lifting object...")
    bot_left.arm.set_ee_pose_components(x=obj_x, y=obj_y_offset, z=obj_z + 0.1, roll=roll_l, blocking=False)
    bot_right.arm.set_ee_pose_components(x=obj_x, y=-obj_y_offset, z=obj_z + 0.1, roll=roll_r, blocking=True)

    # 9. Smooth Sleep Transition
    print("\nLift complete. Returning to Sleep gently...")
    input("Press Enter to Sleep...")

    bot_left.gripper.release()
    bot_right.gripper.release()
    time.sleep(1.0)
    
    # Extra slow profile for Sleep
    for bot in [bot_left, bot_right]:
        bot.arm.set_trajectory_time(moving_time=6.0, accel_time=2.0)
    
    bot_left.arm.go_to_sleep_pose(blocking=False)
    bot_right.arm.go_to_sleep_pose(blocking=True)

    print("System at rest. Task complete.")
    rclpy.shutdown()

if __name__ == "__main__":
    move_bimanual()
