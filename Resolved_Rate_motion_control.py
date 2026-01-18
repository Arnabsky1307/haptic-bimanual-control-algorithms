import sys
import time
import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

# --- Configuration ---
ROBOT_MODEL = 'vx300s'       # 9-motor 6-DOF model
LEFT_NAME   = 'left_arm'
RIGHT_NAME  = 'right_arm'

# Circle Parameters
RADIUS      = 0.07
PERIOD      = 12.0
DT          = 0.05
DURATION    = 35.0

# --- Data Logging Lists ---
log_time = []
log_joints_left = []      # Joint angles
log_manipulability = []   # Dexterity
log_gripper_force = []    # Gripper Effort (Force Feedback)

def get_circle_velocity_vector(t, radius, period):
    w = (2 * np.pi) / period
    v_angular = [0, 0, 0]
    # Circle in Y-Z plane
    v_linear  = [0, -radius * w * np.sin(w * t), radius * w * np.cos(w * t)]
    return np.array(v_angular + v_linear)

def get_gripper_effort(bot):
    """
    Reads the current load (effort) on the gripper motor.
    Returns value in mA (milli-amps) or raw units depending on firmware.
    """
    try:
        # We look for the joint named 'gripper' in the latest state message
        # Note: Depending on namespace, it might be 'vx300_left/gripper' or just 'gripper'
        # We search for the first joint containing 'gripper'
        joint_names = bot.core.joint_states.name
        gripper_idx = -1
        
        for i, name in enumerate(joint_names):
            if 'gripper' in name:
                gripper_idx = i
                break
        
        if gripper_idx != -1:
            return bot.core.joint_states.effort[gripper_idx]
        return 0.0
    except:
        return 0.0

def solve_jacobian_step(bot, t, radius, period, log_data=False):
    current_joints = bot.arm.get_joint_commands()
    
    # 1. Calculate Jacobian
    J_space = mr.JacobianSpace(bot.arm.robot_des.Slist, current_joints)

    # 2. Calculate Manipulability
    jj_t = np.dot(J_space, J_space.T)
    mu = np.sqrt(np.linalg.det(jj_t))

    # 3. Solve for Velocity
    V_des = get_circle_velocity_vector(t, radius, period)
    J_pinv = np.linalg.pinv(J_space)
    q_dot = np.dot(J_pinv, V_des)

    # 4. Integrate
    next_joints = current_joints + (q_dot * DT)

    # 5. Log Data (Left arm only)
    if log_data:
        log_joints_left.append(current_joints)
        log_manipulability.append(mu)
        # Log Gripper Force
        force = get_gripper_effort(bot)
        log_gripper_force.append(force)

    return next_joints, mu

def plot_results():
    """Generates 3 Graphs: Smoothness, Dexterity, and Force."""
    print("Generating Analysis Graphs...")
    
    joints_arr = np.array(log_joints_left)
    
    # Create 3 subplots now
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Bimanual Task2 Analysis", fontsize=17)
    # Plot 1: Joint Angles
    labels = ['Waist', 'Shoulder', 'Elbow', 'Forearm']
    for i in range(4):
        ax1.plot(log_time, joints_arr[:, i], label=labels[i])
    ax1.set_title("1. Motion Smoothness (Joint Angles)")
    ax1.set_ylabel("Angle (rad)")
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize='small')

    # Plot 2: Manipulability
    ax2.plot(log_time, log_manipulability, color='green', linewidth=2)
    ax2.set_title("2. Dexterity (Manipulability Index)")
    ax2.set_ylabel("Index")
    ax2.axhline(y=0.01, color='r', linestyle='--', label='Singularity Limit')
    ax2.grid(True)
    ax2.legend(loc="upper right")

    # Plot 3: Gripper Force (Current)
    ax3.plot(log_time, log_gripper_force, color='purple', linewidth=1.5)
    ax3.set_title("3. Grip Feedback (Motor Effort)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Current (mA)")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    print("--- Bimanual Control: Grasp & Circle Task ---")
    
    try:
        # Initialize
        bot_left = InterbotixManipulatorXS(
            robot_model=ROBOT_MODEL, group_name='arm', gripper_name='gripper',
            robot_name=LEFT_NAME
        )
        bot_right = InterbotixManipulatorXS(
            robot_model=ROBOT_MODEL, group_name='arm', gripper_name='gripper',
            robot_name=RIGHT_NAME, node=bot_left.core.robot_node
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        # --- PHASE 1: PREPARE TO GRASP ---
        print("1. Opening Grippers...")
        bot_left.gripper.open(delay=1.0)
        bot_right.gripper.open(delay=1.0)

        print("2. Moving to Capture Position...")
        # Start positions
        bot_left.arm.set_ee_pose_components(x=0.4, y=0.2, z=0.3, pitch=0, moving_time=2.5, blocking=False)
        bot_right.arm.set_ee_pose_components(x=0.4, y=-0.2, z=0.3, pitch=0, moving_time=2.5, blocking=True)
        time.sleep(0.5)

        # --- PHASE 2: GRASP OBJECT ---
        print("3. Grasping Object (Closing)...")
        # Set pressure (0.0 to 1.0). 0.6 is a firm grip.
        bot_left.gripper.set_pressure(0.6)
        bot_right.gripper.set_pressure(0.6)
        
        # Close until object is hit
        bot_left.gripper.grasp(delay=0)   # Non-blocking for sync
        bot_right.gripper.grasp(delay=1.0) # Blocking to wait for both

        print("   -> Object Captured. Holding force applied.")
        time.sleep(1.0)

        # --- PHASE 3: EXECUTE TASK ---
        print(f"4. Starting Trajectory Loop ({DURATION}s)...")
        start_time = time.time()
        
        while (time.time() - start_time) < DURATION:
            t_now = time.time()
            t_loop = t_now - start_time
            
            # Left Arm (Logs data)
            next_q_left, mu = solve_jacobian_step(bot_left, t_loop, RADIUS, PERIOD, log_data=True)
            
            # Right Arm
            next_q_right, _ = solve_jacobian_step(bot_right, t_loop, -RADIUS, PERIOD, log_data=False)
            
            log_time.append(t_loop)

            # Console Feedback
            sys.stdout.write(f"\r[T: {t_loop:.2f}s] Dexterity: {mu:.4f} | Grip Force: {log_gripper_force[-1]:.1f}mA")
            sys.stdout.flush()

            # Execute Move
            bot_left.arm.set_joint_positions(next_q_left, blocking=False)
            bot_right.arm.set_joint_positions(next_q_right, blocking=False)
            
            # Sleep remainder of cycle
            elapsed = time.time() - t_now
            time.sleep(max(0, DT - elapsed))

    except KeyboardInterrupt:
        print("\n\nUser Stopped Operation.")

    finally:
        # --- PHASE 4: CLEANUP & SLEEP ---
        print("\n\n--- Task Complete. Shutting Down ---")
        
        print("1. Releasing Object...")
        bot_left.gripper.open(delay=0.1)
        bot_right.gripper.open(delay=1.0)
        
        print("2. Going to Sleep Pose...")
        # Move safely to sleep
        bot_left.arm.go_to_sleep_pose(moving_time=3.0, blocking=False)
        bot_right.arm.go_to_sleep_pose(moving_time=3.0, blocking=True)
        
        print("Done.")
        
        # Plot only if we gathered data
        if len(log_time) > 0:
            plot_results()

if __name__ == '__main__':
    main()
