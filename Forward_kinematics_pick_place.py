import sys
import time
import numpy as np
import modern_robotics as mr
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

# --- Configuration ---
ROBOT_MODEL = 'vx300'
GRIPPER_CURRENT_THRESHOLD = 150  # Threshold (mA) - Tune this based on your object
MANIPULABILITY_THRESHOLD = 0.05  # Lower limit for singularity warning

class SmartRobotController:
    def __init__(self):
        # Initialize the Interbotix Robot
        self.bot = InterbotixManipulatorXS(
            robot_model=ROBOT_MODEL,
            group_name='arm',
            gripper_name='gripper'
        )
       
        # Access the robot's kinematic model (Screw axes)
        # The 'Slist' is the Space Screw Axes matrix required for Jacobian calculation
        # This is usually available in the robot_model property of the arm module
        try:
            self.Slist = self.bot.arm.robot_model.Slist
        except AttributeError:
            print("Error: Could not find Slist in robot model. Ensure Interbotix MR descriptions are loaded.")
            sys.exit(1)

    def get_jacobian_health(self):
        """Calculates the Jacobian and returns the Manipulability Measure."""
        # Get current joint positions (read from the real servos)
        joint_positions = self.bot.arm.capture_joint_positions()
       
        # Calculate Space Jacobian using Modern Robotics library
        # J_s(theta) = JacobianSpace(Slist, theta)
        J = mr.JacobianSpace(self.Slist, joint_positions)
       
        # Calculate Manipulability Measure w = sqrt(det(J * J_T))
        # For a 6DOF arm, J is 6x6, so we can just take absolute determinant
        manipulability = np.sqrt(np.linalg.det(np.dot(J, J.T)))
        return J, manipulability

    def get_gripper_feedback(self):
        """Checks if the gripper is holding an object based on motor effort."""
        # The Interbotix API exposes internal core states
        # We need to find the index of the gripper motor in the joint states
        try:
            # joint_states.effort is usually in mA for Dynamixel items in this wrapper
            # Note: You might need to check the specific index for your gripper in the 'joint_states' msg
            # For simplicity, we assume the gripper is the last joint or named explicitly.
            # A more robust way in the API is checking the present_current register directly if available,
            # but we will use the published joint states for speed.
             
            # Sum of efforts if multiple fingers, or just the main gripper motor
            effort = self.bot.core.joint_states.effort[-1]
            return abs(effort)
        except Exception:
            return 0.0

    def monitor_move(self, target_action, description="Moving"):
        """
        Executes a motion non-blockingly and monitors feedback during the move.
        """
        print(f"\n--- {description} ---")
       
        # Execute the lambda function passed (the move command)
        # Crucial: The move command inside the lambda must have blocking=False
        target_action()
       
        # Feedback Loop
        while True:
            # 1. Jacobian Feedback
            J, w = self.get_jacobian_health()
           
            # 2. Gripper Feedback
            current = self.get_gripper_feedback()
           
            # Real-time dashboard print
            status = "HEALTHY" if w > MANIPULABILITY_THRESHOLD else "SINGULARITY RISK"
            sys.stdout.write(f"\r[Status: {status}] Manipulability: {w:.4f} | Gripper Current: {current:.1f} mA")
            sys.stdout.flush()
           
            # Check if the robot has finished moving
            # The Interbotix API tracks this via the 'moving' register or trajectory status
            if self.bot.arm.get_joint_commands() == self.bot.arm.capture_joint_positions():
                # Note: Exact float equality is rare; usually, we check if trajectory is done.
                # A safer check for the API:
                time.sleep(0.1)
                # If using sleep moves, we can't loop.
                # Since we used blocking=False, we check a simple time or distance threshold.
                # For this demo, we break when the error is small
                error = np.linalg.norm(np.array(self.bot.arm.get_joint_commands()) - np.array(self.bot.arm.capture_joint_positions()))
                if error < 0.05: # Radians tolerance
                    break
           
            time.sleep(0.05) # 20Hz update rate
           
        print("\nMove Complete.")

    def run_task(self):
        # 1. Go to Home
        self.monitor_move(lambda: self.bot.arm.go_to_home_pose(blocking=False), "Homing")

        # 2. Open Gripper
        self.bot.gripper.release()

        # 3. Approach Object (Define your coordinates here)
        # Using Relative move for demonstration
        self.monitor_move(lambda: self.bot.arm.set_ee_cartesian_trajectory(x=0.3, z=-0.1, blocking=False), "Approaching Object")

        # 4. Grasp
        print("\n--- Grasping ---")
        self.bot.gripper.grasp(2.0) # Apply pressure
        time.sleep(1.0) # Wait for grip to settle
       
        # 5. Haptic Check
        grip_current = self.get_gripper_feedback()
        if grip_current > GRIPPER_CURRENT_THRESHOLD:
            print(f"Object Detected! (Current: {grip_current} mA)")
        else:
            print(f"Warning: No object felt (Current: {grip_current} mA). Check position.")

        # 6. Lift Object
        self.monitor_move(lambda: self.bot.arm.set_ee_cartesian_trajectory(z=0.2, blocking=False), "Lifting")

        # 7. Place Object
        self.monitor_move(lambda: self.bot.arm.set_ee_cartesian_trajectory(x=-0.2, blocking=False), "Placing")
        self.bot.gripper.release()

        # 8. Return Home
        self.monitor_move(lambda: self.bot.arm.go_to_home_pose(blocking=False), "Returning Home")

if __name__ == '__main__':
    controller = SmartRobotController()
    try:
        controller.run_task()
    except KeyboardInterrupt:
        print("Stopping robot...")
        controller.bot.shutdown()
