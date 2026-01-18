import sys
import time
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

# --- CONFIGURATION ---
robot_model = 'vx300s'

# Shared variable to pass data from Robot Thread to GUI
current_joint_angles = [0.0] * 6
gui_running = True

def robot_logic_thread(bot):
    """
    This function contains your original logic, modified to:
    1. Run in a separate thread so it doesn't block the GUI.
    2. Update the shared 'current_joint_angles' variable.
    3. Open the gripper automatically after grasping.
    """
    global current_joint_angles
    
    print("Going to home pose")
    bot.arm.go_to_home_pose()
    bot.gripper.release()
    time.sleep(2.0)

    # Initial update of joints
    current_joint_angles = bot.arm.get_joint_commands()

    while gui_running:
        print("\n--- Enter coordinates ---")
        print("Type 'q' to quit")
        try:
            # Note: This input() blocks this thread, but NOT the GUI thread
            user_input = input("Enter coordinates of X, Y, Z: ")

            if user_input.lower() == 'q':
                print("Quitting...")
                sys.exit()

            coords = user_input.split()
            if len(coords) != 3:
                print("Error: Please enter exactly 3 numbers!")
                continue

            x_target = float(coords[0])
            y_target = float(coords[1])
            z_target = float(coords[2])

            print(f"Moving to: {x_target}, {y_target}, {z_target}")

            # 1. First, move the arm (Inverse Kinematics happens here)
            result = bot.arm.set_ee_pose_components(
                x=x_target,
                y=y_target,
                z=z_target,
                roll=0,
                pitch=1.57,  # 1.57 rad is approx 90 deg (Gripper pointing DOWN)
                yaw=0,
                execute=True
            )

            # 2. Update the GUI with the calculated IK joint angles
            # We fetch the commands the robot just executed
            current_joint_angles = bot.arm.get_joint_commands()

            # 3. Check if movement succeeded, THEN grasp
            if result:
                print("SUCCESS!! Reached target.")
                
                print(">> Closing Gripper (Grasp)...")
                bot.gripper.grasp()
                time.sleep(2.0) # Hold object for 2 seconds
                
                print(">> Opening Gripper (Release)...")
                bot.gripper.release()
                time.sleep(1.0) # Wait for release
                
            else:
                print("FAILED!! Target unreachable.")

        except ValueError:
            print("Invalid Input. Please enter numbers.")
        except Exception as e:
            print(f"An error occurred: {e}")

    print("Moving to sleep position")
    bot.arm.go_to_sleep_pose()
    print("Done")

# --- GUI CLASS ---
class JointDisplayGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Interbotix {robot_model} - IK Monitor")
        self.root.geometry("350x300")
        
        # Title
        ttk.Label(root, text="Inverse Kinematics Solution", font=("Helvetica", 12, "bold")).pack(pady=10)
        ttk.Label(root, text="(Joint Angles in Radians)", font=("Helvetica", 9)).pack(pady=0)

        # Labels for 6 joints
        self.joint_labels = []
        self.joint_names = ['Waist', 'Shoulder', 'Elbow', 'Forearm Roll', 'Wrist Angle', 'Wrist Rotate']
        
        frame = ttk.Frame(root, padding=10)
        frame.pack(fill="both", expand=True)

        for name in self.joint_names:
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=5)
            ttk.Label(row, text=f"{name}:", width=15, font=("Consolas", 10)).pack(side="left")
            val_lbl = ttk.Label(row, text="0.0000", font=("Consolas", 10, "bold"), foreground="blue")
            val_lbl.pack(side="right")
            self.joint_labels.append(val_lbl)

        # Start the update loop
        self.root.after(100, self.update_display)

    def update_display(self):
        """Polls the shared variable and updates the text."""
        global current_joint_angles
        
        for i, lbl in enumerate(self.joint_labels):
            if i < len(current_joint_angles):
                lbl.config(text=f"{current_joint_angles[i]:.4f}")
        
        # Schedule next update
        self.root.after(100, self.update_display)

def main():
    global gui_running
    
    # 1. Initialize Robot (This must happen in the main process)
    print(f"Initializing {robot_model}")
    bot = InterbotixManipulatorXS(
        robot_model=robot_model,
        group_name='arm',
        gripper_name='gripper'
    )

    # 2. Start the GUI Root
    root = tk.Tk()
    gui = JointDisplayGUI(root)

    # 3. Start the Robot Logic in a separate Daemon thread
    # Daemon means it will die if the main GUI window closes
    t = threading.Thread(target=robot_logic_thread, args=(bot,), daemon=True)
    t.start()

    # 4. Run GUI Main Loop (Blocking)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        gui_running = False
        # Try to shut down gracefully
        # bot.arm.go_to_sleep_pose() # Optional: Sleep on exit

if __name__ == '__main__':
    main()
