import tkinter as tk
from tkinter import ttk
import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from scipy.spatial.transform import Rotation as R

class ViperXControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ViperX 300S Hardware Control")
        self.root.geometry("600x700")

        # 1. Initialize Robot Hardware
        # Ensure 'roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=vx300s' is running
        try:
            self.robot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm")
            self.robot.arm.go_to_home_pose()
        except Exception as e:
            print(f"Error initializing robot: {e}")
            self.root.destroy()
            return

        # Joint Names and Safe Limits (in Radians)
        self.joint_info = [
            {"name": "Waist", "min": -2.5, "max": 2.5},
            {"name": "Shoulder", "min": -1.8, "max": 1.9},
            {"name": "Elbow", "min": -2.1, "max": 1.6},
            {"name": "Wrist Angle", "min": -1.7, "max": 2.1},
            {"name": "Wrist Rotate", "min": -2.5, "max": 2.5},
            {"name": "Gripper Rotate", "min": -2.5, "max": 2.5}
        ]

        self.sliders = []
        self.setup_ui()
        
        # Start a loop to refresh live feedback
        self.update_feedback_loop()

    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="ViperX 300S Live Control", font=("Arial", 16, "bold"))
        header.pack(pady=10)

        # Sliders Frame
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(pady=10, padx=20, fill="x")

        for i, joint in enumerate(self.joint_info):
            lbl = tk.Label(slider_frame, text=f"{joint['name']} (rad)")
            lbl.grid(row=i, column=0, sticky="w", pady=5)
            
            slider = tk.Scale(slider_frame, from_=joint['min'], to=joint['max'], 
                              resolution=0.01, orient="horizontal", length=300,
                              command=lambda val: self.command_robot())
            slider.set(0.0)
            slider.grid(row=i, column=1, pady=5)
            self.sliders.append(slider)

        # Feedback Frame
        fb_frame = tk.LabelFrame(self.root, text="Forward Kinematics & Feedback", padx=10, pady=10)
        fb_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.pos_label = tk.Label(fb_frame, text="EE Position (m): [0, 0, 0]", font=("Courier", 10))
        self.pos_label.pack(anchor="w")

        self.ori_label = tk.Label(fb_frame, text="EE Orientation (°): [0, 0, 0]", font=("Courier", 10))
        self.ori_label.pack(anchor="w")

        self.vel_label = tk.Label(fb_frame, text="Joint Vels (rad/s): [0, 0, 0, 0, 0, 0]", font=("Courier", 10))
        self.vel_label.pack(anchor="w")

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Home Pose", command=self.go_home, width=15).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Sleep Pose", command=self.go_sleep, width=15, bg="salmon").pack(side="left", padx=5)

    def command_robot(self):
        """Sends slider values to the hardware."""
        targets = [s.get() for s in self.sliders]
        # moving_time=0.1 makes it very responsive to slider movement
        self.robot.arm.set_joint_positions(targets, moving_time=0.1, blocking=False)

    def update_feedback_loop(self):
        """Calculates FK and gets velocities from hardware."""
        try:
            # 1. Get current EE Pose (Forward Kinematics)
            T_ee = self.robot.arm.get_ee_pose()
            pos = T_ee[:3, 3]
            
            # 2. Convert Rotation Matrix to Euler (Orientation)
            rot_matrix = T_ee[:3, :3]
            euler = R.from_matrix(rot_matrix).as_euler('xyz', degrees=True)

            # 3. Get Joint Velocities
            vels = self.robot.dxl.joint_states.velocity[:6]

            # Update Labels
            self.pos_label.config(text=f"EE Position (m): X:{pos[0]:.3f} Y:{pos[1]:.3f} Z:{pos[2]:.3f}")
            self.ori_label.config(text=f"EE Ori (deg):   R:{euler[0]:.1f} P:{euler[1]:.1f} Y:{euler[2]:.1f}")
            self.vel_label.config(text=f"Joint Vels: {['{:.2f}'.format(v) for v in vels]}")

        except Exception as e:
            print(f"Feedback error: {e}")

        # Refresh every 100ms
        self.root.after(100, self.update_feedback_loop)

    def go_home(self):
        self.robot.arm.go_to_home_pose()
        for s in self.sliders: s.set(0.0)

    def go_sleep(self):
        self.robot.arm.go_to_sleep_pose()
        for s in self.sliders: s.set(0.0)

if __name__ == "__main__":
    root = tk.Tk()
    app = ViperXControlApp(root)
    root.mainloop()
