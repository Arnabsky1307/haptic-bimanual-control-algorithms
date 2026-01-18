import tkinter as tk
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np

class RobotSliderGUI:
    def __init__(self, root, bot):
        self.bot = bot
        self.root = root
        self.root.title("RX200 Joint Control")

        # Joint Names for RX200 (5 DOF)
        self.joint_names = ['waist', 'shoulder', 'elbow', 'wrist_angle', 'wrist_rotate']
        
        # Approximate safe limits (in radians) for the GUI
        # You can adjust these based on your specific workspace needs
        self.limits = [
            (-3.1, 3.1),  # Waist
            (-1.8, 1.8),  # Shoulder
            (-1.7, 1.5),  # Elbow
            (-1.7, 2.1),  # Wrist Angle
            (-3.1, 3.1)   # Wrist Rotate
        ]

        self.sliders = []
        
        # Create a slider for each joint
        for i, name in enumerate(self.joint_names):
            lbl = tk.Label(root, text=f"{name.capitalize()} (rad)")
            lbl.pack(pady=5)
            
            # Create scale
            # Resolution is 0.01 radians
            slider = tk.Scale(
                root, 
                from_=self.limits[i][0], 
                to=self.limits[i][1], 
                orient='horizontal', 
                resolution=0.05, 
                length=300
            )
            slider.set(0) # Start at 0
            slider.pack()
            
            # Bind the "release" of the mouse to the move command
            # This prevents jerky motion while dragging
            slider.bind("<ButtonRelease-1>", self.update_robot)
            self.sliders.append(slider)

        # Home Button
        btn_home = tk.Button(root, text="Go Home", command=self.go_home, bg="green", fg="white")
        btn_home.pack(pady=20)

        # Sleep Button
        btn_sleep = tk.Button(root, text="Go to Sleep", command=self.go_sleep, bg="red", fg="white")
        btn_sleep.pack(pady=5)

    def update_robot(self, event=None):
        """Read all sliders and send command to robot"""
        positions = [s.get() for s in self.sliders]
        print(f"Moving to: {positions}")
        
        # blocking=False allows the GUI to remain responsive
        # moving_time=1.5 seconds ensures smooth movement
        self.bot.arm.set_joint_positions(positions, moving_time=1.5, blocking=False)

    def go_home(self):
        """Move robot to Home pose and update sliders"""
        print("Going Home...")
        self.bot.arm.go_to_home_pose()
        # Reset sliders to home positions (usually [0, 0, 0, 0, 0])
        for slider in self.sliders:
            slider.set(0)

    def go_sleep(self):
        """Move robot to Sleep pose"""
        print("Going to Sleep...")
        self.bot.arm.go_to_sleep_pose()

def main():
    # 1. Initialize the Interbotix Robot
    # The 'rx200' model is 5DOF
    bot = InterbotixManipulatorXS(
        robot_model='rx200',
        group_name='arm',
        gripper_name='gripper'
    )

    # 2. Setup the GUI
    root = tk.Tk()
    gui = RobotSliderGUI(root, bot)
    
    # 3. Start the loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        bot.shutdown()

if __name__ == '__main__':
    main()
