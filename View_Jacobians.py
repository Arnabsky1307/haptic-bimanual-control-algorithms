import numpy as np
import math

class ViperXController:
    def __init__(self):
        # --- Robot Constants (Same as before) ---
        self.d1 = 0.12705
        self.a2 = 0.30594
        self.d4 = 0.30000
        self.d6 = 0.17700
        self.gamma2_offset = math.radians(11.31)

    def modified_dh_transform(self, alpha_prev, a_prev, d, theta):
        """Standard Modified DH Transform helper"""
        ct = np.cos(theta); st = np.sin(theta)
        ca = np.cos(alpha_prev); sa = np.sin(alpha_prev)
        return np.array([
            [ct,      -st,      0,    a_prev],
            [st*ca,   ct*ca,   -sa,  -d*sa],
            [st*sa,   ct*sa,    ca,   d*ca],
            [0,       0,        0,    1   ]
        ])

    def get_jacobian(self, q):
        """Computes 6x6 Jacobian Matrix for current joint angles q"""
        # 1. Forward Kinematics to get all frame transforms
        transforms = []
        T_cumulative = np.eye(4)
        transforms.append(T_cumulative) # Base frame
       
        # DH Table with current q
        dh_params = [
            [0,        0,       self.d1,   q[0]],
            [np.pi/2,  0,       0,         q[1] + self.gamma2_offset],
            [0,        self.a2, 0,         q[2]],
            [-np.pi/2, 0,       self.d4,   q[3]],
            [np.pi/2,  0,       0,         q[4]],
            [-np.pi/2, 0,       self.d6,   q[5]]
        ]

        for params in dh_params:
            T_i = self.modified_dh_transform(*params)
            T_cumulative = np.dot(T_cumulative, T_i)
            transforms.append(T_cumulative)

        # 2. Build Jacobian
        p_e = transforms[-1][:3, 3] # End effector position
        J = np.zeros((6, 6))

        for i in range(6):
            T_prev = transforms[i]
            z_prev = T_prev[:3, 2] # Rotation axis
            p_prev = T_prev[:3, 3] # Frame origin
           
            # Linear part (cross product)
            J[:3, i] = np.cross(z_prev, p_e - p_prev)
            # Angular part
            J[3:, i] = z_prev
           
        return J

    # ---------------------------------------------------------
    # NEW: Inverse Jacobian Methods
    # ---------------------------------------------------------

    def compute_pseudo_inverse(self, J):
        """
        Calculates Moore-Penrose Pseudo-Inverse (J+).
        Standard NumPy implementation.
        """
        # Threshold below which singular values are treated as zero
        return np.linalg.pinv(J, rcond=1e-3)

    def compute_damped_least_squares(self, J, damping_lambda=0.01):
        """
        Calculates Inverse using Damped Least Squares (DLS).
        Formula: J* = J_T * (J * J_T + lambda^2 * I)^-1
       
        Best for: Singularities. It sacrifices exact accuracy for
        stability (stops robot from shaking violently).
        """
        rows, cols = J.shape
        J_T = J.T
        I = np.eye(rows)
       
        # The DLS formula
        # (J * J_T) + (lambda^2 * I)
        inner_term = np.dot(J, J_T) + (damping_lambda**2 * I)
        inverse_inner = np.linalg.inv(inner_term)
       
        return np.dot(J_T, inverse_inner)

    def solve_velocity_ik(self, q_current, desired_twist, method="dls"):
        """
        Calculates required JOINT VELOCITIES (q_dot) to achieve
        a desired CARTESIAN VELOCITY (twist).
       
        Args:
            q_current: Current joint angles [rad]
            desired_twist: Desired [vx, vy, vz, wx, wy, wz]
            method: 'pinv' or 'dls'
           
        Returns:
            q_dot: Array of 6 joint velocities [rad/s]
        """
        # 1. Get current Jacobian
        J = self.get_jacobian(q_current)
       
        # 2. Invert Jacobian
        if method == "pinv":
            J_inv = self.compute_pseudo_inverse(J)
        else:
            J_inv = self.compute_damped_least_squares(J, damping_lambda=0.1)
           
        # 3. Calculate q_dot = J_inv * desired_twist
        q_dot = np.dot(J_inv, desired_twist)
       
        return q_dot

# --- MAIN EXECUTION SIMULATION ---
if __name__ == "__main__":
    controller = ViperXController()
    np.set_printoptions(precision=4, suppress=True)

    # 1. Current State (Arbitrary Position)
    current_q = [0.0, -0.5, 0.5, 0.0, 1.0, 0.0]
   
    # 2. Command: "Move Straight Up"
    # [vx=0, vy=0, vz=0.1 m/s, wx=0, wy=0, wz=0]
    desired_cartesian_vel = np.array([0, 0, 0.1, 0, 0, 0])

    print(f"Current Joint Config: {current_q}")
    print(f"Desired Cartesian Vel: {desired_cartesian_vel}")
    print("-" * 50)

    # --- Method A: Standard Pseudo-Inverse ---
    q_dot_pinv = controller.solve_velocity_ik(current_q, desired_cartesian_vel, method="pinv")
    print("\n[Method A] Standard Pseudo-Inverse Joint Velocities (rad/s):")
    print(q_dot_pinv)
   
    # --- Method B: Damped Least Squares (DLS) ---
    q_dot_dls = controller.solve_velocity_ik(current_q, desired_cartesian_vel, method="dls")
    print("\n[Method B] Damped Least Squares Joint Velocities (rad/s):")
    print(q_dot_dls)

    print("\n" + "="*50)
    print("INTERPRETATION:")
    print(f"To move the robot UP at 10cm/s, Joint 2 must move at {q_dot_dls[1]:.4f} rad/s")
    print(f"and Joint 3 must move at {q_dot_dls[2]:.4f} rad/s.")
