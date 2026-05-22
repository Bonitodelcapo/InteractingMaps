import numpy as np
import matplotlib.pyplot as plt
import os

# Update this path if your data is somewhere else!
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'shapes_rotation')
IMU_FILE = os.path.join(DATA_DIR, 'imu.txt')

def find_constant_velocity():
    print(f"Loading IMU data from {IMU_FILE}...")
    
    # Load the IMU data. 
    # UZH datasets usually have a header we need to skip.
    try:
        data = np.loadtxt(IMU_FILE, comments='#')
    except Exception as e:
        print(f"Error loading IMU file: {e}")
        return

    # Extract timestamps and the Z-axis angular velocity (w_z)
    timestamps = data[:, 0]
    
    # Normalize timestamps to start at 0 seconds
    t_start_zero = timestamps - timestamps[0] 
    
    w_z = data[:, 6] # Column 6 is usually w_z (Rotation around the Z axis)

    # Plot the angular velocity
    plt.figure(figsize=(10, 5))
    plt.plot(t_start_zero, w_z, label="Angular Velocity (w_z)", color='blue')
    plt.title("Camera Rotation Velocity over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.grid(True)
    
    # Draw a line at zero
    plt.axhline(0, color='black', linewidth=1)
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    find_constant_velocity()