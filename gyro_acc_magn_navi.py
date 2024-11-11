import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Placeholder functions to simulate real sensor readings
def get_gyroscope_data():
    return np.random.randn(3) * 0.01  # Simulated small random rotation rates

def get_accelerometer_data():
    return np.array([0, 0, 9.81]) + np.random.randn(3) * 0.02  # Simulated gravity with noise

def get_magnetometer_data():
    return np.array([1.0, 0.0, 0.0]) + np.random.randn(3) * 0.02  # Simulating magnetic north with noise

# Initialize EKF parameters
dt = 0.01  # Time step
state = np.zeros(10)  # [qx, qy, qz, qw, vx, vy, vz, px, py, pz] (orientation, velocity, position)
state[:4] = [0, 0, 0, 1]  # Initialize quaternion to no rotation
covariance = np.eye(10) * 0.01  # Initial covariance matrix
process_noise = np.eye(10) * 0.001
measurement_noise = np.eye(6) * 0.1

# Gravity vector in world frame
gravity_world = np.array([0, 0, 9.81])

# Bias correction values (should ideally be calculated offline)
gyro_bias = np.array([0.001, 0.002, 0.001])  # Gyroscope biases (in rad/s)
accel_bias = np.array([0.1, -0.05, 0.05])    # Accelerometer biases (in m/s^2)

# Prediction and update functions
def predict(state, covariance, gyro, accel, dt):
    q = state[:4]  # Quaternion
    vel = state[4:7]  # Velocity
    pos = state[7:]  # Position

    # Correct for sensor bias
    gyro_corrected = gyro - gyro_bias
    accel_corrected = accel - accel_bias

    # Predict orientation using gyroscope data
    rot = R.from_quat(q)
    angular_velocity = R.from_rotvec(gyro_corrected * dt)
    q_new = (rot * angular_velocity).as_quat()

    # Rotate accelerometer data to world frame and subtract gravity
    accel_world = rot.apply(accel_corrected) - gravity_world

    # Predict new velocity and position
    vel_new = vel + accel_world * dt
    pos_new = pos + vel * dt + 0.5 * accel_world * (dt ** 2)

    # Update state and covariance
    state[:4] = q_new
    state[4:7] = vel_new
    state[7:] = pos_new
    covariance = covariance + process_noise

    return state, covariance

def update(state, covariance, accel, mag):
    q = state[:4]
    mag_normalized = mag / np.linalg.norm(mag)
    mag_drift_constraint = np.array([1.0, 0.0, 0.0])  # Assuming north aligns with x-axis
    yaw_error = np.arctan2(mag_normalized[1], mag_normalized[0]) - np.arctan2(mag_drift_constraint[1], mag_drift_constraint[0])
    accel_normalized = accel / np.linalg.norm(accel)
    gravity_body = np.array([0, 0, 1])  # Expected gravity in body frame
    residual_accel = accel_normalized - gravity_body
    residual_mag = np.array([yaw_error, 0, 0])
    residual = np.concatenate((residual_accel, residual_mag))
    S = covariance[:6, :6] + measurement_noise
    K = covariance[:, :6] @ np.linalg.inv(S)
    state[:4] += K[:4, :3] @ residual[:3]
    state[4:] += K[4:, 3:] @ residual[3:]
    covariance = (np.eye(len(covariance)) - K @ K.T) @ covariance
    return state, covariance

# Set up Matplotlib figure and subplots for live visualization
fig, (ax_orientation, ax_velocity, ax_position) = plt.subplots(3, 1, figsize=(8, 10))
plt.tight_layout()

# Initialize the orientation, velocity, and position lines
pitch_line, = ax_orientation.plot([], [], 'r-', label='Pitch')
roll_line, = ax_orientation.plot([], [], 'g-', label='Roll')
yaw_line, = ax_orientation.plot([], [], 'b-', label='Yaw')
ax_orientation.legend()
ax_orientation.set_xlim(0, 100)
ax_orientation.set_ylim(-180, 180)

vx_line, = ax_velocity.plot([], [], 'r-', label='Vx')
vy_line, = ax_velocity.plot([], [], 'g-', label='Vy')
vz_line, = ax_velocity.plot([], [], 'b-', label='Vz')
ax_velocity.legend()
ax_velocity.set_xlim(0, 100)
ax_velocity.set_ylim(-10, 10)
ax_velocity.set_title("Velocity (m/s)")

px_line, = ax_position.plot([], [], 'r-', label='Px')
py_line, = ax_position.plot([], [], 'g-', label='Py')
pz_line, = ax_position.plot([], [], 'b-', label='Pz')
ax_position.legend()
ax_position.set_xlim(0, 100)
ax_position.set_ylim(-50, 50)
ax_position.set_title("Position (m)")

# Data history for plotting
time_data = []
pitch_data, roll_data, yaw_data = [], [], []
vx_data, vy_data, vz_data = [], [], []
px_data, py_data, pz_data = [], [], []

# Update function for animation
def update_plot(frame):
    global state, covariance

    # Simulate sensor readings
    gyro = get_gyroscope_data()
    accel = get_accelerometer_data()
    mag = get_magnetometer_data()

    # Prediction and Update steps
    state, covariance = predict(state, covariance, gyro, accel, dt)
    state, covariance = update(state, covariance, accel, mag)

    # Extract orientation (Pitch, Roll, Yaw)
    q = state[:4]
    orientation = R.from_quat(q)
    euler = orientation.as_euler('xyz', degrees=True)  # Pitch, Roll, Yaw

    # Update data history for orientation, velocity, and position
    time_data.append(len(time_data))
    pitch_data.append(euler[0])
    roll_data.append(euler[1])
    yaw_data.append(euler[2])

    vx_data.append(state[4])
    vy_data.append(state[5])
    vz_data.append(state[6])

    px_data.append(state[7])
    py_data.append(state[8])
    pz_data.append(state[9])

    # Update orientation plot
    pitch_line.set_data(time_data, pitch_data)
    roll_line.set_data(time_data, roll_data)
    yaw_line.set_data(time_data, yaw_data)

    # Update velocity plot
    vx_line.set_data(time_data, vx_data)
    vy_line.set_data(time_data, vy_data)
    vz_line.set_data(time_data, vz_data)

    # Update position plot
    px_line.set_data(time_data, px_data)
    py_line.set_data(time_data, py_data)
    pz_line.set_data(time_data, pz_data)

    # Update plot limits for scrolling effect
    ax_orientation.set_xlim(max(0, len(time_data) - 100), len(time_data))
    ax_velocity.set_xlim(max(0, len(time_data) - 100), len(time_data))
    ax_position.set_xlim(max(0, len(time_data) - 100), len(time_data))

    return pitch_line, roll_line, yaw_line, vx_line, vy_line, vz_line, px_line, py_line, pz_line

# Start animation
ani = FuncAnimation(fig, update_plot, frames=1000, interval=dt * 1000, blit=True)
plt.show()
