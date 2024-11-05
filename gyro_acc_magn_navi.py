import numpy as np
from scipy.spatial.transform import Rotation as R
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

# Prediction step function
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

# Update step function
def update(state, covariance, accel, mag):
    q = state[:4]

    # Normalize magnetometer reading and apply a yaw constraint
    mag_normalized = mag / np.linalg.norm(mag)
    mag_drift_constraint = np.array([1.0, 0.0, 0.0])  # Assuming north aligns with x-axis
    yaw_error = np.arctan2(mag_normalized[1], mag_normalized[0]) - np.arctan2(mag_drift_constraint[1], mag_drift_constraint[0])

    # Measurement update based on accelerometer (gravity direction) and magnetometer (magnetic north)
    accel_normalized = accel / np.linalg.norm(accel)
    gravity_body = np.array([0, 0, 1])  # Expected gravity in body frame

    # Compute measurement residuals
    residual_accel = accel_normalized - gravity_body
    residual_mag = np.array([yaw_error, 0, 0])  # Use only yaw correction for magnetometer

    # Concatenate measurements
    residual = np.concatenate((residual_accel, residual_mag))

    # Adaptive tuning for measurement noise
    if np.linalg.norm(state[4:7]) < 0.1:  # If velocity is close to zero
        measurement_noise[:3, :3] *= 0.5  # Reduce noise for accelerometer
    else:
        measurement_noise[:3, :3] *= 1.5  # Increase noise for accelerometer

    # Kalman gain
    S = covariance[:6, :6] + measurement_noise
    K = covariance[:, :6] @ np.linalg.inv(S)

    # Update state and covariance
    state[:4] += K[:4, :3] @ residual[:3]  # Update quaternion with accel residual
    state[4:] += K[4:, 3:] @ residual[3:]  # Update with magnetometer residual
    covariance = (np.eye(len(covariance)) - K @ K.T) @ covariance

    return state, covariance

# Main loop to simulate the filter over time
for _ in range(1000):
    gyro = get_gyroscope_data()
    accel = get_accelerometer_data()
    mag = get_magnetometer_data()

    # Prediction step
    state, covariance = predict(state, covariance, gyro, accel, dt)
    
    # Update step
    state, covariance = update(state, covariance, accel, mag)
    
    # Extract orientation from state
    q = state[:4]
    orientation = R.from_quat(q)
    euler = orientation.as_euler('xyz', degrees=True)  # Pitch, Roll, Yaw

    # Display orientation, velocity, and position
    print(f"Pitch: {euler[0]:.2f}, Roll: {euler[1]:.2f}, Yaw: {euler[2]:.2f}")
    print(f"Velocity (m/s): {state[4:7]}")
    print(f"Position (m): {state[7:]}")
    
    # Simulate sensor reading frequency
    time.sleep(dt)
