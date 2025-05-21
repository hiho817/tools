import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

# Define the file path
file_path = 'data/0520/force_test_stand.csv'

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File {file_path} not found.")
    exit(1)

# Read the CSV file
data = pd.read_csv(file_path)

# Print some information about the data
print(f"Data shape: {data.shape}")
# print(f"Columns: {data.columns.tolist()}")

# Extract force Fy data from all four modules
columns = [ 'force_Fx_a', 'force_Fx_b', 
            'force_Fx_c', 'force_Fx_d', 
            'force_Fy_a', 'force_Fy_b', 
            'force_Fy_c', 'force_Fy_d', 
            'imu_lin_acc_x', 'imu_lin_acc_y', 'imu_lin_acc_z']

# Check if all required columns exist
missing_columns = [col for col in columns if col not in data.columns]
if missing_columns:
    print(f"Warning: Missing columns: {missing_columns}")
    # print("Available columns:", data.columns.tolist())
    exit(1)


MASS = 20.0  # kg
GRAVITY = 9.81  # m/s^2
Fz = data['force_Fy_a'] + data['force_Fy_b'] + data['force_Fy_c'] + data['force_Fy_d'] - data['imu_lin_acc_z'] * MASS - GRAVITY * MASS
Fx = data['force_Fx_a'] + data['force_Fx_b'] + data['force_Fx_c'] + data['force_Fx_d'] - data['imu_lin_acc_x'] * MASS


# Define a parameter for the time range to plot
plot_time = [5.5, 10]  # in seconds, [init, final]

# Create time array (1000Hz sampling rate)
time = np.arange(len(data)) / 1000.0  # time in seconds

# Filter data based on the plot_time range
start_idx = int(plot_time[0] * 1000)  # Convert seconds to index
end_idx = int(plot_time[1] * 1000)    # Convert seconds to index

# Ensure indices are within bounds
start_idx = max(0, start_idx)
end_idx = min(len(time), end_idx)

# Plot the sum of forces over the specified time range
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time[start_idx:end_idx], np.array(Fz)[start_idx:end_idx], label='Fz')
plt.xlabel('Time (seconds)')
plt.ylabel('Force Fz (N)')
plt.title('Force Fz')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time[start_idx:end_idx], np.array(Fx)[start_idx:end_idx], label='Fx')
plt.xlabel('Time (seconds)')
plt.ylabel('Force Fx (N)')
plt.title('Force Fx')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()