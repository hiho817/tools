import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

# Define the file path
file_path = 'data/0520/force_test_stand_updown.csv'

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

# Calculate the sum of forces in Y direction for all modules
fx_sum = data['force_Fx_a'] + data['force_Fx_b'] + data['force_Fx_c'] + data['force_Fx_d']
fy_sum = data['force_Fy_a'] + data['force_Fy_b'] + data['force_Fy_c'] + data['force_Fy_d']
A_force = np.sqrt(data['force_Fx_a']**2 + data['force_Fy_a']**2)

fy_sum_acc = fx_sum + data['imu_lin_acc_z'] * 20.0 # kg

# Create time array (1000Hz sampling rate)
time = np.arange(len(data)) / 1000.0  # time in seconds

# Plot the sum of forces over time
plt.figure(figsize=(10, 6))
plt.plot(time, np.array(fy_sum))
plt.xlabel('Time (seconds)')
plt.ylabel('Forces (N)')
plt.title('Total Force in Y Direction from All Modules')
plt.grid(True)
plt.tight_layout()
plt.savefig('fy_sum.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, np.array(fx_sum))
plt.xlabel('Time (seconds)')
plt.ylabel('Forces (N)')
plt.title('Total Force in X Direction from All Modules')
plt.grid(True)
plt.tight_layout()
plt.savefig('fx_sum.png')
plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(time, A_force)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Forces (N)')
# plt.title('Force from A Modules')
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig('total_fx_force.png')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(time, data['force_Fy_a'])
# plt.xlabel('Time (seconds)')
# plt.ylabel('Forces (N)')
# plt.title('Force Fy from A Modules')
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig('total_fx_force.png')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(time, np.array(data['force_Fx_a']))
# plt.xlabel('Time (seconds)')
# plt.ylabel('Forces (N)')
# plt.title('Force Fx from A Modules')
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig('total_fx_force.png')
# plt.show()

# fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True)

# # Plot for module A
# axes[0].plot(time, np.array(data['force_Fx_a']), label='Fx')
# # axes[0].plot(time, data['force_Fy_a'], label='Fy')
# axes[0].set_ylabel('Force (N)')
# axes[0].set_title('Module A Forces')
# axes[0].legend()
# axes[0].grid(True)

# # Plot for module B
# axes[1].plot(time, np.array(data['force_Fx_b']), label='Fx')
# # axes[1].plot(time, data['force_Fy_b'], label='Fy')
# axes[1].set_ylabel('Force (N)')
# axes[1].set_title('Module B Forces')
# axes[1].legend()
# axes[1].grid(True)

# # Plot for module C
# axes[2].plot(time, np.array(data['force_Fx_c']), label='Fx')
# # axes[2].plot(time, data['force_Fy_c'], label='Fy')
# axes[2].set_ylabel('Force (N)')
# axes[2].set_title('Module C Forces')
# axes[2].legend()
# axes[2].grid(True)

# # Plot for module D
# axes[3].plot(time, np.array(data['force_Fx_d']), label='Fx')
# # axes[3].plot(time, data['force_Fy_d'], label='Fy')
# axes[3].set_xlabel('Time (seconds)')
# axes[3].set_ylabel('Force (N)')
# axes[3].set_title('Module D Forces')
# axes[3].legend()
# axes[3].grid(True)

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(time, np.array(data['imu_lin_acc_z']))
# plt.xlabel('Time (seconds)')
# plt.ylabel('acceleration (N)')
# plt.title('Body Z acceleration')
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig('total_fx_force.png')
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, np.array(fy_sum_acc))
plt.xlabel('Time (seconds)')
plt.ylabel('Total Force (N)')
plt.title('Z direction kinostaic')
plt.grid(True)
plt.tight_layout()
plt.savefig('total_fx_force.png')
plt.show()