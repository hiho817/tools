import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

# Define the file path
file_path = 'data/0514/force_walk.csv'

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
force_columns = ['force_Fx_a', 'force_Fx_b', 'force_Fx_c', 'force_Fx_d', 'force_Fy_a', 'force_Fy_b', 'force_Fy_c', 'force_Fy_d']

# Check if all required columns exist
missing_columns = [col for col in force_columns if col not in data.columns]
if missing_columns:
    print(f"Warning: Missing columns: {missing_columns}")
    # print("Available columns:", data.columns.tolist())
    exit(1)

# Calculate the sum of forces in Y direction for all modules
fx_sum = data['force_Fx_a'] + data['force_Fx_b'] + data['force_Fx_c'] + data['force_Fx_d']
fy_sum = data['force_Fy_a'] + data['force_Fy_b'] + data['force_Fy_c'] + data['force_Fy_d']
A_force = np.sqrt(data['force_Fx_a']**2 + data['force_Fy_a']**2)

# Create time array (1000Hz sampling rate)
time = np.arange(len(data)) / 1000.0  # time in seconds

# # Plot the sum of forces over time
# plt.figure(figsize=(10, 6))
# plt.plot(time, fy_sum)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Forces (N)')
# plt.title('Total Force in Y Direction from All Modules')
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig('total_fx_force.png')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(time, fx_sum)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Forces (N)')
# plt.title('Total Force in X Direction from All Modules')
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig('total_fx_force.png')
# plt.show()

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

plt.figure(figsize=(10, 6))
plt.plot(time, np.array(data['force_Fx_a']))
plt.xlabel('Time (seconds)')
plt.ylabel('Forces (N)')
plt.title('Force Fx from A Modules')
plt.grid(True)
plt.tight_layout()
# plt.savefig('total_fx_force.png')
plt.show()