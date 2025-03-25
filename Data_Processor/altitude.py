import numpy as np
from Data_Processor import ROS_Data, VICON_Data
from Marker_Vector import restrct

import pandas as pd

def quaternion_to_euler_df(df):
    """
    Convert a DataFrame of quaternions (ROS order: x, y, z, w) into Euler angles (roll, pitch, yaw).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the following columns:
            - 'imu_orien_x'
            - 'imu_orien_y'
            - 'imu_orien_z'
            - 'imu_orien_w'

    Returns
    -------
    euler_df : pandas.DataFrame
        A DataFrame with columns 'roll', 'pitch', and 'yaw' (in radians).
    """
    
    def convert_row(row):
        # Extract quaternion components (ROS convention: x, y, z, w)
        x = row['imu_orien_x']
        y = row['imu_orien_y']
        z = row['imu_orien_z']
        w = row['imu_orien_w']
        
        # Conversion formulas (assuming intrinsic rotations about X (roll), then Y (pitch), then Z (yaw))
        # Roll (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        
        # Pitch (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        # Clamp to avoid numerical issues with arcsin
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        
        # Yaw (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        
        return pd.Series({'roll': roll, 'pitch': pitch, 'yaw': yaw})
    
    # Apply the conversion row-wise.
    euler_df = df.apply(convert_row, axis=1)
    return euler_df

def compute_attitude_from_rect_points(O1, O2, O3, O4):
    """
    Compute the roll, pitch, and yaw (attitude) of a rigid body given 4 points
    that form a rectangle on the body. The points are assumed to be:
    
      O1: left front
      O2: right front
      O3: right behind
      O4: left behind
    
    The points may not lie exactly on a plane, so a best-fit plane is computed.
    
    Parameters
    ----------
    O1, O2, O3, O4 : array-like (3,)
        Each point is provided as [x, y, z] in global coordinates.
    
    Returns
    -------
    roll, pitch, yaw : tuple of floats
        The Euler angles (in radians) representing the orientation of the body.
        The rotation matrix is constructed as:
            R = [ forward, right, up ]
        and is assumed to decompose as:
            R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    """
    # Check for NaN values in the input points
    if any(np.isnan(O1)) or any(np.isnan(O2)) or any(np.isnan(O3)) or any(np.isnan(O4)):
        return np.nan, np.nan, np.nan
    # Stack the points into an array of shape (4,3)
    points = np.array([O1, O2, O3, O4])
    
    # Compute the centroid
    centroid = np.mean(points, axis=0)
    
    # Fit a plane via SVD (PCA) on the centered points
    centered_points = points - centroid
    U, S, Vt = np.linalg.svd(centered_points)
    # The plane normal is the eigenvector corresponding to the smallest singular value.
    n = Vt[2, :]
    # Ensure the normal points upward (global z positive)
    if n[2] < 0:
        n = -n
        
    # Project points onto the fitted plane
    def project_point(P, center, normal):
        return P - np.dot(P - center, normal) * normal
    
    O1_proj = project_point(O1, centroid, n)
    O2_proj = project_point(O2, centroid, n)
    O3_proj = project_point(O3, centroid, n)
    O4_proj = project_point(O4, centroid, n)
    
    # Compute front center (average of left-front and right-front) and back center (average of left-behind and right-behind)
    front_center = (np.array(O1_proj) + np.array(O2_proj)) / 2.0
    back_center  = (np.array(O4_proj) + np.array(O3_proj)) / 2.0
    
    # Define the forward direction from back to front (projected on the plane)
    forward = front_center - back_center
    forward = forward / np.linalg.norm(forward)
    
    # Compute the right axis: right = cross(up, forward)
    right = np.cross(n, forward)
    right = right / np.linalg.norm(right)
    
    # Construct the rotation matrix with columns: forward (x-axis), right (y-axis), and up (z-axis)
    R = np.column_stack((forward, right, n))
    
    # --- Extract Euler angles from R ---
    # We assume the following convention:
    #   R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    # A common extraction (for non-singular cases) is:
    #   yaw   = atan2(R[1,0], R[0,0])
    #   pitch = arcsin(-R[2,0])
    #   roll  = atan2(R[2,1], R[2,2])
    
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(-R[2, 0])
    roll  = np.arctan2(R[2, 1], R[2, 2])
    
    return roll, pitch, yaw

# Example usage:
if __name__ == "__main__":

    ros_data = ROS_Data("data/imu_test_from_imu.csv")
    ros_data.load_data()
    ros_data.process_data()
    ros_data_process = ros_data.get_processed_data()
    imu_data = {
        "imu_orien_x": ros_data_process.imu_orien_x,
        "imu_orien_y": ros_data_process.imu_orien_y,
        "imu_orien_z": ros_data_process.imu_orien_z,
        "imu_orien_w": ros_data_process.imu_orien_w
    }
    # Convert the dictionary to a DataFrame
    imu_data = pd.DataFrame(imu_data)
    imu_altitude_df = quaternion_to_euler_df(imu_data)

    vicon_data = VICON_Data("data/imu_test_from_vicon.csv", trigger_name="20250324:O5")
    vicon_data.load_data()
    LF = restrct(vicon_data.traj_data, 'O1')
    RF = restrct(vicon_data.traj_data, 'O2')
    RB = restrct(vicon_data.traj_data, 'O3')
    LB = restrct(vicon_data.traj_data, 'O4')

    # Compute attitude (roll, pitch, yaw) for each time frame (row)
    vicon_altitude = []
    for idx in vicon_data.traj_data.index[vicon_data.trigger_index+1:]:
        # For each marker, extract its X, Y, Z as a list of floats.
        try:
            # Parse coordinates safely, replacing empty strings with NaN
            def safe_float(val):
                if val == '' or pd.isna(val):
                    return np.nan
                return float(val)
            
            LF_point = [safe_float(LF.loc[idx, col]) for col in ['X', 'Y', 'Z']]
            RF_point = [safe_float(RF.loc[idx, col]) for col in ['X', 'Y', 'Z']]
            RB_point = [safe_float(RB.loc[idx, col]) for col in ['X', 'Y', 'Z']]
            LB_point = [safe_float(LB.loc[idx, col]) for col in ['X', 'Y', 'Z']]
        
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            continue
        
        # print(f"LF: {LF_point}, RF: {RF_point}, RB: {RB_point}, LB: {LB_point}")

        roll, pitch, yaw = compute_attitude_from_rect_points(LF_point, RF_point, RB_point, LB_point)
        vicon_altitude.append({'roll': roll, 'pitch': pitch, 'yaw': yaw})

    # Convert the list of Euler angles into a DataFrame for further analysis
    vicon_altitude_df = pd.DataFrame(vicon_altitude)



    ############################ compare altitude ############################

    #data pre processing
    vicon_altitude_df['pitch'] = -vicon_altitude_df['pitch']
    vicon_altitude_df['yaw'] = -vicon_altitude_df['yaw']

    # offset correction
    offset = vicon_altitude_df['roll'][0] - imu_altitude_df['roll'][0]
    imu_altitude_df['roll'] += offset
    offset = vicon_altitude_df['pitch'][0] - imu_altitude_df['pitch'][0]
    imu_altitude_df['pitch'] += offset
    offset = vicon_altitude_df['yaw'][0] - imu_altitude_df['yaw'][0]
    imu_altitude_df['yaw'] += offset


    # Import matplotlib for plotting
    import matplotlib.pyplot as plt

    # Create a figure with 3 subplots (one for each angle)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot roll, pitch and yaw on separate subplots
    # Plot Vicon data
    axs[0].plot(vicon_altitude_df.index*2, vicon_altitude_df['roll'], 'r-', label='Vicon Roll')
    axs[1].plot(vicon_altitude_df.index*2, vicon_altitude_df['pitch'], 'g-', label='Vicon Pitch')
    axs[2].plot(vicon_altitude_df.index*2, vicon_altitude_df['yaw'], 'b-', label='Vicon Yaw')

    # Plot IMU data
    axs[0].plot(imu_altitude_df.index, imu_altitude_df['roll'], 'r--', label='IMU Roll')
    axs[1].plot(imu_altitude_df.index, imu_altitude_df['pitch'], 'g--', label='IMU Pitch')
    axs[2].plot(imu_altitude_df.index, imu_altitude_df['yaw'], 'b--', label='IMU Yaw')

    # Add labels and titles
    axs[0].set_ylabel('Roll (rad)')
    axs[0].set_title('Roll Angle Comparison')
    axs[0].legend()

    axs[1].set_ylabel('Pitch (rad)')
    axs[1].set_title('Pitch Angle Comparison')
    axs[1].legend()

    axs[2].set_ylabel('Yaw (rad)')
    axs[2].set_title('Yaw Angle Comparison')
    axs[2].set_xlabel('Frame Index')
    axs[2].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

    #calculate RMS error
    roll_error = np.sqrt(np.mean((vicon_altitude_df['roll'] - imu_altitude_df['roll'])**2))
    pitch_error = np.sqrt(np.mean((vicon_altitude_df['pitch'] - imu_altitude_df['pitch'])**2))  
    yaw_error = np.sqrt(np.mean((vicon_altitude_df['yaw'] - imu_altitude_df['yaw'])**2))

    print(f"Roll RMS Error: {roll_error}")
    print(f"Pitch RMS Error: {pitch_error}")
    print(f"Yaw RMS Error: {yaw_error}")
    
    sampling_time = len(vicon_data.traj_data.index[vicon_data.trigger_index+1:])/vicon_data.traj_frequency
    print(sampling_time)
