from Data_Processor import ROS_Data, VICON_Data
from Marker_Vector import restrct
from altitude import compute_attitude_from_rect_points
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

YAML = False 

def low_pass_filter(data, cutoff_freq, sample_rate):
    """
    Apply a low-pass filter to the data.
    
    Parameters:
        data (np.ndarray): The input data to filter.
        cutoff_freq (float): The cutoff frequency for the low-pass filter.
        sample_rate (float): The sample rate of the data.
        
    Returns:
        np.ndarray: The filtered data.
    """
    from scipy.signal import butter, filtfilt

    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def FFT(data, sample_rate):
    """
    Plot the FFT of the data.
    
    Parameters:
        data (np.ndarray): The input data to analyze.
        sample_rate (float): The sample rate of the data.
        
    Returns:
        None
    """
    n = len(data)
    freq = np.fft.fftfreq(n, d=1/sample_rate)
    fft_values = np.fft.fft(data)
    
    plt.figure(figsize=(12, 6))
    plt.plot(freq[:n//2], np.abs(fft_values)[:n//2])
    plt.title('FFT of Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()

def PSD (data, sample_rate):
    """
    Plot the Power Spectral Density (PSD) of the data.
    
    Parameters:
        data (np.ndarray): The input data to analyze.
        sample_rate (float): The sample rate of the data.
        
    Returns:
        None
    """
    from scipy.signal import welch

    f, Pxx = welch(data, fs=sample_rate, nperseg=1024)
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(f, Pxx)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid()
    plt.show()

# Load configurations from YAML file
def load_yaml_config(file_path):
    """
    Load YAML configurations from a file.
    
    Parameters:
        file_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Loaded configuration dictionary or empty dict if file not found.
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Warning: Configuration file {file_path} not found. Using default configs.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}

if __name__ == "__main__":
    
    
    file_path = ["data/filtered_odometry.csv", "data/filtered_odometry_odo.csv","data/z_test.csv"]
    file_sample_rate = 1000.0
    
    if YAML:
        # Get configurations from file
        configs = load_yaml_config("plotter_config.yaml")
        configs_yaml = yaml.dump(configs) if configs else print("The configuration file is missing.")
        # Select which configuration to use
        config_name = "p_v_state"  # Change this to select different configurations

        # Load all configurations
        all_configs = yaml.safe_load(configs_yaml)
        
        # Check if the selected configuration exists
        if config_name not in all_configs:
            print(f"Configuration '{config_name}' not found. Available configurations: {list(all_configs.keys())}")
            exit(1)
        
        # Extract configuration values
        selected_config = all_configs[config_name]
        plot_num = selected_config['plot_num']
        plot_column = selected_config['plot_column']
        plot_title = selected_config['plot_title']
        
        print(f"Using configuration: {config_name}")

    else:
        # Default configurations
        plot_num = [1, 3]
        plot_column = [['sim_pos_x', 'p.x','filtered_p.x'], 
                       ['sim_pos_y', 'p.y','filtered_p.y'], 
                       ['sim_pos_z', 'p.z','filtered_p.z', 'estimate_z_position']]
        plot_title = ['X_position', 'Y_position', 'Z_position']
    
    # Initialize an empty DataFrame to hold all the data
    data = pd.DataFrame()

    # Process each file path
    for path in file_path:
        ros_data = ROS_Data(path)
        ros_data.load_data()
        ros_data.process_data()
        ros_data = ros_data.get_processed_data()
        # For the first iteration, just assign directly
        if data.empty:
            data = ros_data
        else:
            # For subsequent iterations, concatenate
            data = pd.concat([data, ros_data], axis=1)

    #############################

    # vicon_data = VICON_Data("data/0407_walk_vicon.csv", trigger_name="20250407:Trigger")
    # vicon_data.load_data()
    # LF = restrct(vicon_data.traj_data, 'O1')
    # RF = restrct(vicon_data.traj_data, 'O2')
    # RB = restrct(vicon_data.traj_data, 'O3')
    # LB = restrct(vicon_data.traj_data, 'O4')

    # vicon_position = []
    # vicon_velocity = []
    # last_centroid = [0, 0, 0]
    # for idx in vicon_data.traj_data.index[vicon_data.trigger_index+1:]:
    #     # For each marker, extract its X, Y, Z as a list of floats.
    #     try:
    #         # Parse coordinates safely, replacing empty strings with NaN
    #         def safe_float(val):
    #             if val == '' or pd.isna(val):
    #                 return np.nan
    #             return float(val)
            
    #         LF_point = [safe_float(LF.loc[idx, col]) for col in ['X', 'Y', 'Z']]
    #         RF_point = [safe_float(RF.loc[idx, col]) for col in ['X', 'Y', 'Z']]
    #         RB_point = [safe_float(RB.loc[idx, col]) for col in ['X', 'Y', 'Z']]
    #         LB_point = [safe_float(LB.loc[idx, col]) for col in ['X', 'Y', 'Z']]
        
    #     except Exception as e:
    #         print(f"Error processing frame {idx}: {e}")
    #         continue
        


    #     roll, pitch, yaw, centroid = compute_attitude_from_rect_points(LF_point, RF_point, RB_point, LB_point)
    #     vicon_position.append({'vicon_p_x': centroid[0], 'vicon_p_y': centroid[1], 'vicon_p_z': centroid[2]})
    #     vicon_velocity.append({'vicon_v_x': (centroid[0] - last_centroid[0]), 'vicon_v_y': (centroid[1] - last_centroid[1]), 'vicon_v_z': (centroid[2] - last_centroid[2])})
    #     last_centroid = centroid
        
    # data = pd.concat([data, pd.DataFrame(vicon_position)], axis=1)
    # data = pd.concat([data, pd.DataFrame(vicon_velocity)], axis=1)

    fig, axes = plt.subplots(plot_num[0], plot_num[1], figsize=(15, 10))

    # Check if we have a single subplot or multiple subplots
    if plot_num[0] == 1 and plot_num[1] == 1:
        # For single subplot, convert to array for consistent handling
        axes = np.array([axes])
    else:
        # For multiple subplots, flatten the 2D array
        axes = axes.flatten()

    for i, columns in enumerate(plot_column):
        ax = axes[i]  # Get the current subplot
        title = plot_title[i][0]
        for column in columns:

            plot_data = getattr(data, column)
            # if (column == 'estimate_z_position'):
            #     # Apply low-pass filter to the data
            #     cutoff_freq = 10.0
            #     plot_data = low_pass_filter(plot_data, cutoff_freq, file_sample_rate)

            if (column in ['p.x','filtered_p.x', 'filtered_p.y','p.y','filtered_p.z','p.z']):
                time = np.arange(len(plot_data)) / file_sample_rate
            else:
                time = np.arange(len(plot_data)) / file_sample_rate
            # Plot data on the current subplot
            ax.plot(time, plot_data, label=column)
        
        ax.set_title(title)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('(m)')
        # ax.set_xlim(0, 10)  # Set x-axis limits
        # ax.set_ylim(0.18, 0.2)  # Set y-axis limits
        ax.legend()
        ax.grid(True)

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()
    
