from Data_Processor import ROS_Data
import matplotlib.pyplot as plt
import numpy as np
import yaml

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

# Get configurations from file
configs = load_yaml_config("plotter_config.yaml")
configs_yaml = yaml.dump(configs) if configs else print("The configuration file is missing.")

if __name__ == "__main__":
    
    
    file_path = "data/imu_test_from_imu.csv"  # Replace with your actual file path
    file_sample_rate = 1000  # Replace with your actual sample rate
    
    # Select which configuration to use
    config_name = "trq_ratio_state"  # Change this to select different configurations

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

    ##############
    ros_data = ROS_Data(file_path)
    ros_data.load_data()
    ros_data.process_data()
    ros_data = ros_data.get_processed_data()

    # Create a single figure with 2x2 subplots
    fig, axes = plt.subplots(plot_num[0], plot_num[1], figsize=(15, 10))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    for i, columns in enumerate(plot_column):
        ax = axes[i]  # Get the current subplot
        title = plot_title[i][0]
        for column in columns:
            # Check if there are "+,-,/,*" in column
            operators = ['+', '-', '*', '/']
            if any(op in column for op in operators):
                # Parse the expression (e.g., "state_trq_r_a/state_trq_l_a")
                for op in operators:
                    if op in column:
                        left, right = column.split(op)
                        left_data = getattr(ros_data, left.strip())
                        right_data = getattr(ros_data, right.strip())
                        
                        if op == '+':
                            data = left_data + right_data
                        elif op == '-':
                            data = left_data - right_data
                        elif op == '*':
                            data = left_data * right_data
                        elif op == '/':
                            # Avoid division by zero
                            data = np.divide(left_data, right_data, out=np.zeros_like(left_data), where=right_data!=0)
                        break
            else:
                # Get data from ros_data using attribute access
                data = getattr(ros_data, column)

            # Apply low-pass filter to the data
            cutoff_freq = 10
            filtered_data = low_pass_filter(data, cutoff_freq, file_sample_rate)

            # Calculate time axis
            time = np.arange(len(data)) / file_sample_rate
            # Plot data on the current subplot
            ax.plot(time, filtered_data, label=column)
        
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()
    
