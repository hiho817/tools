import pandas as pd
import re

class ROS_Data:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.data_process = None

    def load_data(self):
        # Load data from CSV, assuming the first row contains the column titles
        self.data = pd.read_csv(self.file_path, header=0)
        print("Data loaded successfully.")

    def process_data(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        # Example of processing: copy data and fill missing values with 0
        self.data_process = self.data.copy()
        self.data_process.fillna(0, inplace=True)
        print("Data processed successfully.")

    def get_processed_data(self):
        if self.data_process is None:
            self.process_data()
        return self.data_process

class VICON_Data:
    def __init__(self, file_path, trigger_name="trigger"):
        """
        Initializes the VICON_Data class.

        Parameters:
        - file_path: Path to the data file.
        - trigger_name: The name of the trigger marker. Defaults to "trigger" if not provided.
        """
        self.file_path = file_path
        self.trigger_name = trigger_name if trigger_name and trigger_name.strip() != "" else "trigger"
        self.device_frequency = None   # Device frequency from the Devices section (e.g., 1000 Hz)
        self.traj_frequency = 500      # Trajectory frequency (default 500 Hz)
        self.time_between_frames = 1 / self.traj_frequency  # Time between frames (e.g., 0.002 sec)
        self.sub_frame_interval = 0.001  # Interval between sub-frames

        self.raw_data = None       # Raw file content (list of strings)
        self.traj_data = None      # Trajectory data stored as a pandas DataFrame
        self.data_process = None   # (Optional) processed trajectory data
        self.trigger_time = None   # Trigger time based on the trigger marker
        self.forceplates = {}      # Dictionary to store ForcePlate data (key: plate number, value: DataFrame)

    def load_data(self):
        # Read file using 'utf-8-sig' encoding to remove BOM and filter out empty lines.
        with open(self.file_path, 'r', encoding='utf-8-sig') as f:
            self.raw_data = [line.rstrip() for line in f if line.strip() != '']

        # ---------------------------
        # Process Devices Section
        # ---------------------------
        try:
            dev_index = self.raw_data.index("Devices")
        except ValueError:
            print("Devices section not found.")
            dev_index = None

        if dev_index is not None:
            # Get device frequency from the line following "Devices"
            try:
                self.device_frequency = int(self.raw_data[dev_index + 1].strip())
            except ValueError:
                print("Invalid device frequency format in file.")

            # Assume the header row for forceplate labels is at dev_index+2
            dev_header = self.raw_data[dev_index + 2].split(',')
            # Use regex to map each ForcePlate's columns (e.g., 'ForcePlate4')
            forceplate_columns = {}  # key: plate number, value: list of column indices
            for idx, col in enumerate(dev_header):
                match = re.search(r'ForcePlate(\d+)', col)
                if match:
                    plate_num = int(match.group(1))
                    forceplate_columns.setdefault(plate_num, []).append(idx)

            # Helper function: Find first data row (line starting with a digit)
            def find_first_data_row(start):
                for i in range(start, len(self.raw_data)):
                    if self.raw_data[i] and self.raw_data[i][0].isdigit():
                        return i
                return None

            dev_data_start = find_first_data_row(dev_index)
            dev_data = []
            # Extract data rows until "Trajectories" section is reached or end of file.
            for line in self.raw_data[dev_data_start:]:
                if "Trajectories" in line:
                    break
                tokens = line.split(',')
                if tokens[0].isdigit():
                    dev_data.append(tokens)

            # For each detected ForcePlate, extract its data and store as a DataFrame.
            for plate, cols in forceplate_columns.items():
                plate_data = []
                for row in dev_data:
                    # Safely extract columns; use empty string if the column is missing.
                    row_data = [row[i] if i < len(row) else '' for i in cols]
                    plate_data.append(row_data)
                # Use header names from the original header for DataFrame columns.
                col_names = [dev_header[i] for i in cols]
                self.forceplates[plate] = pd.DataFrame(plate_data, columns=col_names)

        # ---------------------------
        # Process Trajectories Section
        # ---------------------------
        try:
            traj_index = self.raw_data.index("Trajectories")
        except ValueError:
            print("Trajectories section not found.")
            traj_index = None

        if traj_index is not None:
            # Get trajectory frequency from the line following "Trajectories"
            try:
                self.traj_frequency = int(self.raw_data[traj_index + 1].strip())
                self.time_between_frames = 1 / self.traj_frequency
            except ValueError:
                print("Invalid trajectory frequency format in file.")

            # Assume file structure:
            #   traj_index+2: Marker names row (e.g., ",,20250324:O1,,,20250324:O2,...")
            #   traj_index+3: Column header row (e.g., "Frame,Sub Frame,X,Y,Z,...")
            #   traj_index+4: Units row (to be skipped)
            marker_names_row = self.raw_data[traj_index + 2].split(',')
            traj_header_line = self.raw_data[traj_index + 3]
            traj_columns = traj_header_line.split(',')

            # Combine the marker names and header row to create new column names.
            # For marker names of the form "20250324:O1", we extract the part after ":".
            new_columns = []
            tmp_marker = ''
            for marker, header in zip(marker_names_row, traj_columns):
                marker = marker.strip()
                header = header.strip()
                if marker:
                    # Split at ":" and take the part after the colon.
                    parts = marker.split(":")
                    # Use the last part as the marker name.
                    tmp_marker = parts[-1]
                # If the header indicates a coordinate (e.g., X, Y, Z), append it to the marker name.
                if header in ["X", "Y", "Z"]:
                    new_columns.append(tmp_marker + header)
                else:
                    # Otherwise, just use the header value (useful for Frame, Sub Frame, etc.)
                    new_columns.append(header)

            # Use the provided trigger name from __init__
            print(f"Using trigger marker: '{self.trigger_name}'")
            trigger_marker_index = None
            # Search for the trigger marker in the marker names row.
            for idx, marker in enumerate(marker_names_row):
                if marker.strip() == self.trigger_name:
                    trigger_marker_index = idx
                    break
            if trigger_marker_index is None:
                print(f"Trigger marker '{self.trigger_name}' not found in the marker names row.")
            else:
                print(f"Trigger marker '{self.trigger_name}' found at column index {trigger_marker_index}.")

            # Helper function: Find the first trajectory data row (line starting with a digit)
            def find_first_traj_data(start):
                for i in range(start, len(self.raw_data)):
                    if self.raw_data[i] and self.raw_data[i][0].isdigit():
                        return i
                return None

            traj_data_start = find_first_traj_data(traj_index+3) # Start from the row after the header
            traj_data_rows = []
            # Extract trajectory data rows until another section header is encountered.
            for line in self.raw_data[traj_data_start:]:
                if line in ["Devices", "Trajectories"]:
                    break
                tokens = line.split(',')
                if tokens[0].isdigit():
                    traj_data_rows.append(tokens)

            # Create a DataFrame for the trajectories using the header row.
            self.traj_data = pd.DataFrame(traj_data_rows, columns=new_columns)
            # Convert 'Frame' and 'Sub Frame' columns to numeric for later computations.
            self.traj_data['Frame'] = pd.to_numeric(self.traj_data['Frame'], errors='coerce')
            self.traj_data['Sub Frame'] = pd.to_numeric(self.traj_data['Sub Frame'], errors='coerce')

            # ---------------------------
            # Determine Trigger Time using the trigger marker
            # ---------------------------
            if trigger_marker_index is not None and trigger_marker_index < len(traj_columns):
                for _, row in self.traj_data.iterrows():
                    # Use iloc to get the value at the trigger marker's column index.
                    val = row.iloc[trigger_marker_index]
                    # Check if the value is valid (not null and not empty).
                    if pd.notnull(val) and val != '':
                        self.trigger_index = _
                        frame = row['Frame']
                        sub_frame = row['Sub Frame']
                        # Calculate trigger time based on frame and sub-frame values.
                        self.trigger_time = (frame - 1) * self.time_between_frames + sub_frame * self.sub_frame_interval
                        break

        print("Data loaded successfully.")

# ---------------------------
# Testing the VICON_Data class
# ---------------------------
if __name__ == "__main__":
    file_path = "data/imu_test_from_vicon.csv"  # Replace with your actual file path
    # Create a VICON_Data object with a specified trigger marker name; if empty, defaults to "trigger"
    vicon = VICON_Data(file_path, trigger_name="20250324:O5")
    vicon.load_data()
    
#     # Display ForcePlate data (each stored as a DataFrame)
#     for plate, df in vicon.forceplates.items():
#         print(f"\nForcePlate {plate} DataFrame:")
#         print(df.head())
    
    # # Display Trajectories DataFrame if available
    # if vicon.traj_data is not None:
    #     print("\nTrajectories DataFrame:")
    #     print(vicon.traj_data.head())
    
#     # Display the computed trigger time
#     print("\nTrigger Time:", vicon.trigger_time)


#Example usage of ros data:
# if __name__ == "__main__":
#     file_path = "data/imu_test_from_imu.csv"  # Replace with your actual file path
#     ros_data = ROS_Data(file_path)
#     ros_data.load_data()
#     ros_data.process_data()
#     processed_data = ros_data.get_processed_data()
