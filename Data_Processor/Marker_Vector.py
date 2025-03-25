def restrct(df, marker):
    """
    Reconstructs marker data into a new DataFrame by extracting columns that start with the given marker.
    The returned DataFrame will have its column names stripped of the marker prefix.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The source DataFrame containing trajectory data.
    marker : str
        The marker identifier (e.g., 'O1') to search for in the column names.
    
    Returns
    -------
    marker_df : pandas.DataFrame
        A DataFrame with columns corresponding to the marker's components (e.g., 'X', 'Y', 'Z').
        You can access the data using attribute access if the column names are valid Python identifiers.
        For example: O1.X['23'].
    """
    # Identify all columns that start with the marker
    marker_cols = [col for col in df.columns if col.startswith(marker)]
    
    # If no matching columns are found, return an empty DataFrame
    if not marker_cols:
        return pd.DataFrame()
    
    # Create a new DataFrame with only the marker's columns
    marker_df = df[marker_cols].copy()
    
    # Rename the columns by stripping off the marker prefix
    new_names = {col: col[len(marker):] for col in marker_cols}
    marker_df.rename(columns=new_names, inplace=True)
    
    return marker_df


# === Example Usage ===
if __name__ == "__main__":
    import pandas as pd
    
    # Create a sample DataFrame with marker columns.
    data = {
        "O1X": [1.0, 1.1, 1.2],
        "O1Y": [2.0, 2.1, 2.2],
        "O1Z": [3.0, 3.1, 3.2],
        "O2X": [4.0, 4.1, 4.2],
        "O2Y": [5.0, 5.1, 5.2],
        "O2Z": [6.0, 6.1, 6.2]
    }
    df = pd.DataFrame(data)
    
    # Reconstruct marker vector for O1.
    O1 = restrct(df, "O1")
    O2 = restrct(df, "O2")
    
    # Access data by component and row label.
    print(O1)
    print(O2.Y)