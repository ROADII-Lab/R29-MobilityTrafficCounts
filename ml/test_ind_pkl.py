import pandas as pd
import pickle
import os

def load_and_export_to_csv(pkl_filepath):
  """
  Loads a large table of data from a pickle file and exports it to a CSV file.

  Args:
      pkl_filepath (str): The filepath of the pickle file.

  Raises:
      FileNotFoundError: If the pickle file does not exist.
  """
  # Check if the file exists
  if not os.path.isfile(pkl_filepath):
    raise FileNotFoundError(f"Pickle file not found: {pkl_filepath}")

  # Extract the base path and filename (without extension)
  base_path, filename = os.path.split(pkl_filepath)

  # Construct a descriptive CSV filename based on the original filename
  csv_filename = f"{filename}_data.csv"
  csv_filepath = os.path.join(base_path, csv_filename)

  # Load data using pandas.read_pickle (optimized for large dataframes)
  try:
    df = pd.read_pickle(pkl_filepath)
  except (IOError, pickle.UnpicklingError) as e:
    print(f"Error loading pickle file: {e}")
    return

  # Export data to CSV
  df.to_csv(csv_filepath, index=False)  # Save without index column

  print(f"Successfully exported data from {pkl_filepath} to {csv_filepath}")


def read_pickle_to_csv(pkl_filepath, output_filepath, num_rows=10000):
  """
  Reads the first 'num_rows' from a pickle file and exports them to a CSV file
  at the specified output location.

  Args:
      pkl_filepath (str): The filepath of the pickle file.
      output_filepath (str): The desired filepath for the output CSV file.
      num_rows (int, optional): The number of rows to read from the pickle file. Defaults to 1000.

  Raises:
      FileNotFoundError: If the pickle file does not exist.
  """
  # Check if the file exists
  if not os.path.isfile(pkl_filepath):
    raise FileNotFoundError(f"Pickle file not found: {pkl_filepath}")

  # Load data using pandas.read_pickle
  try:
    data = pd.read_pickle(pkl_filepath)
  except (IOError, pickle.UnpicklingError) as e:
    print(f"Error loading pickle file: {e}")
    return

  # Ensure num_rows is within data size
  if num_rows > len(data):
    print(f"Warning: Requested number of rows ({num_rows}) exceeds data size ({len(data)}). Reading all data.")
    num_rows = len(data)

  # Read the first 'num_rows' rows
  data_subset = data.head(num_rows)

  # Export data to CSV
  data_subset.to_csv(output_filepath, index=False)  # Save without index column

  print(f"Successfully exported first {num_rows} rows from {pkl_filepath} to {output_filepath}")

# Example usage
pkl_filepath = r'C:\Users\Michael.Barzach\Documents\GitHub\R29-MobilityTrafficCounts\data\TMAS_Class_Clean_2021.pkl'
output_filepath = r"C:\Users\Michael.Barzach\Documents\GitHub\R29-MobilityTrafficCounts\data\tmas2021_10000.csv"  # Choose your desired output location

read_pickle_to_csv(pkl_filepath, output_filepath)
# Example usage
#load_and_export_to_csv(pkl_filepath)