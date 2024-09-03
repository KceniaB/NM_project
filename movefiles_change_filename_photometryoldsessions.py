#%% 
""" KB 19Aug2024 """
"""
move files to corresponding date folders - works! 
"""
import os
import shutil
import re
from datetime import datetime

# Define the source path and destination base path
source_path = '/mnt/h0/kb/data/external_drive/HCW_S2_23082021/May2022'
destination_base_path = '/mnt/h0/kb/data/external_drive/HCW_S2_23082021/Sorted_Files'

# Ensure the destination base path exists
os.makedirs(destination_base_path, exist_ok=True)

# List all files in the directory
all_files = os.listdir(source_path)

# Filter out only CSV files and sort them alphabetically
csv_files = sorted([file for file in all_files if file.endswith('.csv')])

# Function to extract date from the file name
def extract_date(file_name):
    try:
        date_match = re.search(r'(\d{1,2}[A-Za-z]{3}\d{4})', file_name)
        if date_match:
            date_str = date_match.group(1)
            return datetime.strptime(date_str, '%d%b%Y').strftime('%Y-%m-%d')
        return None
    except Exception as e:
        print(f"Error extracting date from file {file_name}: {e}")
        return None

# Organize files into folders based on the extracted date
for file in csv_files:
    # Extract the date
    date_str = extract_date(file)
    
    if date_str:
        # Create a folder for the date if it does not exist
        date_folder = os.path.join(destination_base_path, date_str)
        os.makedirs(date_folder, exist_ok=True)
        
        # Move the file to the respective folder
        source_file = os.path.join(source_path, file)
        destination_file = os.path.join(date_folder, file)
        shutil.move(source_file, destination_file)
        print(f"Moved {file} to {date_folder}")

print("File organization complete.")





#%% 
import os
import pandas as pd
import re

# Define the source path
source_path = '/mnt/h0/kb/data/external_drive/HCW_S2_23082021/May2022'

# List all files in the directory
all_files = os.listdir(source_path)

# Filter out only CSV files and sort them alphabetically
csv_files = sorted([file for file in all_files if file.endswith('.csv')])

# Define a function to extract information from the file name
def extract_info(file_name):
    try:
        # Extract Mouse (letters and numbers between first underscore and the second underscore)
        mouse_match = re.search(r'DI\d+_(.*?)_', file_name)
        mouse = mouse_match.group(1) if mouse_match else None
        
        # Extract Date (format: 28Jun2022)
        date_match = re.search(r'(\d{1,2}[A-Za-z]{3}\d{4})', file_name)
        date = pd.to_datetime(date_match.group(1), format='%d%b%Y').strftime('%Y-%m-%d') if date_match else None
        
        # Extract Session number (digit after the date)
        session_match = re.search(r'_(\d+)\.csv', file_name)
        session_number = session_match.group(1) if session_match else '0'
        
        # Digital input (first digit after 'DI')
        digital_input_match = re.search(r'DI(\d+)_', file_name)
        digital_input = digital_input_match.group(1) if digital_input_match else None
        
        return mouse, date, session_number, digital_input
    
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None, None, None, None

# Create a pandas DataFrame with extracted information
df_csv_files = pd.DataFrame(csv_files, columns=['CSV Files'])

# Apply the function to extract Mouse, Date, Session number, and Digital input
df_csv_files[['Mouse', 'Date', 'Session_number', 'Digital_input']] = df_csv_files['CSV Files'].apply(lambda x: pd.Series(extract_info(x)))

# Display the DataFrame
print(df_csv_files) 

# %%
file_path = '/home/ibladmin/Downloads/processed_files.xlsx'
df = pd.read_excel(file_path)




