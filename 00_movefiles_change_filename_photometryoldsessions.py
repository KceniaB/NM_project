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









#%%
""" 
for M1 M2 M3 M4 
KB 26-Aug-2024
"""
import os
import shutil
import pandas as pd

root_dir = "/mnt/h0/kb/data/external_drive/HCW_S2_23082021/Sorted_Files2"
df2=df1[0:5]
# Iterate over each row in the DataFrame
for index, row in df1.iterrows():
    # Get the date, nphfile, and bncfile for the current row
    date = row['date']
    nphfile = row['photometryfile']
    bncfile = row['bncfile']
    
    # Create the directory for the current date
    date_dir = os.path.join(root_dir, date)
    os.makedirs(date_dir, exist_ok=True)
    
    # Copy the nphfile to the date directory
    shutil.copy(nphfile, date_dir)
    
    # Copy the bncfile to the date directory
    shutil.copy(bncfile, date_dir)

print("Files copied successfully!")






"""
for D1 D2 D3 etc 
KB 26-Aug-2024 
""" 

source_path = "/mnt/h0/kb/data/external_drive/HCW_S2_23082021/DA/"
all_files = os.listdir(source_path)
all_files2 = sorted([file for file in all_files])

import pandas as pd
import re
from datetime import datetime  # Import datetime module

# Filter files that start with 'TCW'
tcw_files = [f for f in all_files2 if f.startswith('TCW')]

# Function to extract date from filename
def extract_date(filename):
    # This regex pattern matches dates like 28Jan2022
    match = re.search(r'(\d{2})([A-Za-z]{3})(\d{4})', filename)
    if match:
        day, month_str, year = match.groups()
        # Convert month abbreviation to a number and format date
        date_str = f"{day}{month_str}{year}"
        date_obj = datetime.strptime(date_str, '%d%b%Y')
        return date_obj.strftime('%Y-%m-%d')
    return None

# Create DataFrame
df = pd.DataFrame({
    'Filename': tcw_files,
    'Date': [extract_date(f) for f in tcw_files]
})

# Display the DataFrame
print(df)