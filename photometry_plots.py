#%% 
# KB 20240606
# load photometry (raw & processed) files
# load psth files
# load behav files
# plot 

#%% ##################################################################################################################
# 1. import
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
# from functions_nm import load_trials 
# import iblphotometry.kcenia as kcenia 
import neurodsp.utils 
from pathlib import Path
# import iblphotometry.plots
# import iblphotometry.dsp 
from brainbox.io.one import SessionLoader
# import functions_nm 
import scipy.signal
import ibllib.plots 
import re
from datetime import datetime 

from one.api import ONE #always after the imports 
one = ONE()
#######################################################################################################################
# %% ##################################################################################################################
# Example usage
################################################
""" CHANGE HERE """ 
path = '/mnt/h0/kb/code_kcenia/photometry_files/' 
prefix = 'trialstable_' 
EVENT_NAME = "feedback_times" 
################################################ 
# 2. read the files - 1 file
def list_files_in_folder(folder_path, prefix):
    try:
        # Get the list of all files and directories in the specified folder
        files = os.listdir(folder_path)
        # Filter the list to include only files that start with the specified prefix
        files = [f for f in files if os.path.isfile(os.path.join(folder_path, f)) and f.startswith(prefix)]
        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return [] 

def get_eid(mouse,date): 
    eids = one.search(subject=mouse, date=date) 
    eid = eids[0]
    ref = one.eid2ref(eid)
    print(eid)
    print(ref) 
    try:
        # Try to load the trials directly
        a = one.load_object(eid, 'trials')
        trials = a.to_df()
    except Exception as e:
        # If loading fails, use the alternative method
        print("Failed to load trials directly. Using alternative method...")
        session_path_behav = f'/home/kceniabougrova/Documents/nph/Behav_2024Mar20/{rec.mouse}/{rec.date}/001/'
        df_alldata = extract_all(session_path_behav)
        table_data = df_alldata[0]['table']
        trials = pd.DataFrame(table_data) 
    return eid, trials 

psth_idx_1 = np.load(path+"psthidx_feedback_times_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.npy") 
behav_1 = pd.read_parquet(path+"trialstable_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 

indices = behav_1["feedbackType"]==1 #pick the feedback == 1
indices_incorrect = behav_1["feedbackType"]==-1 #pick the feedback == 1

# 3. plot it 
plt.plot(psth_idx_1, linewidth=0.2, color='gray', alpha=0.15)
feedback_correct = psth_idx_1[:,indices]
plt.plot(feedback_correct, linewidth=0.2, color='blue', alpha=0.15)
feedback_incorrect = psth_idx_1[:,indices_incorrect] 
plt.plot(feedback_incorrect, linewidth=0.2, color='red', alpha=0.15)
plt.axvline(x=30)
# plt.xlim(15, 55) 
plt.ylim(-0.02, 0.06) 
plt.show() 

plt.figure(figsize=(10,10))
sns.heatmap(feedback_correct.T)
plt.axvline(x=30) 
plt.show()
##################################################################################################################
#%% ##################################################################################################################
# 4. read the files - multiple files 
#    concatenate the data 

def list_files_in_folder(path, prefix):
    return [f for f in os.listdir(path) if f.startswith(prefix)]

def extract_trialstable_key(filename):
    pattern = re.compile(r'trialstable_(ZFM-\d+)_(\d{4}-\d{2}-\d{2})_(\d+)_')
    match = pattern.search(filename)
    if match:
        id_part = match.group(1)
        date_part = match.group(2)
        region_number = int(match.group(3))
        date_obj = datetime.strptime(date_part, '%Y-%m-%d')
        return id_part, date_obj, region_number
    return None

def extract_psthidx_key(filename):
    pattern = re.compile(r'psthidx_feedback_times_(.+?)_(\d{4}-\d{2}-\d{2})_(\d+)_')
    match = pattern.search(filename)
    if match:
        name_part = match.group(1)
        date_part = match.group(2)
        region_number = int(match.group(3))
        date_obj = datetime.strptime(date_part, '%Y-%m-%d')
        return name_part, date_obj, region_number
    return None

# Define paths
trialstable_path = '/mnt/h0/kb/code_kcenia/photometry_files/' 
psthidx_path = '/mnt/h0/kb/code_kcenia/photometry_files/' 

# List and sort trialstable files
trialstable_prefix = 'trialstable_'
trialstable_files = list_files_in_folder(trialstable_path, trialstable_prefix)
sorted_trialstable_files = sorted(trialstable_files, key=extract_trialstable_key)

# List and sort psthidx_feedback files
psthidx_prefix = 'psthidx_feedback_times_'
psthidx_files = list_files_in_folder(psthidx_path, psthidx_prefix)
sorted_psthidx_files = sorted(psthidx_files, key=extract_psthidx_key)

# Create dictionaries to hold the loaded data
trialstable_data = {}
psthidx_data = {}

# Load trialstable data
for file_name in sorted_trialstable_files:
    key = extract_trialstable_key(file_name)
    if key:
        name, date, region = key
        var_name_b = f"trialstable_{name}_{date.strftime('%Y_%m_%d')}_{region}"
        data_array_b = pd.read_parquet(os.path.join(trialstable_path, file_name))
        trialstable_data[(name, date, region)] = data_array_b
        print(f"Loaded DataFrame: {var_name_b}") 

# Load psthidx_feedback data
for file_name in sorted_psthidx_files:
    key = extract_psthidx_key(file_name)
    if key:
        name, date, region = key
        var_name = f"psth_idx_{name}_{date.strftime('%Y_%m_%d')}_{region}"
        data_array = np.load(os.path.join(psthidx_path, file_name))
        psthidx_data[(name, date, region)] = data_array
        print(f"Loaded numpy array: {var_name}")

# Find matching keys based on mouse, date, and region number
matching_keys = set(trialstable_data.keys()) & set(psthidx_data.keys())

# Concatenate matching DataFrames and numpy arrays
behav_multiple = [trialstable_data[key] for key in matching_keys]
psth_multiple = [pd.DataFrame(psthidx_data[key].T) for key in matching_keys]

if behav_multiple:
    behav_concat = pd.concat(behav_multiple).reset_index(drop=True)
    print("Concatenated behavior DataFrame:")
    print(behav_concat.head())

if psth_multiple:
    psth_concat = pd.concat(psth_multiple).reset_index(drop=True)
    print("Concatenated PSTH DataFrame:")
    print(psth_concat.head()) 
    ##################################################################################################################
# %% ##################################################################################################################
# 5. plot the data 

############################
# 5. a) feedbackType 
indices = behav_concat["feedbackType"]==1 #pick the feedback == 1
indices_incorrect = behav_concat["feedbackType"]==-1 #pick the feedback == 1
psth_concat_T = psth_concat.T 

plt.plot(psth_concat_T, linewidth=0.2, color='gray', alpha=0.15)
feedback_correct = psth_concat[indices] 
plt.plot(feedback_correct.T, linewidth=0.2, color='blue', alpha=0.15)
feedback_incorrect = psth_concat[indices_incorrect] 
plt.plot(feedback_incorrect.T, linewidth=0.2, color='red', alpha=0.15)
plt.axvline(x=30)
# plt.xlim(15, 55) 
plt.ylim(-0.02, 0.06) 
plt.show() 

plt.figure(figsize=(10,10))
sns.heatmap(feedback_correct)
plt.axvline(x=30) 
plt.show()

#%% 
############################
# 5. b) contrast  
indices_100 = behav_concat["allContrasts"]==1 
indices_25 = behav_concat["allContrasts"]==0.25 
indices_12 = behav_concat["allContrasts"]==0.125
indices_6 = behav_concat["allContrasts"]==0.0625
indices_0 = behav_concat["allContrasts"]==0.0 

contrast_100 = psth_concat[indices_100] 
plt.plot(contrast_100.T, linewidth=0.2, color='blue', alpha=0.15)
contrast_25 = psth_concat[indices_25] 
plt.plot(contrast_25.T, linewidth=0.2, color='red', alpha=0.15) 
contrast_12 = psth_concat[indices_12] 
plt.plot(contrast_12.T, linewidth=0.2, color='green', alpha=0.15)
contrast_6 = psth_concat[indices_6] 
plt.plot(contrast_6.T, linewidth=0.2, color='orange', alpha=0.15) 
contrast_0 = psth_concat[indices_0] 
plt.plot(contrast_0.T, linewidth=0.2, color='purple', alpha=0.15)
plt.axvline(x=30, linewidth=1, color='black', linestyle='dashed')
plt.legend() 
# plt.xlim(15, 55) 
plt.ylim(-0.02, 0.06) 
plt.show() 

plt.figure(figsize=(10,10))
sns.heatmap(contrast_100)
sns.heatmap(contrast_25) 

plt.axvline(x=30) 
plt.show()

#%%
############################
# 5. c) probabilityLeft   
indices_50 = behav_concat["probabilityLeft"]==0.5
indices_20 = behav_concat["probabilityLeft"]==0.2
indices_80 = behav_concat["probabilityLeft"]==0.8

probL_50 = psth_concat[indices_50] 
plt.plot(probL_50.T, linewidth=0.2, color='blue', alpha=0.15)
probL_20 = psth_concat[indices_20] 
plt.plot(probL_20.T, linewidth=0.2, color='red', alpha=0.15) 
probL_80 = psth_concat[indices_80] 
plt.plot(probL_80.T, linewidth=0.2, color='green', alpha=0.15)
contrast_6 = psth_concat[indices_6] 
plt.plot(contrast_6.T, linewidth=0.2, color='orange', alpha=0.15) 
contrast_0 = psth_concat[indices_0] 
plt.plot(contrast_0.T, linewidth=0.2, color='purple', alpha=0.15)
plt.axvline(x=30, linewidth=1, color='black', linestyle='dashed')
plt.legend() 
# plt.xlim(15, 55) 
plt.ylim(-0.02, 0.06) 
plt.show() 

plt.figure(figsize=(10,10))
sns.heatmap(probL_50)
sns.heatmap(probL_20) 
sns.heatmap(probL_80)

plt.axvline(x=30) 
plt.show()








#%% 
# Z 
# Z. a) normalize 
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())
d = min_max_normalize(psth_concat.T)
psth_concat_T = d.T 
