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
import ibldsp.utils 
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

#%% 
# READ NPH DATA 
""" CHANGE HERE """ 
# path = '/home/kceniabougrova/Documents/results_for_OW/' 
path = '/mnt/h0/kb/code_kcenia/photometry_files/' 
prefix = 'demux_' 

nphca_1 = pd.read_parquet(path+"demux_nph_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 

df = nphca_1

#%% 
#######################################################################################################################
# Figure 1. 
# Calculate the moving average of calcium
window_size = 250  # You can adjust this size based on your needs
df['calcium_moving_avg'] = df['calcium'].rolling(window=window_size).mean()
# window_size_2 = 1500  # You can adjust this size based on your needs
# df['calcium_moving_avg_2'] = df['calcium'].rolling(window=window_size_2).mean()

# Define width ratios for subplots
width_ratios = [4, 1]
FONTSIZE_1 = 30
FONTSIZE_2 = 25
FONTSIZE_3 = 20  
FONTSIZE_4 = 18  

# Create subplots
fig, axes = plt.subplots(3, 2, figsize=(18, 21), gridspec_kw={'width_ratios': width_ratios}, dpi=300)
fig.suptitle('Isosbestic and GCaMP traces', fontsize=FONTSIZE_1)  # Increase main title font size

# Define start and end indices for x-axis limits
start_index = df.times[int(len(df.times)/2)]
end_index = df.times[int(len(df.times)/2)+250]

# raw_isosbestic and raw_calcium
ax1 = axes[0, 0]
ax2 = ax1.twinx()
sns.lineplot(ax=ax1, x=df['times'], y=df['raw_isosbestic'], linewidth=0.1, color="#9d4edd")
sns.lineplot(ax=ax2, x=df['times'], y=df['raw_calcium'], linewidth=0.1, color="#43aa8b")
ax1.set_title("raw_isosbestic and raw_calcium", fontsize=FONTSIZE_2)
ax1.set_xlabel('Time', fontsize=FONTSIZE_4)
ax1.set_ylabel('raw_isosbestic', fontsize=FONTSIZE_4, color="#9d4edd")
ax2.set_ylabel('raw_calcium', fontsize=FONTSIZE_4, color="#43aa8b")
ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#9d4edd")
ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#43aa8b")

# Zoomed raw_isosbestic and raw_calcium
ax3 = axes[0, 1]
ax4 = ax3.twinx()
sns.lineplot(ax=ax3, x=df['times'], y=df['raw_isosbestic'], linewidth=2, color="#9d4edd")
sns.lineplot(ax=ax4, x=df['times'], y=df['raw_calcium'], linewidth=2, color="#43aa8b")
ax3.set_xlim(start_index, end_index)
ax3.set_title("Zoomed raw_isos and raw_calcium", fontsize=FONTSIZE_2)
ax3.set_xlabel('Time', fontsize=FONTSIZE_4)
ax3.set_ylabel('raw_isosbestic', fontsize=FONTSIZE_4, color="#9d4edd")
ax4.set_ylabel('raw_calcium', fontsize=FONTSIZE_4, color="#43aa8b")
ax3.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#9d4edd")
ax4.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#43aa8b")

# Calcium
ax5 = axes[1, 0]
sns.lineplot(ax=ax5, x=df['times'], y=df['calcium'], linewidth=0.1, color="#0081a7")
ax5.set_title("calcium", fontsize=FONTSIZE_2)
ax5.set_xlabel('Time', fontsize=FONTSIZE_4)
ax5.set_ylabel('calcium', fontsize=FONTSIZE_4, color="#0081a7")
ax5.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#0081a7")

# Zoomed calcium
ax6 = axes[1, 1]
sns.lineplot(ax=ax6, x=df['times'], y=df['calcium'], linewidth=2, color="#0081a7")
ax6.set_xlim(start_index, end_index)
ax6.set_title("Zoomed calcium", fontsize=FONTSIZE_2)
ax6.set_xlabel('Time', fontsize=FONTSIZE_4)
ax6.set_ylabel('calcium', fontsize=FONTSIZE_4, color="#0081a7")
ax6.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#0081a7")

# Moving average of calcium
ax7 = axes[2, 0]
sns.lineplot(ax=ax7, x=df['times'], y=df['calcium_moving_avg'], linewidth=1, color="#f4a261")
# sns.lineplot(ax=ax7, x=df['times'], y=df['calcium_moving_avg_2'], linewidth=1, color="#c75000")
ax7.set_title("Moving Average of Calcium", fontsize=FONTSIZE_2)
ax7.set_xlabel('Time', fontsize=FONTSIZE_4)
ax7.set_ylabel('Calcium Moving Average', fontsize=FONTSIZE_4, color="#f4a261")
ax7.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#f4a261") 

# Zoomed calcium
ax8 = axes[2, 1]
sns.lineplot(ax=ax8, x=df['times'], y=df['calcium_moving_avg'], linewidth=2, color="#f4a261")
# sns.lineplot(ax=ax8, x=df['times'], y=df['calcium_moving_avg_2'], linewidth=2, color="#c75000")
ax8.set_xlim(start_index, end_index)
ax8.set_title("Zoomed moving average calcium", fontsize=FONTSIZE_2)
ax8.set_xlabel('Time', fontsize=FONTSIZE_4)
ax8.set_ylabel('calcium mov average', fontsize=FONTSIZE_4, color="#f4a261")
ax8.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#f4a261")


plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the main title
plt.show() 


# %% ##################################################################################################################
# reaction time = 1stMov - stimOn
# response time = response_times - stimOn
# movement time = total wheel movement time (excluding when wheel was hold still) #new var(?) - to explore if the mouse was moving more anxiously the wheel, if there was almost no mov... 


# %% ##################################################################################################################
# Fig 2. Heatmap with all trials in 1 session & lineplot of all trials in 1 session aligned to an event 
"""
change the x axix to seconds; FR=30FPS
to add:     highlight the selected trial
            keep the means, add a line for the selected trial to appear in ax2 
remove the legends? 
"""
path = '/mnt/h0/kb/code_kcenia/photometry_files/' 
psth_idx_1 = np.load(path+"psthidx_feedback_times_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.npy") 
behav_1 = pd.read_parquet(path+"trialstable_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 
nphca_1 = pd.read_parquet(path+"demux_nph_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 

# Indices for correct and incorrect feedback
indices = behav_1["feedbackType"] == 1
indices_incorrect = behav_1["feedbackType"] == -1
feedback_correct = psth_idx_1[:, indices]
feedback_incorrect = psth_idx_1[:, indices_incorrect] 

# Calculate the mean for feedback_correct and feedback_incorrect
mean_feedback_correct = np.mean(feedback_correct, axis=1)
mean_feedback_incorrect = np.mean(feedback_incorrect, axis=1)

fig = plt.figure(figsize=(6, 15))
gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 2], hspace=0.1)

# First subplot: heatmap
ax1 = fig.add_subplot(gs[0])
sns.heatmap(feedback_correct.T, ax=ax1, cbar=False)
ax1.axvline(x=30.5, linestyle='dashed', color='white')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Second subplot: line plot
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(feedback_correct, linewidth=0.2, color='#0077b6', alpha=0.1)
ax2.plot(feedback_incorrect, linewidth=0.2, color='#ff5858', alpha=0.1)
# Plot the mean lines
ax2.plot(mean_feedback_correct, linewidth=2, color='#0077b6', label='Mean Correct')
ax2.plot(mean_feedback_incorrect, linewidth=2, color='#ff5858', label='Mean Inc')
ax2.axvline(x=30.5, linestyle='dashed', color='black')
# Add legend
ax2.legend()
# Remove right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# Set labels
ax2.set_xlabel('Time since event')
ax2.set_ylabel('Signal')

# Show the plot
plt.show() 


# %% ##################################################################################################################
# Fig 3. Get single line for signal during a trial 
"""
add color: #0077b6 if correct, #ff5858 if inc 
"""
path = '/mnt/h0/kb/code_kcenia/photometry_files/' 
psth_idx_1 = np.load(path+"psthidx_feedback_times_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.npy") 
behav_1 = pd.read_parquet(path+"trialstable_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 
nphca_1 = pd.read_parquet(path+"demux_nph_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 

nph_singletrial = [] 
for i in range(len(behav_1.trialNumber)): 
    # Define the interval for the current trial
    interval_start = behav_1.intervals_0[i]
    interval_end = behav_1.intervals_1[i] 
    
    # Use boolean indexing to find the times within the current interval
    mask = (nphca_1.times >= interval_start) & (nphca_1.times <= interval_end)
    
    # Append the corresponding calcium values to the list
    nph_singletrial.append(nphca_1.calcium[mask]) 

# Example for trial 1
# Get the times corresponding to the first trial's calcium values 
index = 0 #select trial (trial = index+1)
times_trial_1 = nphca_1.times[(nphca_1.times >= behav_1.intervals_0[index]) & (nphca_1.times <= behav_1.intervals_1[index])]

plt.figure(figsize=(10, 5))
plt.plot(times_trial_1, nph_singletrial[index], label='Calcium Signal', color='#0077b6')
plt.axvline(behav_1.stimOnTrigger_times[index], color='blue', linestyle='dashed', label='Stim Onset') 
plt.axvline(behav_1.firstMovement_times[index], color='green', linestyle='dashed', label='First Mov') 
plt.axvline(behav_1.feedback_times[index], color='red', linestyle='dashed', label='Feedback')
plt.xlabel('Time')
plt.ylabel('Calcium Signal') 
plt.title(f"Trial number: {index + 1}")
plt.legend()
plt.show() 


# %% ##################################################################################################################
# Fig 4. Get single line for signal during a trial 
"""
to plot: 
    split by block: 
        for correct and incorrect: 
            time from stim on 
            time from feedback 
    split by contrast: 
        for correct and incorrect: 
            time from stim on 
            time from feedback 
    split by left and right: #maybe for c and inc 
            time from 1st mov 
    split by correct and incorrect in general #maybe not needed because it will appear in the plots above 
            for stim on, 1st mov and feednack time 
plot all lines as in the code below, or only the avg line with the error 
add a "smart" ylim? 
"""
path = '/mnt/h0/kb/code_kcenia/photometry_files/' 
psth_idx_1 = np.load(path+"psthidx_feedback_times_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.npy") 
behav_1 = pd.read_parquet(path+"trialstable_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 
nphca_1 = pd.read_parquet(path+"demux_nph_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 

indices_50 = behav_1["probabilityLeft"] == 0.5
indices_20 = behav_1["probabilityLeft"] == 0.2 
indices_80 = behav_1["probabilityLeft"] == 0.8 

block_50 = psth_idx_1[:, indices_50]
block_20 = psth_idx_1[:, indices_20] 
block_80 = psth_idx_1[:, indices_80]

mean_50 = np.mean(block_50, axis=1)
mean_20 = np.mean(block_20, axis=1) 
mean_80 = np.mean(block_80, axis=1)

fig = plt.figure(figsize=(5, 5))
ax2 = fig.add_subplot(111)
# # Plot all lines OR DO ONLY AVG AND SEM
# ax2.plot(block_50, linewidth=0.2, color='gray', alpha=0.1)
# ax2.plot(block_20, linewidth=0.2, color='#ff6d00', alpha=0.1)
# ax2.plot(block_80, linewidth=0.2, color='teal', alpha=0.1)
# Plot the mean lines
ax2.plot(mean_50, linewidth=2, color='gray', label='Mean 0.5') 
ax2.plot(mean_20, linewidth=2, color='#ff6d00', label='Mean 0.2') 
ax2.plot(mean_80, linewidth=2, color='teal', label='Mean 0.8')
# Add a vertical line 
ax2.axvline(x=30.5, linestyle='dashed', color='black')
# Add legend
ax2.legend()
# Remove right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# Set labels
ax2.set_xlabel('Time since event')
ax2.set_ylabel('Signal')
# Show the plot
plt.show() 

#%%
# All contrasts
indices_100 = behav_1["allContrasts"] == 1 
indices_25 = behav_1["allContrasts"] == 0.25
indices_12 = behav_1["allContrasts"] == 0.125
indices_6 = behav_1["allContrasts"] == 0.0625
indices_0 = behav_1["allContrasts"] == 0

contrast_100 = psth_idx_1[:, indices_100]
contrast_25 = psth_idx_1[:, indices_25]
contrast_12 = psth_idx_1[:, indices_12]
contrast_6 = psth_idx_1[:, indices_6]
contrast_0 = psth_idx_1[:, indices_0]

mean_100 = np.mean(contrast_100, axis=1)
mean_25 = np.mean(contrast_25, axis=1)
mean_12 = np.mean(contrast_12, axis=1)
mean_6 = np.mean(contrast_6, axis=1)
mean_0 = np.mean(contrast_0, axis=1)

fig = plt.figure(figsize=(5, 5))
ax2 = fig.add_subplot(111)

# Plot the mean lines
ax2.plot(mean_100, linewidth=2, color='black', label='Contrast 1', alpha=1)
ax2.plot(mean_25, linewidth=2, color='black', label='Contrast 0.25', alpha=0.7)
ax2.plot(mean_12, linewidth=2, color='black', label='Contrast 0.12', alpha=0.4) 
ax2.plot(mean_6, linewidth=2, color='black', label='Contrast 0.06', alpha=0.2)
ax2.plot(mean_0, linewidth=2, color='#bcd4e6', label='Contrast 0', alpha=1)

# Add a vertical line 
ax2.axvline(x=30.5, linestyle='dashed', color='black') 

# Add legend
ax2.legend() 

# Remove right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# Set labels
ax2.set_xlabel('Time since event')
ax2.set_ylabel('Signal')

plt.show() 




# %% ##################################################################################################################
# Example usage
################################################
""" CHANGE HERE """ 
# path = '/home/kceniabougrova/Documents/results_for_OW/' 
path = '/mnt/h0/kb/code_kcenia/photometry_files/' 
# prefix = 'trialstable_' 
prefix = 'demux_' 
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
nphca_1 = pd.read_parquet(path+"demux_nph_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 

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

#%% 
# READ NPH DATA 
""" CHANGE HERE """ 
# path = '/home/kceniabougrova/Documents/results_for_OW/' 
path = '/mnt/h0/kb/code_kcenia/photometry_files/' 
prefix = 'demux_' 

nphca_1 = pd.read_parquet(path+"demux_nph_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 

df = nphca_1

#%% 
#######################################################################################################################
# Figure 1. 
# Calculate the moving average of calcium
window_size = 250  # You can adjust this size based on your needs
df['calcium_moving_avg'] = df['calcium'].rolling(window=window_size).mean()
# window_size_2 = 1500  # You can adjust this size based on your needs
# df['calcium_moving_avg_2'] = df['calcium'].rolling(window=window_size_2).mean()

# Define width ratios for subplots
width_ratios = [4, 1]
FONTSIZE_1 = 30
FONTSIZE_2 = 25
FONTSIZE_3 = 20  
FONTSIZE_4 = 18  

# Create subplots
fig, axes = plt.subplots(3, 2, figsize=(18, 21), gridspec_kw={'width_ratios': width_ratios}, dpi=300)
fig.suptitle('Isosbestic and GCaMP traces', fontsize=FONTSIZE_1)  # Increase main title font size

# Define start and end indices for x-axis limits
start_index = df.times[int(len(df.times)/2)]
end_index = df.times[int(len(df.times)/2)+250]

# raw_isosbestic and raw_calcium
ax1 = axes[0, 0]
ax2 = ax1.twinx()
sns.lineplot(ax=ax1, x=df['times'], y=df['raw_isosbestic'], linewidth=0.1, color="#9d4edd")
sns.lineplot(ax=ax2, x=df['times'], y=df['raw_calcium'], linewidth=0.1, color="#43aa8b")
ax1.set_title("raw_isosbestic and raw_calcium", fontsize=FONTSIZE_2)
ax1.set_xlabel('Time', fontsize=FONTSIZE_4)
ax1.set_ylabel('raw_isosbestic', fontsize=FONTSIZE_4, color="#9d4edd")
ax2.set_ylabel('raw_calcium', fontsize=FONTSIZE_4, color="#43aa8b")
ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#9d4edd")
ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#43aa8b")

# Zoomed raw_isosbestic and raw_calcium
ax3 = axes[0, 1]
ax4 = ax3.twinx()
sns.lineplot(ax=ax3, x=df['times'], y=df['raw_isosbestic'], linewidth=2, color="#9d4edd")
sns.lineplot(ax=ax4, x=df['times'], y=df['raw_calcium'], linewidth=2, color="#43aa8b")
ax3.set_xlim(start_index, end_index)
ax3.set_title("Zoomed raw_isos and raw_calcium", fontsize=FONTSIZE_2)
ax3.set_xlabel('Time', fontsize=FONTSIZE_4)
ax3.set_ylabel('raw_isosbestic', fontsize=FONTSIZE_4, color="#9d4edd")
ax4.set_ylabel('raw_calcium', fontsize=FONTSIZE_4, color="#43aa8b")
ax3.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#9d4edd")
ax4.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#43aa8b")

# Calcium
ax5 = axes[1, 0]
sns.lineplot(ax=ax5, x=df['times'], y=df['calcium'], linewidth=0.1, color="#0081a7")
ax5.set_title("calcium", fontsize=FONTSIZE_2)
ax5.set_xlabel('Time', fontsize=FONTSIZE_4)
ax5.set_ylabel('calcium', fontsize=FONTSIZE_4, color="#0081a7")
ax5.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#0081a7")

# Zoomed calcium
ax6 = axes[1, 1]
sns.lineplot(ax=ax6, x=df['times'], y=df['calcium'], linewidth=2, color="#0081a7")
ax6.set_xlim(start_index, end_index)
ax6.set_title("Zoomed calcium", fontsize=FONTSIZE_2)
ax6.set_xlabel('Time', fontsize=FONTSIZE_4)
ax6.set_ylabel('calcium', fontsize=FONTSIZE_4, color="#0081a7")
ax6.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#0081a7")

# Moving average of calcium
ax7 = axes[2, 0]
sns.lineplot(ax=ax7, x=df['times'], y=df['calcium_moving_avg'], linewidth=1, color="#f4a261")
# sns.lineplot(ax=ax7, x=df['times'], y=df['calcium_moving_avg_2'], linewidth=1, color="#c75000")
ax7.set_title("Moving Average of Calcium", fontsize=FONTSIZE_2)
ax7.set_xlabel('Time', fontsize=FONTSIZE_4)
ax7.set_ylabel('Calcium Moving Average', fontsize=FONTSIZE_4, color="#f4a261")
ax7.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#f4a261") 

# Zoomed calcium
ax8 = axes[2, 1]
sns.lineplot(ax=ax8, x=df['times'], y=df['calcium_moving_avg'], linewidth=2, color="#f4a261")
# sns.lineplot(ax=ax8, x=df['times'], y=df['calcium_moving_avg_2'], linewidth=2, color="#c75000")
ax8.set_xlim(start_index, end_index)
ax8.set_title("Zoomed moving average calcium", fontsize=FONTSIZE_2)
ax8.set_xlabel('Time', fontsize=FONTSIZE_4)
ax8.set_ylabel('calcium mov average', fontsize=FONTSIZE_4, color="#f4a261")
ax8.tick_params(axis='both', which='major', labelsize=FONTSIZE_3, colors="#f4a261")


plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the main title
plt.show() 


# %% ##################################################################################################################
# reaction time = 1stMov - stimOn
# response time = response_times - stimOn
# movement time = total wheel movement time (excluding when wheel was hold still) #new var(?) - to explore if the mouse was moving more anxiously the wheel, if there was almost no mov... 


# %% ##################################################################################################################
# Fig 2. Heatmap with all trials in 1 session & lineplot of all trials in 1 session aligned to an event 
"""
change the x axix to seconds; FR=30FPS
to add:     highlight the selected trial
            keep the means, add a line for the selected trial to appear in ax2 
remove the legends? 
"""
path = '/mnt/h0/kb/code_kcenia/photometry_files/' 
psth_idx_1 = np.load(path+"psthidx_feedback_times_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.npy") 
behav_1 = pd.read_parquet(path+"trialstable_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 
nphca_1 = pd.read_parquet(path+"demux_nph_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 

# Indices for correct and incorrect feedback
indices = behav_1["feedbackType"] == 1
indices_incorrect = behav_1["feedbackType"] == -1
feedback_correct = psth_idx_1[:, indices]
feedback_incorrect = psth_idx_1[:, indices_incorrect] 

# Calculate the mean for feedback_correct and feedback_incorrect
mean_feedback_correct = np.mean(feedback_correct, axis=1)
mean_feedback_incorrect = np.mean(feedback_incorrect, axis=1)

fig = plt.figure(figsize=(6, 15))
gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 2], hspace=0.1)

# First subplot: heatmap
ax1 = fig.add_subplot(gs[0])
sns.heatmap(feedback_correct.T, ax=ax1, cbar=False)
ax1.axvline(x=30.5, linestyle='dashed', color='white')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Second subplot: line plot
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(feedback_correct, linewidth=0.2, color='#0077b6', alpha=0.1)
ax2.plot(feedback_incorrect, linewidth=0.2, color='#ff5858', alpha=0.1)
# Plot the mean lines
ax2.plot(mean_feedback_correct, linewidth=2, color='#0077b6', label='Mean Correct')
ax2.plot(mean_feedback_incorrect, linewidth=2, color='#ff5858', label='Mean Inc')
ax2.axvline(x=30.5, linestyle='dashed', color='black')
# Add legend
ax2.legend()
# Remove right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# Set labels
ax2.set_xlabel('Time since event')
ax2.set_ylabel('Signal')

# Show the plot
plt.show() 


# %% ##################################################################################################################
# Fig 3. Get single line for signal during a trial 
"""
add color: #0077b6 if correct, #ff5858 if inc 
"""
path = '/mnt/h0/kb/code_kcenia/photometry_files/' 
psth_idx_1 = np.load(path+"psthidx_feedback_times_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.npy") 
behav_1 = pd.read_parquet(path+"trialstable_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 
nphca_1 = pd.read_parquet(path+"demux_nph_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 

nph_singletrial = [] 
for i in range(len(behav_1.trialNumber)): 
    # Define the interval for the current trial
    interval_start = behav_1.intervals_0[i]
    interval_end = behav_1.intervals_1[i] 
    
    # Use boolean indexing to find the times within the current interval
    mask = (nphca_1.times >= interval_start) & (nphca_1.times <= interval_end)
    
    # Append the corresponding calcium values to the list
    nph_singletrial.append(nphca_1.calcium[mask]) 

# Example for trial 1
# Get the times corresponding to the first trial's calcium values 
index = 0 #select trial (trial = index+1)
times_trial_1 = nphca_1.times[(nphca_1.times >= behav_1.intervals_0[index]) & (nphca_1.times <= behav_1.intervals_1[index])]

plt.figure(figsize=(10, 5))
plt.plot(times_trial_1, nph_singletrial[index], label='Calcium Signal', color='#0077b6')
plt.axvline(behav_1.stimOnTrigger_times[index], color='blue', linestyle='dashed', label='Stim Onset') 
plt.axvline(behav_1.firstMovement_times[index], color='green', linestyle='dashed', label='First Mov') 
plt.axvline(behav_1.feedback_times[index], color='red', linestyle='dashed', label='Feedback')
plt.xlabel('Time')
plt.ylabel('Calcium Signal') 
plt.title(f"Trial number: {index + 1}")
plt.legend()
plt.show() 


# %% ##################################################################################################################
# Fig 4. Get single line for signal during a trial 
"""
to plot: 
    split by block: 
        for correct and incorrect: 
            time from stim on 
            time from feedback 
    split by contrast: 
        for correct and incorrect: 
            time from stim on 
            time from feedback 
    split by left and right: #maybe for c and inc 
            time from 1st mov 
    split by correct and incorrect in general #maybe not needed because it will appear in the plots above 
            for stim on, 1st mov and feednack time 
plot all lines as in the code below, or only the avg line with the error 
add a "smart" ylim? 
"""
path = '/mnt/h0/kb/code_kcenia/photometry_files/' 
psth_idx_1 = np.load(path+"psthidx_feedback_times_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.npy") 
behav_1 = pd.read_parquet(path+"trialstable_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 
nphca_1 = pd.read_parquet(path+"demux_nph_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 

indices_50 = behav_1["probabilityLeft"] == 0.5
indices_20 = behav_1["probabilityLeft"] == 0.2 
indices_80 = behav_1["probabilityLeft"] == 0.8 

block_50 = psth_idx_1[:, indices_50]
block_20 = psth_idx_1[:, indices_20] 
block_80 = psth_idx_1[:, indices_80]

mean_50 = np.mean(block_50, axis=1)
mean_20 = np.mean(block_20, axis=1) 
mean_80 = np.mean(block_80, axis=1)

fig = plt.figure(figsize=(5, 5))
ax2 = fig.add_subplot(111)
# # Plot all lines OR DO ONLY AVG AND SEM
# ax2.plot(block_50, linewidth=0.2, color='gray', alpha=0.1)
# ax2.plot(block_20, linewidth=0.2, color='#ff6d00', alpha=0.1)
# ax2.plot(block_80, linewidth=0.2, color='teal', alpha=0.1)
# Plot the mean lines
ax2.plot(mean_50, linewidth=2, color='gray', label='Mean 0.5') 
ax2.plot(mean_20, linewidth=2, color='#ff6d00', label='Mean 0.2') 
ax2.plot(mean_80, linewidth=2, color='teal', label='Mean 0.8')
# Add a vertical line 
ax2.axvline(x=30.5, linestyle='dashed', color='black')
# Add legend
ax2.legend()
# Remove right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# Set labels
ax2.set_xlabel('Time since event')
ax2.set_ylabel('Signal')
# Show the plot
plt.show() 

#%%
# All contrasts
indices_100 = behav_1["allContrasts"] == 1 
indices_25 = behav_1["allContrasts"] == 0.25
indices_12 = behav_1["allContrasts"] == 0.125
indices_6 = behav_1["allContrasts"] == 0.0625
indices_0 = behav_1["allContrasts"] == 0

contrast_100 = psth_idx_1[:, indices_100]
contrast_25 = psth_idx_1[:, indices_25]
contrast_12 = psth_idx_1[:, indices_12]
contrast_6 = psth_idx_1[:, indices_6]
contrast_0 = psth_idx_1[:, indices_0]

mean_100 = np.mean(contrast_100, axis=1)
mean_25 = np.mean(contrast_25, axis=1)
mean_12 = np.mean(contrast_12, axis=1)
mean_6 = np.mean(contrast_6, axis=1)
mean_0 = np.mean(contrast_0, axis=1)

fig = plt.figure(figsize=(5, 5))
ax2 = fig.add_subplot(111)

# Plot the mean lines
ax2.plot(mean_100, linewidth=2, color='black', label='Contrast 1', alpha=1)
ax2.plot(mean_25, linewidth=2, color='black', label='Contrast 0.25', alpha=0.7)
ax2.plot(mean_12, linewidth=2, color='black', label='Contrast 0.12', alpha=0.4) 
ax2.plot(mean_6, linewidth=2, color='black', label='Contrast 0.06', alpha=0.2)
ax2.plot(mean_0, linewidth=2, color='#bcd4e6', label='Contrast 0', alpha=1)

# Add a vertical line 
ax2.axvline(x=30.5, linestyle='dashed', color='black') 

# Add legend
ax2.legend() 

# Remove right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# Set labels
ax2.set_xlabel('Time since event')
ax2.set_ylabel('Signal')

plt.show() 




# %% ##################################################################################################################
# Example usage
################################################
""" CHANGE HERE """ 
# path = '/home/kceniabougrova/Documents/results_for_OW/' 
path = '/mnt/h0/kb/code_kcenia/photometry_files/' 
# prefix = 'trialstable_' 
prefix = 'demux_' 
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
nphca_1 = pd.read_parquet(path+"demux_nph_ZFM-04022_2022-11-29_3_65f90bf6-5124-430a-ab73-134ac6fb374f.pqt") 

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
