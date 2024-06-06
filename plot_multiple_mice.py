############################################################
#
# KceniaB
# 
# 29May2024
#
# PLOTTING THE AVERAGE ACROSS SEVERAL SESSIONS 
#
#
#
############################################################

#%%
import os
import numpy as np
import pandas as pd 
from brainbox.io.one import SessionLoader
import matplotlib.pyplot as plt 
import seaborn as sns
# from functions_nm import load_trials 
from functions_nm import * 
import neurodsp.utils 
from pathlib import Path
import iblphotometry.plots
import iblphotometry.dsp 
from one.api import ONE
one = ONE()

path = '/home/kceniabougrova/Documents/results_for_OW/'

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
    
################################################
""" CHANGE HERE """
mouse = "ZFM-04022"
prefix = f"demux_nph_{mouse}_"
################################################ 
file_list = list_files_in_folder(path, prefix)

# Dictionary to store the DataFrames 
dataframes = {} 
dataframes_trials = {}

for file_name in file_list: 
    mouse = file_name[10:19]
    date = file_name[20:30] 
    region_number = file_name[31:32]
    print(mouse, date, region_number)

    def get_eid(mouse, date): 
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
            session_path_behav = f'/home/kceniabougrova/Documents/nph/Behav_2024Mar20/{mouse}/{date}/001/'
            df_alldata = extract_all(session_path_behav)
            table_data = df_alldata[0]['table']
            trials = pd.DataFrame(table_data) 
        return eid, trials 

    eid, df_trials = get_eid(mouse, date) 
    path_nph = f'demux_nph_{mouse}_{date}_{region_number}_{eid}' 
    df_nph = pd.read_parquet(os.path.join(path, path_nph+'.pqt')) 

    # Add the date column to the df_nph and df_trials dataframes
    df_nph['mouse'] = mouse
    df_trials['mouse'] = mouse
    df_nph['date'] = date
    df_trials['date'] = date
    df_nph['region_number'] = region_number
    df_trials['region_number'] = region_number
    df_nph['NM'] = "DA"
    df_trials['NM'] = "DA"

    # Store the DataFrame in the dictionary with a unique key
    key = f"df_nph_{mouse.replace('-', '_')}_{date.replace('-', '_')}"
    dataframes[key] = df_nph

    # Store the DataFrame in the dictionary with a unique key
    key_2 = f"df_trials_{mouse.replace('-', '_')}_{date.replace('-', '_')}" 
    df_trials = df_trials[0:len(df_trials)-1] #to avoid the last trial not having photometry data 
    dataframes_trials[key_2] = df_trials

#%%
# Example to access a specific DataFrame
for key in dataframes:
    print(f"DataFrame for {key}:")
    print(dataframes[key].head())  # Display the first few rows of each DataFrame
for key, df in dataframes_trials.items():
    print(f"DataFrame for {key}:")
    print(df.head())  # Display the first few rows of each DataFrame

# df_nph_ZFM_04022_2022_09_16 = dataframes['df_nph_ZFM_04022_2022_09_16']
# df_nph_ZFM_04022_2023_03_16 = dataframes['df_nph_ZFM_04022_2023_03_16']
# df_nph_ZFM_04022_2023_01_10 = dataframes['df_nph_ZFM_04022_2023_01_10']
# df_nph_ZFM_04022_2023_01_06 = dataframes['df_nph_ZFM_04022_2023_01_06']
# df_nph_ZFM_04022_2023_06_27 = dataframes['df_nph_ZFM_04022_2023_06_27']
# # df_nph_ZFM_04022_2022_09_15 = dataframes['df_nph_ZFM_04022_2022_09_15']
# df_nph_ZFM_04022_2023_03_24 = dataframes['df_nph_ZFM_04022_2023_03_24']
# df_nph_ZFM_04022_2023_01_12 = dataframes['df_nph_ZFM_04022_2023_01_12']
# df_nph_ZFM_04022_2023_03_02 = dataframes['df_nph_ZFM_04022_2023_03_02']
# df_nph_ZFM_04022_2023_01_19 = dataframes['df_nph_ZFM_04022_2023_01_19']
# df_nph_ZFM_04022_2022_09_21 = dataframes['df_nph_ZFM_04022_2022_09_21']
# df_nph_ZFM_04022_2023_03_28 = dataframes['df_nph_ZFM_04022_2023_03_28']
# df_nph_ZFM_04022_2023_04_04 = dataframes['df_nph_ZFM_04022_2023_04_04']
# df_nph_ZFM_04022_2023_04_03 = dataframes['df_nph_ZFM_04022_2023_04_03']
# df_nph_ZFM_04022_2023_03_22 = dataframes['df_nph_ZFM_04022_2023_03_22']
# # df_nph_ZFM_04022_2022_12_30 = dataframes['df_nph_ZFM_04022_2022_12_30']
# df_nph_ZFM_04022_2022_11_24 = dataframes['df_nph_ZFM_04022_2022_11_24']
# # df_nph_ZFM_04022_2022_12_29 = dataframes['df_nph_ZFM_04022_2022_12_29']
# df_nph_ZFM_04022_2023_01_11 = dataframes['df_nph_ZFM_04022_2023_01_11']


# df_trials_04019_2022_09_16 = dataframes_trials.get('df_trials_ZFM_04019_2022_09_16')
# df_trials_04019_2023_03_16 = dataframes_trials.get('df_trials_04019_2023_03_16')
# df_trials_04019_2023_01_10 = dataframes_trials.get('df_trials_04019_2023_01_10')
# df_trials_04019_2023_01_06 = dataframes_trials.get('df_trials_04019_2023_01_06')
# df_trials_04019_2023_06_27 = dataframes_trials.get('df_trials_04019_2023_06_27')
# # df_trials_04019_2022_09_15 = dataframes_trials.get('df_trials_04019_2022_09_15')
# df_trials_04019_2023_03_24 = dataframes_trials.get('df_trials_04019_2023_03_24')
# df_trials_04019_2023_01_12 = dataframes_trials.get('df_trials_04019_2023_01_12')
# df_trials_04019_2023_03_02 = dataframes_trials.get('df_trials_04019_2023_03_02')
# df_trials_04019_2023_01_19 = dataframes_trials.get('df_trials_04019_2023_01_19')
# df_trials_04019_2022_09_21 = dataframes_trials.get('df_trials_04019_2022_09_21')
# df_trials_04019_2023_03_28 = dataframes_trials.get('df_trials_04019_2023_03_28')
# df_trials_04019_2023_04_04 = dataframes_trials.get('df_trials_04019_2023_04_04')
# df_trials_04019_2023_04_03 = dataframes_trials.get('df_trials_04019_2023_04_03')
# df_trials_04019_2023_03_22 = dataframes_trials.get('df_trials_04019_2023_03_22')
# # df_trials_04019_2022_12_30 = dataframes_trials.get('df_trials_04019_2022_12_30')
# df_trials_04019_2022_11_24 = dataframes_trials.get('df_trials_04019_2022_11_24')
# # df_trials_04019_2022_12_29 = dataframes_trials.get('df_trials_04019_2022_12_29')
# df_trials_04019_2023_01_11 = dataframes_trials.get('df_trials_04019_2023_01_11')


#%%
###############################################################
# using example sessions to plot the data individually first 
############################################################### 
# df_nph = df_nph_04019_2022_09_16
# df_trials = df_trials_04019_2022_09_16 
###############################################################
# WORKED 
###############################################################
# df_trials = df_trials[0:len(df_trials)-1] #to avoid the last trial not having photometry data 

#%%

# Assuming the dictionaries 'dataframes_nph' and 'dataframes_trials' already contain the necessary dataframes

# List of dates for which the dataframes need to be concatenated
dates = [
    "2022_12_30",
    "2022_11_29",
    "2023_01_12",
    "2023_01_10"
]

# Mouse ID
mouse_id = "ZFM_04022"

# Collect the nph dataframes
frames_nph = [dataframes[f"df_nph_{mouse_id}_{date}"] for date in dates]

# Collect the trials dataframes
frames_trials = [dataframes_trials[f"df_trials_{mouse_id}_{date}"] for date in dates]

# Concatenate the nph dataframes and reset the index
df_nph = pd.concat(frames_nph).reset_index(drop=True)

# Concatenate the trials dataframes and reset the index
df_trials = pd.concat(frames_trials).reset_index(drop=True)

# Print to verify
print(df_nph.head())
print(df_trials.head())



#%% 
#%%



# #%%
# # Behavior 
# idx = 2
# new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
# df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
# #create allUContrasts 
# df_trials['allUContrasts'] = df_trials['allContrasts']
# df_trials.loc[df_trials['contrastRight'].isna(), 'allUContrasts'] = df_trials['allContrasts'] * -1
# df_trials.insert(loc=3, column='allUContrasts', value=df_trials.pop('allUContrasts'))

# #create reactionTime
# reactionTime = np.array((df_trials["firstMovement_times"])-(df_trials["stimOnTrigger_times"]))
# df_trials["reactionTime"] = reactionTime 


# a = one.load_object(eid, 'trials')
# trials = a.to_df()

# df_trials = df_trials[0:len(df_trials)-1] #to avoid the last trial not having photometry data 

# """
# now I should have df_trials and df_nph
# """

# df_nph_1 = df_nph_04019_2022_09_16
# df_trials_1 = df_trials_04019_2022_09_16 
# df_nph_2 = df_nph_04019_2023_03_16
# df_trials_2 = df_trials_04019_2023_03_16
# frames_nph = [df_nph_1, df_nph_2]
# frames_trials = [df_trials_1, df_trials_2]
# df_nph = pd.concat(frames_nph)
# df_nph = df_nph.reset_index(drop=True)
# df_trials = pd.concat(frames_trials) 
# df_trials = df_trials.reset_index(drop=True)

df_trials = df_trials[0:len(df_trials)-1] #to avoid the last trial not having photometry data 

#%%
# rearrange and plot the data 
# fig, ax = iblphotometry.plots.plot_raw_data_df(df_nph) 

array_timestamps_bpod = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(array_timestamps_bpod, event_test) #check idx where they would be included, in a sorted way 
print(idx_event) 

""" create a column with the trial number in the nph df """
df_nph["trial_number"] = 0 #create a new column for the trial_number 
df_nph.loc[idx_event,"trial_number"]=1
df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR

################################################
""" CHANGE HERE """
EVENT_NAME = "feedback_times"
################################################

sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_trials.shape[0]

psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

event_feedback = np.array(df_trials[EVENT_NAME]) #pick the feedback timestamps 

feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

psth_idx += feedback_idx

df_nph.calcium.values[psth_idx] 




""" all subplots with sem """ 
# Your conditions and data setup
plt.rcParams["figure.figsize"] = (8,6)
width_ratios = [1, 1]
FONTSIZE_1 = 16
FONTSIZE_2 = 15
FONTSIZE_3 = 15 
event_time = 30 

psth_error = df_nph.calcium.values[psth_idx[:,(df_trials.feedbackType == -1)]]
psth_good = df_nph.calcium.values[psth_idx[:,(df_trials.feedbackType == 1)]]
sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])
sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1]) 

average_values_error = psth_error.mean(axis=1) 
average_values_good = psth_good.mean(axis=1) 
plt.plot(average_values_error, color='#d63230', linewidth=3) 
plt.fill_between(range(len(average_values_error)), average_values_error - sem_error, average_values_error + sem_error, color='#d63230', alpha=0.15)
plt.plot(average_values_good, color='#1789fc', linewidth=3) 
plt.fill_between(range(len(average_values_good)), average_values_good - sem_good, average_values_good + sem_good, color='#1789fc', alpha=0.15)
plt.title(f"NM response at {EVENT_NAME}\n"+f"{mouse}_{date}_{region_number}", fontsize=FONTSIZE_1)
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
plt.legend(fontsize=FONTSIZE_3, frameon=False) 

# plt.ylim(-0.005,0.004) 
plt.grid(False) 
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)

ax.spines['bottom'].set_color("black")
ax.spines['left'].set_color("black") 
# plt.savefig(f'/home/kceniabougrova/Documents/figures_forlabmeeting_May2024/Fig01_{mouse}_{date}_{region_number}_{EVENT_NAME}.png')

plt.show() 







# %%











#%%
#%%
import os
import numpy as np
import pandas as pd 
from brainbox.io.one import SessionLoader
import matplotlib.pyplot as plt 
import seaborn as sns
# from functions_nm import load_trials 
from functions_nm import * 
import neurodsp.utils 
from pathlib import Path
import iblphotometry.plots
import iblphotometry.dsp 
from one.api import ONE
one = ONE()

path = '/home/kceniabougrova/Documents/results_for_OW/'

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
    
# Example usage
################################################
""" CHANGE HERE """
prefix = "demux_nph_ZFM-04022_"
################################################
path = '/home/kceniabougrova/Documents/results_for_OW/' 
file_list = list_files_in_folder(path, prefix)

# Dictionary to store the DataFrames
dataframes = {}
dataframes_trials = {}

for file_name in file_list: 
    mouse = file_name[10:19]
    date = file_name[20:30] 
    region_number = file_name[31:32]
    print(mouse, date, region_number)

    def get_eid(mouse, date): 
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
            session_path_behav = f'/home/kceniabougrova/Documents/nph/Behav_2024Mar20/{mouse}/{date}/001/'
            df_alldata = extract_all(session_path_behav)
            table_data = df_alldata[0]['table']
            trials = pd.DataFrame(table_data) 
        return eid, trials 

    eid, df_trials = get_eid(mouse, date) 
    path_nph = f'demux_nph_{mouse}_{date}_{region_number}_{eid}' 
    df_nph = pd.read_parquet(os.path.join(path, path_nph+'.pqt')) 

    # Add the date column to the df_nph and df_trials dataframes
    df_nph['mouse'] = mouse
    df_trials['mouse'] = mouse
    df_nph['date'] = date
    df_trials['date'] = date
    df_nph['region_number'] = region_number
    df_trials['region_number'] = region_number
    df_nph['NM'] = "DA"
    df_trials['NM'] = "DA"

    key = f"df_nph_{mouse.replace('-', '_')}_{date.replace('-', '_')}"
    dataframes[key] = df_nph
    # Store the DataFrame in the dictionary with a unique key
    key_2 = f"df_trials_{mouse.replace('-', '_')}_{date.replace('-', '_')}"
    dataframes_trials[key_2] = df_trials

# Function to dynamically create variables from dictionary
def create_variables_from_dict(dataframes_dict):
    for key, df in dataframes_dict.items():
        # Creating the variable name from the key
        var_name = key
        # Using exec to create the variable with the DataFrame
        exec(f"globals()['{var_name}'] = df")
        print(f"{var_name} created")

# Call the function to create variables
create_variables_from_dict(dataframes)
create_variables_from_dict(dataframes_trials)

# # Verify if the variables are created
# for key in dataframes.keys():
#     var_name = key
#     # Check if the variable is in the global variables
#     if var_name in globals():
#         print(f"{var_name} exists")

# Now you can work with these DataFrames directly
# For example:
# print(df_trials_ZFM_04019_2022_09_16.head())




# List of nph dataframes
frames_nph = [
    df_nph_ZFM_04022_2022_12_30,
    df_nph_ZFM_04022_2022_11_29,
]

# List of trials dataframes
frames_trials = [
    df_trials_ZFM_04022_2022_12_30,
    df_trials_ZFM_04022_2022_11_29,
]

# Concatenate the nph dataframes and reset the index
df_nph = pd.concat(frames_nph).reset_index(drop=True)

# Concatenate the trials dataframes and reset the index
df_trials = pd.concat(frames_trials).reset_index(drop=True)

# Print to verify the results
print(df_nph.head())
print(df_trials.head())





# %%
