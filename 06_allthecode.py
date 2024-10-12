#%%
"""
KceniaBougrova 
08October2024 

Create a df with the mouse_name, session_date and pm (performance values) 
    eid
    Subject (e.g. ZFM-03448)
    unsigned performance
    signed performance by contrast
        -100
        -50
        -25
        -12
        -06
        00
        06
        12
        25
        50
        100 
""" 

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from brainbox.behavior.training import compute_performance 
from brainbox.io.one import SessionLoader
from one.api import ONE #always after the imports 
one = ONE(cache_dir="/mnt/h0/kb/data/one") 

#%%
""" useful""" 
eids = one.search(project='ibl_fibrephotometry') 

# Initialize the DataFrame
new_df = pd.DataFrame() 
excluded_2 = pd.DataFrame()

# Initialize lists to store values
a = []
b = []
c = []
d = [] 
cn100 = [] 
cn50 = []
cn25 = [] 
cn12 = []
cn06 = []
c00 = []
c06 = []
c12 = []
c25 = [] 
c50 = [] 
c100 = [] 
excluded = [] 
error = []

# Predefined contrast values
contrast_values = np.array([-100.,  -50.,   -25., -12.5, -6.25, 0, 6.25, 12.5, 25.,  50.,  100.])

# Loop through the eids
for eid in eids[665:700]: 
# for eid in eids:
    try: 
        # Load trials and session data
        trials = one.load_object(eid, 'trials')
        ref = one.eid2ref(eid)
        subject = ref.subject
        session_date = str(ref.date)

        # Compute performance and contrasts
        performance, contrasts, n_contrasts = compute_performance(trials)
        
        # Initialize contrast performance dictionary for current session
        contrast_perf_dict = dict(zip(contrasts, performance))
        
        # Performance mean for contrasts -100 and 100
        pm_values = [contrast_perf_dict.get(-100, np.nan), contrast_perf_dict.get(100, np.nan)]
        pm = np.nanmean(pm_values)
        
        # Append eid, subject, session_date, and pm to the lists
        a.append(eid)
        b.append(subject)
        c.append(session_date)
        d.append(str(round(pm, 2)))

        # Append contrast-specific performance values
        cn100.append(contrast_perf_dict.get(-100., np.nan))
        cn50.append(contrast_perf_dict.get(-50., np.nan))
        cn25.append(contrast_perf_dict.get(-25., np.nan))
        cn12.append(contrast_perf_dict.get(-12.5, np.nan))
        cn06.append(contrast_perf_dict.get(-6.25, np.nan))
        c00.append(contrast_perf_dict.get(0, np.nan))
        c06.append(contrast_perf_dict.get(6.25, np.nan))
        c12.append(contrast_perf_dict.get(12.5, np.nan))
        c25.append(contrast_perf_dict.get(25., np.nan))
        c50.append(contrast_perf_dict.get(50., np.nan))
        c100.append(contrast_perf_dict.get(100., np.nan)) 
        print(f"DONE eid {eid}")

    except Exception as e: 
        print(f"excluded eid: {eid} due to error: {e}") 
        excluded.append(eid)
        error.append(e)
        pass

# Create df from the lists
new_df["eid"] = a
new_df["subject"] = b
new_df["session_date"] = c
new_df["unsigned_performance"] = d
new_df["cn100"] = cn100
new_df["cn50"] = cn50
new_df["cn25"] = cn25
new_df["cn12"] = cn12
new_df["cn06"] = cn06
new_df["c00"] = c00
new_df["c06"] = c06
new_df["c12"] = c12
new_df["c25"] = c25
new_df["c50"] = c50
new_df["c100"] = c100 

excluded_2["excluded"] = excluded
excluded_2["error"] = error
# # Save the performance values and the error eids into csv 
# new_df.to_csv('performance_all_photometry_sessions.csv', index=False) 
# excluded_2.to_csv('excluded_photometry_sessions.csv', index=False) 


#%% #########################################################################################################
""" useful to confirm """
# """ EXTRACT THE 2 COLUMNS OF THE INTERVALS FROM TRIALS """
# # Extract feedback_times separately if it's a 2D array (nested)
# if len(trials['intervals'].shape) == 2:
#     trials['intervals_0'] = trials['intervals'][:, 0]
#     trials['intervals_1'] = trials['intervals'][:, 1]
#     del trials['intervals']  # Remove original nested array

# # Convert to DataFrame again
# trials_df = pd.DataFrame(trials)

# # Display the first few rows
# print(trials_df.head())

# #%% 
# """ LEFT IS NEGATIVE """
# """ LEFT IS NEGATIVE """
# df_trials_corr = df_trials[
#     (df_trials['feedbackType'] == 1) & 
#     ((df_trials['contrastLeft'] == 1.0))
# ]
# df_trials_all = df_trials[
#     ((df_trials['contrastLeft'] == 1.0))
# ]
# len(df_trials_corr)/len(df_trials_all) 




#%% #########################################################################################################
#########################################################################################################
#########################################################################################################
# """ HOW TO SOLVE THE PROBABILITYLEFT BLOCKS BUG """ 
# https://int-brain-lab.github.io/ONE/notebooks/one_list/one_list.html

dataset_task_settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json')  
values = dataset_task_settings.get('LEN_BLOCKS', 'Key not found') 
# values gives the block length 
# example for eid = 'be3208c9-43de-44dc-bdc6-ff8963464f98'
# [90, 27, 82, 50, 30, 30, 31, 78, 64, 83, 24, 42, 74, 72, 34, 41, 52, 56, 68, 39, 45, 88, 37, 35, 29, 69, 85, 52, 37, 78, 80, 28, 68, 95, 34, 36, 42] 

values_sum = np.cumsum(values) 

#%%
""" LOAD DATA """
eid = '89b8ef70-e620-49c2-a0f7-09890ba9fc0e'
def load_trials_updated(eid=eid): 
    trials = one.load_object(eid, 'trials')
    ref = one.eid2ref(eid)
    subject = ref.subject
    session_date = str(ref.date) 
    if len(trials['intervals'].shape) == 2: 
        trials['intervals_0'] = trials['intervals'][:, 0]
        trials['intervals_1'] = trials['intervals'][:, 1]
        del trials['intervals']  # Remove original nested array 

    df_trials = pd.DataFrame(trials) 
    idx = 2
    new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
    df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
    # create allSContrasts 
    df_trials['allSContrasts'] = df_trials['allContrasts']
    df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
    df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))
    df_trials[["subject", "date", "eid"]] = [subject, session_date, eid]    
    df_trials["reactionTime"] = df_trials["firstMovement_times"] - df_trials["stimOnTrigger_times"]
    df_trials["responseTime"] = df_trials["response_times"] - df_trials["stimOnTrigger_times"] 
    df_trials["trialTime"] = df_trials["intervals_1"] - df_trials["intervals_0"]  
    df_trials["trialNumber"] = range(1, len(df_trials) + 1) 
    return df_trials, subject, session_date
df_trials, subject, session_date = load_trials_updated(eid) 



def debug_probL(df=df_trials, eid=eid): 
    dataset_task_settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json')  
    values = dataset_task_settings.get('LEN_BLOCKS', 'Key not found') 
    values_sum = np.cumsum(values)
    df_trials['probL'] = np.nan
    # Set the first block (first `values_sum[0]` rows) to 0.5
    df_trials.loc[:values_sum[0]-1, 'probL'] = 0.5 
    df_trials.loc[values_sum[0]:values_sum[1]-1, 'probL'] = df_trials.loc[values_sum[0], 'probabilityLeft']
    previous_value = df_trials.loc[values_sum[1]-1, 'probabilityLeft'] 

    # Iterate over the blocks starting from values_sum[1]
    for i in range(1, len(values_sum)-1):
        print("i = ", i)
        start_idx = values_sum[i]
        end_idx = values_sum[i+1]-1
        print("start and end _idx = ", start_idx, end_idx)
        
        # Assign the block value based on the previous one
        if previous_value == 0.2:
            current_value = 0.8
        else:
            current_value = 0.2
        print("current value = ", current_value)

        # Set the 'probL' values for the current block
        df_trials.loc[start_idx:end_idx, 'probL'] = current_value
        
        # Update the previous_value for the next block
        previous_value = current_value

    # Handle any remaining rows after the last value_sum block
    if len(df_trials) > values_sum[-1]:
        df_trials.loc[values_sum[-1] + 1:, 'probL'] = previous_value

    # plt.plot(df_trials.probabilityLeft, alpha=0.5)
    # plt.plot(df_trials.probL, alpha=0.5)
    # plt.show() 
    return df_trials


df_trials = debug_probL(df_trials)
# %%



sns.kdeplot(data=df_trials, x="quiescencePeriod", bw_adjust=.25, hue='feedbackType') 

#%% #########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
""" GET PHOTOMETRY DATA """ 
mouse = subject
date = session_date
def get_eid(mouse,date): 
    eids = one.search(subject=mouse, date=date) 
    eid = eids[0]
    ref = one.eid2ref(eid)
    print(eid)
    print(ref) 
    return eid

def get_regions(rec): 
    regions = [f"Region{rec.region}G"] 
    return regions 

def get_nph(source_path, rec): 
    # source_folder = (f"/home/kceniabougrova/Documents/nph/{rec.date}/") 
    source_folder = source_path
    df_nph = pd.read_csv(source_folder+f"raw_photometry{rec.nph_file}.csv") 
    return df_nph  

import os
import glob
import pandas as pd

def load_photometry_data(subject, session_date, base_path='/mnt/h0/kb/data/one/mainenlab/Subjects'):
    """
    Function to load raw photometry data from folders dynamically based on the structure:
    /{subject}/{session_date}/{anynumber}/alf/{region_field}/raw_photometry.pqt

    Parameters:
    -----------
    subject : str
        The subject ID (e.g., 'ZFM-04019').
    session_date : str
        The session date (e.g., '2023-03-02').
    base_path : str, optional
        The base path where the data is stored (default is '/mnt/h0/kb/data/one/mainenlab/Subjects').

    Returns:
    --------
    data_dict : dict
        A dictionary where keys are (anynumber, region_field) tuples, and values are loaded DataFrames.
        Example: {('001', 'Region3G'): DataFrame, ('002', 'Region4G'): DataFrame, ...}
    """
    # Construct the full path to the session directory
    session_path = os.path.join(base_path, subject, session_date)

    # Use glob to find all the folders with the structure like /001/, /002/ (anynumber folders)
    anynumber_folders = glob.glob(os.path.join(session_path, '*/'))

    # Initialize a dictionary to store the loaded data
    data_dict = {}

    # Loop through all anynumber folders
    for folder in anynumber_folders:
        anynumber = os.path.basename(os.path.normpath(folder))  # Extract folder name (e.g., '001')

        # Use glob to find all the folders with the structure like /RegionXG/ inside the 'alf' folder
        region_folders = glob.glob(os.path.join(folder, 'alf/Region*G/'))  # Match folders like 'Region3G', 'Region4G', etc.

        # Loop through all region folders
        for region_folder in region_folders:
            region_field = os.path.basename(os.path.normpath(region_folder))  # Extract the region folder name (e.g., 'Region3G')

            # Construct the file path
            file_path = os.path.join(region_folder, 'raw_photometry.pqt')

            # Check if the file exists, then load it
            if os.path.exists(file_path):
                print(f"Loading file: {file_path}")
                try:
                    # Load the Parquet file
                    data = pd.read_parquet(file_path)
                    # Store the loaded data in the dictionary with a key tuple of (anynumber, region_field)
                    data_dict[(anynumber, region_field)] = data
                    print(f"File loaded successfully for {region_field} in {anynumber}")
                except Exception as e:
                    print(f"Error loading file: {file_path} \nError: {e}")
            else:
                print(f"File not found: {file_path}")
    
    return data_dict


# Call the function to load photometry data
photometry_data = load_photometry_data(subject, session_date)

# Now `photometry_data` contains all loaded DataFrames, accessible by their folder combination
for (anynumber, region), df in photometry_data.items():
    print(f"Data for {anynumber} - {region}:")
    print(df.head())  # Display the first few rows of the loaded DataFrame
    df_nph = df
    df_nph["mouse"] = subject
    df_nph["date"] = date
    df_nph["region"] = region
    df_nph["eid"] = eid 






    # df_trials = df_trials[0:len(df_trials)-1] #to avoid the last trial not having photometry data 
    session_start = df_trials.intervals_0.values[0] - 10  # Start time, 100 seconds before the first tph value
    session_end = df_trials.intervals_1.values[-1] + 10   # End time, 100 seconds after the last tph value

    # Select data within the specified time range
    selected_data = df_nph[
        (df_nph['times'] >= session_start) &
        (df_nph['times'] <= session_end)
    ] 
    df_nph = selected_data.reset_index(drop=True) 

    time_diffs = df_nph["times"].diff().dropna()
    fs = 1 / time_diffs.median() 


    from iblphotometry.preprocessing import preprocessing_alejandro, jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 

    df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
    df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
    df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
    df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
    df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
    df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)
    df_nph['calcium_alex'] = preprocessing_alejandro(df_nph["raw_calcium"], fs=fs) 
    df_nph['isos_alex'] = preprocessing_alejandro(df_nph['raw_isosbestic'], fs=fs)

    #to check all of the signals 
    plt.figure(figsize=(20, 6))
    for name in column_name: 
        plt.plot(df_nph[name], linewidth=1.25, alpha=0.8, label=name) 
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), handlelength=3, handleheight=2, prop={'size': 10})
    leg = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    for line in leg.get_lines():
        line.set_linewidth(2.5)  # Thicker legend lines
    plt.xlim(350, 810)
    plt.show() 

    """ SELECT THE EVENT AND WHAT INTERVAL TO PLOT IN THE PSTH """ 
    EVENT = "feedback_times" 
    time_bef = -1
    time_aft = 2
    PERIEVENT_WINDOW = [time_bef,time_aft]
    SAMPLING_RATE = int(1/np.mean(np.diff(df_nph.times))) 

    array_timestamps = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
    event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
    idx_event = np.searchsorted(array_timestamps, event_test) #check idx where they would be included, in a sorted way 
    """ create a column with the trial number in the nph df """
    df_nph["trial_number"] = 0 #create a new column for the trial_number 
    df_nph.loc[idx_event,"trial_number"]=1
    df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

    sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
    n_trials = df_trials.shape[0]

    psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) 

    event_times = np.array(df_trials[EVENT]) #pick the feedback timestamps 

    event_idx = np.searchsorted(array_timestamps, event_times) #check idx where they would be included, in a sorted way 

    psth_idx += event_idx

    
    photometry_s_1 = df_nph.calcium_photobleach.values[psth_idx] 
    # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_1)
    photometry_s_2 = df_nph.isosbestic_photobleach.values[psth_idx] 
    # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_2)
    photometry_s_3 = df_nph.calcium_jove2019.values[psth_idx] 
    # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_3)
    photometry_s_4 = df_nph.isosbestic_jove2019.values[psth_idx] 
    # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_4)
    photometry_s_5 = df_nph.calcium_mad.values[psth_idx] 
    # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_5)
    photometry_s_6 = df_nph.isosbestic_mad.values[psth_idx] 
    # np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_6) 
    photometry_s_7 = df_nph.calcium_alex.values[psth_idx] 
    photometry_s_8 = df_nph.isos_alex.values[psth_idx] 

    def plot_heatmap_psth(preprocessingtype=df_nph.calcium_mad): 
        psth_good = preprocessingtype.values[psth_idx[:,(df_trials.feedbackType == 1)]]
        psth_error = preprocessingtype.values[psth_idx[:,(df_trials.feedbackType == -1)]]
        # Calculate averages and SEM
        psth_good_avg = psth_good.mean(axis=1)
        sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
        psth_error_avg = psth_error.mean(axis=1)
        sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

        # Create the figure and gridspec
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

        # Plot the heatmap and line plot for correct trials
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(psth_good.T, cbar=False, ax=ax1) #, center = 0.0)
        ax1.invert_yaxis()
        ax1.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
        ax1.set_title('Correct Trials')

        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2.plot(psth_good_avg, color='#2f9c95', linewidth=3) 
        # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
        ax2.fill_between(range(len(psth_good_avg)), psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
        ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax2.set_ylabel('Average Value')
        ax2.set_xlabel('Time')

        # Plot the heatmap and line plot for incorrect trials
        ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
        sns.heatmap(psth_error.T, cbar=False, ax=ax3) #, center = 0.0)
        ax3.invert_yaxis()
        ax3.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
        ax3.set_title('Incorrect Trials')

        ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax2)
        ax4.plot(psth_error_avg, color='#d62828', linewidth=3)
        ax4.fill_between(range(len(psth_error_avg)), psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
        ax4.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax4.set_ylabel('Average Value')
        ax4.set_xlabel('Time')

        fig.suptitle(f'calcium_mad_{EVENT}_{mouse}_{date}_{region}_{eid}', y=1, fontsize=14)
        plt.tight_layout()
        # plt.savefig(f'/mnt/h0/kb/data/psth_npy/Fig02_{EVENT}_{mouse}_{date}_{region}_{eid}.png')
        plt.show() 
