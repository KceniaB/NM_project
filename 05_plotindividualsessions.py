"""
Code created in order to plot each session from the complete mouse list
Excel file must have the following columns: 

"""

#%% 
"""
IMPORTS AND FUNCTIONS 
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib.colors as mcolors
from matplotlib.dates import date2num
from brainbox.io.one import SessionLoader
from ibldsp.utils import parabolic_max 
from iblphotometry.preprocessing import jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 
from one.api import ONE #always after the imports 
one = ONE(cache_dir="/mnt/h0/kb/data/one") 

#functions 
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
        session_path_behav = f'/home/kceniabougrova/Documents/nph/Behav_2024Mar20/{mouse}/{date}/001/'
        df_alldata = extract_all(session_path_behav)
        table_data = df_alldata[0]['table']
        trials = pd.DataFrame(table_data) 
    return eid, trials 

"""
useful
"""
""" 1. check nans in np arrays"""
# nan_indices = np.where(np.isnan(event_stimOnT))[0]
# # Print the indices with NaN values
# print("Indices with NaN in 'psth_idx':", nan_indices)

""" 2. fill nans """
# Extract 'stimOnTrigger_times' and 'goCueTrigger_times' from the DataFrame
event_stimOnT = np.array(df_trials[EVENT])  # Replace EVENT with 'stimOnTrigger_times'
event_goCueT = np.array(df_trials['goCueTrigger_times'])

# Trim the event_stimOnT array to match the desired length
event_stimOnT = event_stimOnT[:len(event_stimOnT)-1]

# Find the indices where 'event_stimOnT' has NaN values
nan_indices = np.where(np.isnan(event_stimOnT))[0]

# Check if there are any NaN values
if len(nan_indices) > 0:
    # Replace the NaN values with corresponding values from 'goCueTrigger_times'
    event_stimOnT[nan_indices] = event_goCueT[nan_indices]
    
    # Print the number of NaN values and their indices
    print(f"{len(nan_indices)} NaN values were found and replaced.")
    print(f"Indices with NaN: {nan_indices}")
else:
    print("No NaN values found in event_stimOnT.")

#%%
##############################################################
"""
LOAD DATA 
""" 
dtype = {'nph_bnc': int, 'region': int, 'NM': str, 'mouse': str} #nph_file not included, bcs there are rows with empty cells for this column, but should be int
df1 = pd.read_excel('/mnt/h0/kb/Mice performance tables 100.xlsx', 'A4_2024', dtype=dtype)
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
df1.date = df1.date.astype(str) 
# %%
##############################################################
"""
PROCESS DATA 
    1 - get mouse, date, region names
    2 - get eid and df_trials through get_eid function 
    3 - read the raw nph_nph 
            times already in bpod time, raw_isosbestic, raw_calcium 
    4 - add mouse info to df_nph and df_trials 
    5 - add allContrasts and allSContrasts
""" 
test_01 = df1
skipped_idxs = []
for i in range(len(test_01[0:100])): 
    try: 
        # test_01 = df1 
        # i=5
        EVENT = "feedback_times"

        """ 1. """
        mouse = test_01.mouse[i] 
        date = test_01.date[i]
        region = str(f"Region{test_01.region[i]}G")

        """ 2. """
        eid, df_trials = get_eid(mouse,date)
        print(f"{mouse} | {date} | {region} | {eid} | {i}") 

        """ 3. """
        # try: 
        #     nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/001/alf/{region}/raw_photometry.pqt'
        #     df_nph = pd.read_parquet(nph_path)
        # except:
        #     try:
        #         nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/002/alf/{region}/raw_photometry.pqt'
        #         df_nph = pd.read_parquet(nph_path)
        #     except:
        #         try:
        #             nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/003/alf/{region}/raw_photometry.pqt'
        #             df_nph = pd.read_parquet(nph_path)
        #         except:
        #             print(f"Could not find raw_photometry.pqt in paths 001, 002, or 003 for mouse {mouse} on date {date}")
        #             df_nph = None  # Optionally set df_nph to None or handle it appropriately

        def load_photometry_data(mouse, date, region, max_attempts=10):
            df_nph = None
            for i in range(1, max_attempts + 1):
                # Format the folder number with leading zeros
                folder_number = f"{i:03d}"
                nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/{folder_number}/alf/{region}/raw_photometry.pqt'
                
                if os.path.exists(nph_path):  # Check if the file exists
                    try:
                        df_nph = pd.read_parquet(nph_path)
                        print(f"Loaded raw_photometry.pqt from folder {folder_number}")
                        return df_nph
                    except Exception as e:
                        print(f"Error loading parquet from {nph_path}: {e}")
            
            print(f"Could not find raw_photometry.pqt in paths 001 to {max_attempts:03d} for mouse {mouse} on date {date}")
            return df_nph  # Return None if all attempts failed
        df_nph = load_photometry_data(mouse, date, region, max_attempts=10)

        """ 4. """
        df_nph["mouse"] = mouse
        df_nph["date"] = date
        df_nph["region"] = region
        df_nph["eid"] = eid 
        # create trialNumber
        df_trials['trialNumber'] = range(1, len(df_trials) + 1) 
        df_trials["mouse"] = mouse
        df_trials["date"] = date
        df_trials["region"] = region
        df_trials["eid"] = eid 

        # create allContrasts 
        idx = 2
        new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
        df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
        # create allSContrasts 
        df_trials['allSContrasts'] = df_trials['allContrasts']
        df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
        df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))

        # create reactionTime
        reactionTime = np.array((df_trials["firstMovement_times"])-(df_trials["stimOnTrigger_times"]))
        df_trials["reactionTime"] = reactionTime 
        responseTime = np.array((df_trials["feedback_times"])-(df_trials["stimOnTrigger_times"])) 
        df_trials["responseTime"] = responseTime 

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
        fs = round(1 / time_diffs.median())
        # Process the calcium signal and add to df  
        # nph_j = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
        # df_nph["calcium"] = nph_j
        df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
        df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
        df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
        df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
        df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
        df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)

        array_timestamps = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
        event_timestamps = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
        idx_event = np.searchsorted(array_timestamps, event_timestamps) #check idx where they would be included, in a sorted way 

        """ create a column with the trial number in the nph df """
        df_nph["trial_number"] = 0 #create a new column for the trial_number 
        df_nph.loc[idx_event,"trial_number"]=1
        df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

        PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
        SAMPLING_RATE = fs #not a constant: print(1/np.mean(np.diff(array_timestamps))) #sampling rate #acq_FR

        sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
        n_trials = df_trials.shape[0]

        # psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
        psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials-1)) 

        event_stimOnT = np.array(df_trials[EVENT]) #pick the feedback timestamps 
        event_stimOnT = event_stimOnT[0:len(event_stimOnT)-1] #KB added 20240327 CHECK WITH OW

        stimOnT_idx = np.searchsorted(array_timestamps, event_stimOnT) #check idx where they would be included, in a sorted way 

        psth_idx += stimOnT_idx


        ##############################################################
        """
        PLOTTING DATA 
            1 - 
        """ 


        photometry_s_3 = df_nph.calcium_jove2019.values[psth_idx] 
        plt.plot(np.mean(photometry_s_3, axis=1))
        plt.axvline(x=30, linestyle='dashed', color='black')
        plt.show()
    except: 
        skipped_idxs.append(i)

#%% 
# df_trials = df_trials.iloc[:-1] #be careful, only run 1ce 

# psth_good = df_nph.calcium_mad.values[psth_idx[:,(df_trials.feedbackType == 1)]]
# psth_error = df_nph.calcium_mad.values[psth_idx[:,(df_trials.feedbackType == -1)]]
# psth_good_avg = psth_good.mean(axis=1)
# sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
# psth_error_avg = psth_error.mean(axis=1)
# sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

# # Create the figure and gridspec
# fig = plt.figure(figsize=(10, 12))
# gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

# # Plot the heatmap and line plot for correct trials
# ax1 = fig.add_subplot(gs[0, 0])
# sns.heatmap(psth_good.T, cbar=False, ax=ax1) #, center = 0.0)
# ax1.invert_yaxis()
# ax1.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
# ax1.set_title('Correct Trials')

# ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
# ax2.plot(psth_good_avg, color='#2f9c95', linewidth=3) 
# # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
# ax2.fill_between(range(len(psth_good_avg)), psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
# ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
# ax2.set_ylabel('Average Value')
# ax2.set_xlabel('Time')

# # Plot the heatmap and line plot for incorrect trials
# ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
# sns.heatmap(psth_error.T, cbar=False, ax=ax3) #, center = 0.0)
# ax3.invert_yaxis()
# ax3.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
# ax3.set_title('Incorrect Trials')

# ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax2)
# ax4.plot(psth_error_avg, color='#d62828', linewidth=3)
# ax4.fill_between(range(len(psth_error_avg)), psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
# ax4.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
# ax4.set_ylabel('Average Value')
# ax4.set_xlabel('Time')

# fig.suptitle(f'calcium_mad_{EVENT}_{mouse}_{date}_{region}_{eid}', y=1, fontsize=14)
# plt.tight_layout()

#%%
# plot all sessions by mouse 

# Iterate over each mouse
for mouse, sessions in mice_sessions:
    sessions = sessions.sort_values('date').reset_index(drop=True)  # Sort sessions by date
    num_sessions = len(sessions)
    num_plots = (num_sessions // sessions_per_plot) + (num_sessions % sessions_per_plot != 0)  # Calculate number of plots needed

    # Initialize variables to track the global y-limits across all sessions for this mouse
    global_ymin, global_ymax = np.inf, -np.inf
    # Step 1: Calculate the global y-limits (min/max) for all sessions of this mouse
    for plot_index in range(num_plots):
        start_index = plot_index * sessions_per_plot
        end_index = min(start_index + sessions_per_plot, num_sessions)
        current_sessions = sessions.iloc[start_index:end_index]
        current_sessions['date'] = pd.to_datetime(current_sessions['date'], errors='coerce')   

        for i, row in current_sessions.iterrows():
            try:
                if pd.notna(row['date']):  # Check if the date is valid
                    date = row['date'].strftime('%Y-%m-%d')
                    print(date)
                region = row['region']
                eid, df_trials = get_eid(mouse, date)
                nm = row['NM']

                print(f"{mouse} | {date} | {region} | {eid} | {i}") 

                # create trialNumber
                df_trials['trialNumber'] = range(1, len(df_trials) + 1) 
                df_trials["mouse"] = mouse
                df_trials["date"] = date
                df_trials["region"] = region
                df_trials["eid"] = eid 


                # Create allContrasts and allSContrasts
                idx = 2
                new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight'])
                df_trials.insert(loc=idx, column='allContrasts', value=new_col)
                df_trials['allSContrasts'] = df_trials['allContrasts']
                df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
                df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))

                # create reactionTime
                reactionTime = np.array((df_trials["firstMovement_times"])-(df_trials["stimOnTrigger_times"]))
                df_trials["reactionTime"] = reactionTime 
                responseTime = np.array((df_trials["feedback_times"])-(df_trials["stimOnTrigger_times"])) 
                df_trials["responseTime"] = responseTime 

                # Filter df_trials for feedback types and contrast conditions
                feedback1_trials = df_trials[df_trials['feedbackType'] == 1]
                feedback_minus1_trials = df_trials[df_trials['feedbackType'] == -1]
                contrast_high = df_trials[df_trials['allContrasts'].isin([1, 0.5, 0.25])]
                contrast_low = df_trials[df_trials['allContrasts'].isin([0, 0.0625])]

                # Determine the EVENT value based on the plot index
                if plot_index < 2:
                    EVENT = 'feedback_times'
                else:
                    EVENT = 'stimOnTrigger_times'

                try: 
                    def load_photometry_data(mouse, date, region, max_attempts=10):
                        df_nph = None
                        for i in range(1, max_attempts + 1):
                            # Format the folder number with leading zeros
                            folder_number = f"{i:03d}"
                            nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/{folder_number}/alf/{region}/raw_photometry.pqt'
                            
                            if os.path.exists(nph_path):  # Check if the file exists
                                try:
                                    df_nph = pd.read_parquet(nph_path)
                                    print(f"Loaded raw_photometry.pqt from folder {folder_number}")
                                    return df_nph
                                except Exception as e:
                                    print(f"Error loading parquet from {nph_path}: {e}")
                        
                        print(f"Could not find raw_photometry.pqt in paths 001 to {max_attempts:03d} for mouse {mouse} on date {date}")
                        return df_nph  # Return None if all attempts failed
                    df_nph = load_photometry_data(mouse, date, region, max_attempts=10)

                    """ 4. """
                    df_nph["mouse"] = mouse
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
                    fs = round(1 / time_diffs.median())
                    # Process the calcium signal and add to df  
                    # nph_j = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
                    # df_nph["calcium"] = nph_j
                    df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
                    df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
                    df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
                    df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
                    df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
                    df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)

                    array_timestamps = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
                    event_timestamps = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
                    idx_event = np.searchsorted(array_timestamps, event_timestamps) #check idx where they would be included, in a sorted way 

                    """ create a column with the trial number in the nph df """
                    df_nph["trial_number"] = 0 #create a new column for the trial_number 
                    df_nph.loc[idx_event,"trial_number"]=1
                    df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

                    PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
                    SAMPLING_RATE = fs #not a constant: print(1/np.mean(np.diff(array_timestamps))) #sampling rate #acq_FR

                    sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
                    n_trials = df_trials.shape[0]

                    # psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
                    psth_idx2 = np.tile(sample_window[:,np.newaxis], (1, n_trials-1)) 

                    event_stimOnT = np.array(df_trials[EVENT]) #pick the feedback timestamps 
                    event_stimOnT = event_stimOnT[0:len(event_stimOnT)-1] #KB added 20240327 CHECK WITH OW

                    stimOnT_idx = np.searchsorted(array_timestamps, event_stimOnT) #check idx where they would be included, in a sorted way 

                    psth_idx2 += stimOnT_idx

                    psth_idx = df_nph.isosbestic_mad.values[psth_idx2] 
                except: 
                    path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_mad_{EVENT}_etc/'
                    path = path_initial + f'preprocess_isosbestic_mad_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy' 
                    print()

                    # Load psth_idx from file
                    psth_idx = np.load(path)


                # Ensure psth_idx has enough trials before splitting into chunks
                total_trials = psth_idx.shape[1]  # Number of columns (trials)
                chunk_size = 90

                # Split the psth_idx array into available full chunks (0-90, 90-180, ...)
                psth_chunks = [psth_idx[:, i:i + chunk_size] for i in range(0, total_trials, chunk_size)]

                # Loop through each of the trial chunks
                for psth_chunk in psth_chunks:
                    # Filter psth_idx based on feedbackType=1 indices within this chunk
                    feedback1_indices = feedback1_trials.index.tolist()
                    chunk_start = psth_chunks.index(psth_chunk) * chunk_size
                    chunk_end = chunk_start + chunk_size
                    chunk_indices_1 = [i for i in feedback1_indices if chunk_start <= i < chunk_end]

                    if chunk_indices_1:
                        psth_idx_filtered = psth_chunk[:, [i - chunk_start for i in chunk_indices_1]]
                        mean_psth = np.mean(psth_idx_filtered, axis=1)
                        stderr_psth = np.std(psth_idx_filtered, axis=1) / np.sqrt(psth_idx_filtered.shape[1])
                        global_ymin = min(global_ymin, np.min(mean_psth - stderr_psth))
                        global_ymax = max(global_ymax, np.max(mean_psth + stderr_psth))

                    # Filter psth_idx based on feedbackType=-1 indices within this chunk
                    feedback_minus1_indices = feedback_minus1_trials.index.tolist()
                    chunk_indices_minus1 = [i for i in feedback_minus1_indices if chunk_start <= i < chunk_end]

                    if chunk_indices_minus1:
                        psth_idx_filtered = psth_chunk[:, [i - chunk_start for i in chunk_indices_minus1]]
                        mean_psth = np.mean(psth_idx_filtered, axis=1)
                        stderr_psth = np.std(psth_idx_filtered, axis=1) / np.sqrt(psth_idx_filtered.shape[1])
                        global_ymin = min(global_ymin, np.min(mean_psth - stderr_psth))
                        global_ymax = max(global_ymax, np.max(mean_psth + stderr_psth))

                    # Filter psth_idx based on contrast conditions within this chunk
                    contrast_high_indices = contrast_high.index.tolist()
                    contrast_low_indices = contrast_low.index.tolist()

                    chunk_indices_high = [i for i in contrast_high_indices if chunk_start <= i < chunk_end]
                    chunk_indices_low = [i for i in contrast_low_indices if chunk_start <= i < chunk_end]

                    if chunk_indices_high:
                        psth_idx_filtered = psth_chunk[:, [i - chunk_start for i in chunk_indices_high]]
                        mean_psth = np.mean(psth_idx_filtered, axis=1)
                        stderr_psth = np.std(psth_idx_filtered, axis=1) / np.sqrt(psth_idx_filtered.shape[1])
                        global_ymin = min(global_ymin, np.min(mean_psth - stderr_psth))
                        global_ymax = max(global_ymax, np.max(mean_psth + stderr_psth))

                    if chunk_indices_low:
                        psth_idx_filtered = psth_chunk[:, [i - chunk_start for i in chunk_indices_low]]
                        mean_psth = np.mean(psth_idx_filtered, axis=1)
                        stderr_psth = np.std(psth_idx_filtered, axis=1) / np.sqrt(psth_idx_filtered.shape[1])
                        global_ymin = min(global_ymin, np.min(mean_psth - stderr_psth))
                        global_ymax = max(global_ymax, np.max(mean_psth + stderr_psth))

            except Exception as e:
                print(f"ERROR with session {i}: {e}")
                EXCLUDES.append(i)



    # Step 2: Plot the sessions with the same y-limits across all subplots for this mouse
    for plot_index in range(num_plots):
        # Create 4 rows x 4 columns of subplots for each session group
        fig, axes = plt.subplots(4, 4, figsize=(18, 16))  # Adjust figure size for 4 rows

        # Set the main title
        fig.suptitle(f"Mouse = {mouse} (Session Group {plot_index + 1}) isos mad All Conditions", fontsize=16)

        # Define session indices for the current plot
        start_index = plot_index * sessions_per_plot
        end_index = min(start_index + sessions_per_plot, num_sessions)
        current_sessions = sessions.iloc[start_index:end_index]
        current_sessions['date'] = pd.to_datetime(current_sessions['date'], errors='coerce')   

        # Discretize the colormap
        cmap = plt.get_cmap('viridis', sessions_per_plot)  # Get the desired number of colors

        # Dictionary to map color to label
        color_to_label = {}

        # Loop over the sessions for this mouse and plot each session
        for session_idx, (i, row) in enumerate(current_sessions.iterrows()):
            try:
                if pd.notna(row['date']):  # Check if the date is valid
                    date = row['date'].strftime('%Y-%m-%d')
                    print(date)
                region = row['region']
                eid, df_trials = get_eid(mouse, date)

                # Create allContrasts and allSContrasts
                idx = 2
                new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight'])
                df_trials.insert(loc=idx, column='allContrasts', value=new_col)
                df_trials['allSContrasts'] = df_trials['allContrasts']
                df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
                df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))

                # Create reactionTime
                reactionTime = np.array((df_trials["firstMovement_times"]) - (df_trials["stimOnTrigger_times"]))
                df_trials["reactionTime"] = reactionTime

                # Filter df_trials for feedback types and contrast conditions
                feedback1_trials = df_trials[df_trials['feedbackType'] == 1]
                feedback_minus1_trials = df_trials[df_trials['feedbackType'] == -1]
                contrast_high = df_trials[df_trials['allContrasts'].isin([1, 0.5, 0.25])]
                contrast_low = df_trials[df_trials['allContrasts'].isin([0, 0.0625])]

                # Determine the EVENT value based on the row index
                for row_idx in range(4):
                    if row_idx < 2:
                        EVENT = 'feedback_times'
                    else:
                        EVENT = 'stimOnTrigger_times'


                    try: 
                        def load_photometry_data(mouse, date, region, max_attempts=10):
                            df_nph = None
                            for i in range(1, max_attempts + 1):
                                # Format the folder number with leading zeros
                                folder_number = f"{i:03d}"
                                nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/{folder_number}/alf/{region}/raw_photometry.pqt'
                                
                                if os.path.exists(nph_path):  # Check if the file exists
                                    try:
                                        df_nph = pd.read_parquet(nph_path)
                                        print(f"Loaded raw_photometry.pqt from folder {folder_number}")
                                        return df_nph
                                    except Exception as e:
                                        print(f"Error loading parquet from {nph_path}: {e}")
                            
                            print(f"Could not find raw_photometry.pqt in paths 001 to {max_attempts:03d} for mouse {mouse} on date {date}")
                            return df_nph  # Return None if all attempts failed
                        df_nph = load_photometry_data(mouse, date, region, max_attempts=10)

                        """ 4. """
                        df_nph["mouse"] = mouse
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
                        fs = round(1 / time_diffs.median())
                        # Process the calcium signal and add to df  
                        # nph_j = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
                        # df_nph["calcium"] = nph_j
                        df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
                        df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
                        df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
                        df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
                        df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
                        df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)

                        array_timestamps = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
                        event_timestamps = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
                        idx_event = np.searchsorted(array_timestamps, event_timestamps) #check idx where they would be included, in a sorted way 

                        """ create a column with the trial number in the nph df """
                        df_nph["trial_number"] = 0 #create a new column for the trial_number 
                        df_nph.loc[idx_event,"trial_number"]=1
                        df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

                        PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
                        SAMPLING_RATE = fs #not a constant: print(1/np.mean(np.diff(array_timestamps))) #sampling rate #acq_FR

                        sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
                        n_trials = df_trials.shape[0]

                        # psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
                        psth_idx2 = np.tile(sample_window[:,np.newaxis], (1, n_trials-1)) 

                        event_stimOnT = np.array(df_trials[EVENT]) #pick the feedback timestamps 
                        event_stimOnT = event_stimOnT[0:len(event_stimOnT)-1] #KB added 20240327 CHECK WITH OW

                        stimOnT_idx = np.searchsorted(array_timestamps, event_stimOnT) #check idx where they would be included, in a sorted way 

                        psth_idx2 += stimOnT_idx

                        psth_idx = df_nph.isosbestic_mad.values[psth_idx2] 
                        
                    except: 
                        path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_mad_{EVENT}_etc/'
                        path = path_initial + f'preprocess_isosbestic_mad_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy' 
                        print("except code")

                        # Load psth_idx from file
                        psth_idx = np.load(path)


                    # Ensure psth_idx has enough trials before splitting into chunks
                    total_trials = psth_idx.shape[1]  # Number of columns (trials)
                    chunk_size = 90

                    # Split the psth_idx array into available full chunks (0-90, 90-180, ...)
                    psth_chunks = [psth_idx[:, i:i + chunk_size] for i in range(0, total_trials, chunk_size)]

                    # Loop through each of the 4 trial ranges and plot on respective axes
                    for ax_idx in range(4):
                        if ax_idx >= len(psth_chunks):
                            continue  # If fewer than 4 chunks, skip extra axes

                        psth_chunk = psth_chunks[ax_idx]

                        # Select the subplot axis based on the row index and column index
                        ax = axes[row_idx, ax_idx]

                        chunk_start = ax_idx * chunk_size
                        chunk_end = chunk_start + chunk_size

                        # Plot feedbackType = 1 on the top rows (axes[0, ax_idx] and axes[1, ax_idx])
                        if row_idx == 0:
                            trials_indices = feedback1_trials.index
                            title = f"Feedback 1: Trials {chunk_start}-{chunk_end}"
                        elif row_idx == 1:
                            trials_indices = feedback_minus1_trials.index
                            title = f"Feedback -1: Trials {chunk_start}-{chunk_end}"
                        elif row_idx == 2:
                            trials_indices = contrast_high.index
                            title = f"High Contrasts: Trials {chunk_start}-{chunk_end}"
                        elif row_idx == 3:
                            trials_indices = contrast_low.index
                            title = f"Low Contrasts: Trials {chunk_start}-{chunk_end}"

                        filtered_indices = [i for i in trials_indices if chunk_start <= i < chunk_end]
                        if filtered_indices:
                            psth_idx_filtered = psth_chunk[:, [i - chunk_start for i in filtered_indices]]
                            mean_psth = np.mean(psth_idx_filtered, axis=1)
                            stderr_psth = np.std(psth_idx_filtered, axis=1) / np.sqrt(psth_idx_filtered.shape[1])

                            line, = ax.plot(np.arange(len(mean_psth)), mean_psth, color=cmap(session_idx % sessions_per_plot))
                            ax.fill_between(np.arange(len(mean_psth)), mean_psth - stderr_psth, mean_psth + stderr_psth,
                                            color=cmap(session_idx % sessions_per_plot), alpha=0.1)
                            ax.axvline(x=29, color='black', linestyle='dashed')
                            ax.set_title(title)
                            ax.set_ylim(global_ymin, global_ymax)
                            ax.set_xlabel(f"time since {EVENT} (s)")
                            ax.set_ylabel("Processed calcium signal")

                first_360_trials = df_trials.head(360)
                num_unique_contrasts = len(first_360_trials['allContrasts'].unique())

                color_to_label[cmap(session_idx % sessions_per_plot)] = f"Session {i} ({date})\nRegion: {region}\nContrasts: {num_unique_contrasts}"

            except Exception as e:
                print(f"ERROR with session {i}: {e}")
                EXCLUDES.append(i)

        # Add a shared legend at the bottom of the figure
        if color_to_label:
            handles = [plt.Line2D([0], [0], color=color, lw=2) for color in color_to_label.keys()]
            labels = [color_to_label[color] for color in color_to_label.keys()]
            fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, -0.1), fontsize=10)
        else:
            print("No valid session data to create a legend.")

        # Adjust layout to make space for the legend
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make room for the legend

        filename = f"mouse_{mouse}_session_group_{plot_index + 1}_mad_isosbestic_all_conditions.png"
        plt.savefig(filename, bbox_inches='tight')  # Save with tight bounding box to avoid clipping

        plt.show()
        # Close the figure to release memory
        plt.close(fig)
print(EXCLUDES)





#%%


        i=21
        EVENT = "feedback_times"

        """ 1. """
        mouse = test_01.mouse[i] 
        date = test_01.date[i]
        region = str(f"Region{test_01.region[i]}G")

        """ 2. """
        eid, df_trials = get_eid(mouse,date)
        print(f"{mouse} | {date} | {region} | {eid} | {i}") 

        """ 3. """
        # try: 
        #     nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/001/alf/{region}/raw_photometry.pqt'
        #     df_nph = pd.read_parquet(nph_path)
        # except:
        #     try:
        #         nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/002/alf/{region}/raw_photometry.pqt'
        #         df_nph = pd.read_parquet(nph_path)
        #     except:
        #         try:
        #             nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/003/alf/{region}/raw_photometry.pqt'
        #             df_nph = pd.read_parquet(nph_path)
        #         except:
        #             print(f"Could not find raw_photometry.pqt in paths 001, 002, or 003 for mouse {mouse} on date {date}")
        #             df_nph = None  # Optionally set df_nph to None or handle it appropriately

        def load_photometry_data(mouse, date, region, max_attempts=10):
            df_nph = None
            for i in range(1, max_attempts + 1):
                # Format the folder number with leading zeros
                folder_number = f"{i:03d}"
                nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/{folder_number}/alf/{region}/raw_photometry.pqt'
                
                if os.path.exists(nph_path):  # Check if the file exists
                    try:
                        df_nph = pd.read_parquet(nph_path)
                        print(f"Loaded raw_photometry.pqt from folder {folder_number}")
                        return df_nph
                    except Exception as e:
                        print(f"Error loading parquet from {nph_path}: {e}")
            
            print(f"Could not find raw_photometry.pqt in paths 001 to {max_attempts:03d} for mouse {mouse} on date {date}")
            return df_nph  # Return None if all attempts failed
        df_nph = load_photometry_data(mouse, date, region, max_attempts=10)

        """ 4. """
        df_nph["mouse"] = mouse
        df_nph["date"] = date
        df_nph["region"] = region
        df_nph["eid"] = eid 
        # create trialNumber
        df_trials['trialNumber'] = range(1, len(df_trials) + 1) 
        df_trials["mouse"] = mouse
        df_trials["date"] = date
        df_trials["region"] = region
        df_trials["eid"] = eid 

        # create allContrasts 
        idx = 2
        new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
        df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
        # create allSContrasts 
        df_trials['allSContrasts'] = df_trials['allContrasts']
        df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
        df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))

        # create reactionTime
        reactionTime = np.array((df_trials["firstMovement_times"])-(df_trials["stimOnTrigger_times"]))
        df_trials["reactionTime"] = reactionTime 
        responseTime = np.array((df_trials["feedback_times"])-(df_trials["stimOnTrigger_times"])) 
        df_trials["responseTime"] = responseTime 

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
        fs = round(1 / time_diffs.median())
        # Process the calcium signal and add to df  
        # nph_j = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
        # df_nph["calcium"] = nph_j
        df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
        df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
        df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
        df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
        df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
        df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)

        array_timestamps = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
        event_timestamps = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
        idx_event = np.searchsorted(array_timestamps, event_timestamps) #check idx where they would be included, in a sorted way 

        """ create a column with the trial number in the nph df """
        df_nph["trial_number"] = 0 #create a new column for the trial_number 
        df_nph.loc[idx_event,"trial_number"]=1
        df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

        PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
        SAMPLING_RATE = fs #not a constant: print(1/np.mean(np.diff(array_timestamps))) #sampling rate #acq_FR

        sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
        n_trials = df_trials.shape[0]

        # psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
        psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials-1)) 

        event_stimOnT = np.array(df_trials[EVENT]) #pick the feedback timestamps 
        event_stimOnT = event_stimOnT[0:len(event_stimOnT)-1] #KB added 20240327 CHECK WITH OW

        stimOnT_idx = np.searchsorted(array_timestamps, event_stimOnT) #check idx where they would be included, in a sorted way 

        psth_idx += stimOnT_idx


        ##############################################################
        """
        PLOTTING DATA 
            1 - 
        """ 


        photometry_s_3 = df_nph.calcium_jove2019.values[psth_idx] 
        plt.plot(np.mean(photometry_s_3, axis=1))
        plt.axvline(x=30, linestyle='dashed', color='black')
        plt.show()