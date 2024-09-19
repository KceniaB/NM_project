
""" 
2024June28
trying multiple sessions at once - code = MS12345 
MS12345 

WORKS! 
""" 

#imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
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
        session_path_behav = f'/home/kceniabougrova/Documents/nph/Behav_2024Mar20/{rec.mouse}/{rec.date}/001/'
        df_alldata = extract_all(session_path_behav)
        table_data = df_alldata[0]['table']
        trials = pd.DataFrame(table_data) 
    return eid, trials 

#%%
EXCLUDES = [0,1,10,12,15,21,25,52,59,
    71,78,90,97,107,115,116,145,151,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,
    220,229,237,238,241,264,268,306,307,339,365,367,369,375,378,381,384,393,396,397,405,411,415,418,419,423,438,442,444,448,455,456,479,482,484,486]  

IMIN = 0

test_01 = pd.read_parquet('/mnt/h0/kb/data/staged_data/01_recordings_sync.pqt') 





#####

######### 2nd loop, for the older sessions KB 16092024 
# Load the CSV file
df_goodsessions = pd.read_csv('/mnt/h0/kb/Mice performance tables 100 2.csv')

# Convert 'nph_file' to integers, keeping NaN values
df_goodsessions['nph_file'] = pd.to_numeric(df_goodsessions['nph_file'], errors='coerce').astype('Int64')

# Convert the 'date' column to datetime and standardize to 'yyyy-mm-dd' format
df_goodsessions['date'] = pd.to_datetime(df_goodsessions['date'], errors='coerce')

# Now, ensure all dates are in 'yyyy-mm-dd' format
df_goodsessions['date'] = df_goodsessions['date'].dt.strftime('%Y-%m-%d')

df_goodsessions['Date'] = df_goodsessions.date
df_goodsessions['Mouse'] = df_goodsessions.mouse

df1=df_goodsessions
test_01 = df1


#%%
EXCLUDES = []
IMIN = 0
excludes = []
#################
for i in range(len(test_01)): 
    try: 
        if i < IMIN:
            continue
        if i in EXCLUDES:
            continue

        EVENT = "stimOnTrigger_times"
        mouse = test_01.mouse[i] 
        date = test_01.date[i]
        if isinstance(date, pd.Timestamp):
            date = date.strftime('%Y-%m-%d')
        region = test_01.region[i]
        # eid = test_01.eid[i] 
        eid, df_trials = get_eid(mouse,date)
        print(f"{mouse} | {date} | {region} | {eid}")

        # Check if the desired file already exists
        fig_path = f'/mnt/h0/kb/data/psth_npy/Fig03_psth_{EVENT}_{mouse}_{date}_{region}_{eid}.png'

        if os.path.exists(fig_path):
            print(f"File {fig_path} already exists, skipping...")
            continue
        region=str(f"Region{region}G")
        try: 
            nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/001/alf/{region}/raw_photometry.pqt'
            df_nph = pd.read_parquet(nph_path)
        except:
            try:
                nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/002/alf/{region}/raw_photometry.pqt'
                df_nph = pd.read_parquet(nph_path)
            except:
                try:
                    nph_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/003/alf/{region}/raw_photometry.pqt'
                    df_nph = pd.read_parquet(nph_path)
                except:
                    print(f"Could not find raw_photometry.pqt in paths 001, 002, or 003 for mouse {mouse} on date {date}")
                    df_nph = None  # Optionally set df_nph to None or handle it appropriately

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
        # Process the calcium signal and add to df  
        # nph_j = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
        # df_nph["calcium"] = nph_j
        df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
        df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
        df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
        df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
        df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
        df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)

        array_timestamps_bpod = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
        event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
        idx_event = np.searchsorted(array_timestamps_bpod, event_test) #check idx where they would be included, in a sorted way 
        # print(idx_event) 

        """ create a column with the trial number in the nph df """
        df_nph["trial_number"] = 0 #create a new column for the trial_number 
        df_nph.loc[idx_event,"trial_number"]=1
        df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

        PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
        SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR

        sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
        n_trials = df_trials.shape[0]

        psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

        event_stimOnT = np.array(df_trials[EVENT]) #pick the feedback timestamps 

        stimOnT_idx = np.searchsorted(array_timestamps_bpod, event_stimOnT) #check idx where they would be included, in a sorted way 

        psth_idx += stimOnT_idx

        # # Clip the indices to be within the valid range, preserving the original shape
        # psth_idx_clipped = np.clip(psth_idx, 0, len(df_nph.calcium_photobleach.values) - 1)

        # # Now access the values using the clipped indices, keeping the original shape
        # photometry_s_1 = df_nph.calcium_photobleach.values[psth_idx_clipped]
        
        photometry_s_1 = df_nph.calcium_photobleach.values[psth_idx] 
        np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_1)
        photometry_s_2 = df_nph.isosbestic_photobleach.values[psth_idx] 
        np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_2)
        photometry_s_3 = df_nph.calcium_jove2019.values[psth_idx] 
        np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_3)
        photometry_s_4 = df_nph.isosbestic_jove2019.values[psth_idx] 
        np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_4)
        photometry_s_5 = df_nph.calcium_mad.values[psth_idx] 
        np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_5)
        photometry_s_6 = df_nph.isosbestic_mad.values[psth_idx] 
        np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_6)

        psth_good = df_nph.calcium_mad.values[psth_idx[:,(df_trials.feedbackType == 1)]]
        psth_error = df_nph.calcium_mad.values[psth_idx[:,(df_trials.feedbackType == -1)]]
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
        plt.savefig(f'/mnt/h0/kb/data/psth_npy/Fig02_{EVENT}_{mouse}_{date}_{region}_{eid}.png')
        plt.show()





        psth_good_1 = df_nph.calcium_photobleach.values[psth_idx[:,(df_trials.feedbackType == 1)]]
        psth_error_1 = df_nph.calcium_photobleach.values[psth_idx[:,(df_trials.feedbackType == -1)]]
        psth_good_avg_1 = psth_good_1.mean(axis=1)
        sem_good_1 = psth_good_1.std(axis=1) / np.sqrt(psth_good_1.shape[1])
        psth_error_avg_1 = psth_error_1.mean(axis=1)
        sem_error_1 = psth_error_1.std(axis=1) / np.sqrt(psth_error_1.shape[1]) 

        psth_good_2 = df_nph.calcium_jove2019.values[psth_idx[:,(df_trials.feedbackType == 1)]]
        psth_error_2 = df_nph.calcium_jove2019.values[psth_idx[:,(df_trials.feedbackType == -1)]]
        psth_good_avg_2 = psth_good_2.mean(axis=1)
        sem_good_2 = psth_good_2.std(axis=1) / np.sqrt(psth_good_2.shape[1])
        psth_error_avg_2 = psth_error_2.mean(axis=1)
        sem_error_2 = psth_error_2.std(axis=1) / np.sqrt(psth_error_2.shape[1])

        # Third block
        psth_good_3 = df_nph.calcium_mad.values[psth_idx[:,(df_trials.feedbackType == 1)]]
        psth_error_3 = df_nph.calcium_mad.values[psth_idx[:,(df_trials.feedbackType == -1)]]
        psth_good_avg_3 = psth_good_3.mean(axis=1)
        sem_good_3 = psth_good_3.std(axis=1) / np.sqrt(psth_good_3.shape[1])
        psth_error_avg_3 = psth_error_3.mean(axis=1)
        sem_error_3 = psth_error_3.std(axis=1) / np.sqrt(psth_error_3.shape[1])

        # Create the figure and gridspec
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(psth_good_avg_1, color='#0892a5', linewidth=3) 
        # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
        ax1.fill_between(range(len(psth_good_avg_1)), psth_good_avg_1 - sem_good_1, psth_good_avg_1 + sem_good_1, color='#0892a5', alpha=0.15) 
        ax1.plot(psth_error_avg_1, color='#d62828', linewidth=3)
        ax1.fill_between(range(len(psth_error_avg_1)), psth_error_avg_1 - sem_error_1, psth_error_avg_1 + sem_error_1, color='#d62828', alpha=0.15)
        ax1.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax1.set_ylabel('Average Value')
        ax1.set_xlabel('Time') 

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(psth_good_avg_2, color='#0892a5', linewidth=3) 
        # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
        ax2.fill_between(range(len(psth_good_avg_2)), psth_good_avg_2 - sem_good_2, psth_good_avg_2 + sem_good_2, color='#0892a5', alpha=0.15) 
        ax2.plot(psth_error_avg_2, color='#d62828', linewidth=3)
        ax2.fill_between(range(len(psth_error_avg_2)), psth_error_avg_2 - sem_error_2, psth_error_avg_2 + sem_error_2, color='#d62828', alpha=0.15)
        ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax2.set_ylabel('Average Value')
        ax2.set_xlabel('Time') 

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(psth_good_avg_3, color='#0892a5', linewidth=3) 
        # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
        ax3.fill_between(range(len(psth_good_avg_3)), psth_good_avg_3 - sem_good_3, psth_good_avg_3 + sem_good_3, color='#0892a5', alpha=0.15) 
        ax3.plot(psth_error_avg_3, color='#d62828', linewidth=3)
        ax3.fill_between(range(len(psth_error_avg_3)), psth_error_avg_3 - sem_error_3, psth_error_avg_3 + sem_error_3, color='#d62828', alpha=0.15)
        ax3.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax3.set_ylabel('Average Value')
        ax3.set_xlabel('Time') 

        fig.suptitle(f'{EVENT}_{mouse}_{date}_{region}_{eid}', y=1, fontsize=14)
        plt.tight_layout()
        plt.savefig(f'/mnt/h0/kb/data/psth_npy/Fig03_psth_{EVENT}_{mouse}_{date}_{region}_{eid}.png')
        plt.show() 
        
        print(f"DONE {mouse} | {date} | {region} | {eid}")
    except: 
        excludes.append(i)
        print("EXCLUDED: ",i)

# %%
