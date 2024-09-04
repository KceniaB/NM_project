
""" 
2024June28
trying multiple sessions at once - code = MS12345 
MS12345 
""" 

#imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from ibldsp.utils import parabolic_max 
from iblphotometry.preprocessing import jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 
import iblphotometry.kcenia as kcenia
import ibldsp.utils
from pathlib import Path
from brainbox.io.one import SessionLoader 
import scipy.signal
import ibllib.plots
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

""" THIS WAS PART OF THE PREV VERSION, WITHOUT THE 01/02/03 AND NOW AFTER THE ADDITION OF OLDER SESSIONS """
test_01 = pd.read_parquet('/mnt/h0/kb/data/staged_data/01_recordings_sync.pqt') 






EXCLUDES = [0,1,10,12,15,21,25,52,59,78,90,97,107,115,116,145,151,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,229,237,238,241,264,268,306,307,339,365,367,369,375,378,381,384,393,396,397,405,411,415,418,419,423,438,442,444,448,455,456,479,482,484,486]  

IMIN = 0

#%%
for i in range(len(test_01)): 
    if i < IMIN:
        continue
    if i in EXCLUDES:
        continue

    EVENT = "feedback_times"
    mouse = test_01.mouse[i] 
    date = test_01.date[i]
    if isinstance(date, pd.Timestamp):
        date = date.strftime('%Y-%m-%d')
    region = test_01.region[i]
    eid = test_01.eid[i] 
    print(f"{mouse} | {date} | {region} | {eid}")
    eid, df_trials = get_eid(mouse,date)
    try: 
        nph_path = '/mnt/h0/kb/data/one/mainenlab/Subjects/'+mouse+'/'+date+'/001/alf/'+region+'/raw_photometry.pqt' 
        df_nph = pd.read_parquet(nph_path)
    except: 
        nph_path = '/mnt/h0/kb/data/one/mainenlab/Subjects/'+mouse+'/'+date+'/002/alf/'+region+'/raw_photometry.pqt'
        df_nph = pd.read_parquet(nph_path)
    
    df_nph["mouse"] = mouse
    df_nph["date"] = date
    df_nph["region"] = region
    df_nph["eid"] = eid 

    plt.rcParams["figure.figsize"] = (16,6)
    plt.plot(df_nph.times, df_nph.raw_calcium, linewidth=0.5)
    xcoords = df_trials.goCueTrigger_times
    for xc in zip(xcoords):
        plt.axvline(x=xc, color='blue',linewidth=0.2, alpha=0.8) 
    plt.show()

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

    
    plt.rcParams["figure.figsize"] = (16,6)
    plt.plot(df_nph.times, df_nph.calcium_mad, linewidth=0.2, color="teal")
    xcoords = df_trials.goCueTrigger_times
    for xc in zip(xcoords):
        plt.axvline(x=xc, color='black',linewidth=0.15, alpha=0.8) 
    plt.title("nph calcium and goCueTrigger_times")
    plt.show() 

    # Set up the plot
    fig, axs = plt.subplots(3, 2, figsize=(16, 18), sharey='row')
    # First column (calcium)
    axs[0, 0].plot(df_nph['times'], df_nph["calcium_photobleach"], linewidth=0.2, color="teal")
    for xc in df_trials['goCueTrigger_times']:
        axs[0, 0].axvline(x=xc, color='black', linewidth=0.15, alpha=0.8)
    axs[0, 0].set_title("calcium_photobleach and goCueTrigger_times")
    # Second column (isosbestic)
    axs[0, 1].plot(df_nph['times'], df_nph["isosbestic_photobleach"], linewidth=0.2, color="purple")
    for xc in df_trials['goCueTrigger_times']:
        axs[0, 1].axvline(x=xc, color='black', linewidth=0.15, alpha=0.8)
    axs[0, 1].set_title("isosbestic_photobleach and goCueTrigger_times")
    # First column (calcium)
    axs[1, 0].plot(df_nph['times'], df_nph["calcium_jove2019"], linewidth=0.2, color="teal")
    for xc in df_trials['goCueTrigger_times']:
        axs[1, 0].axvline(x=xc, color='black', linewidth=0.15, alpha=0.8)
    axs[1, 0].set_title("calcium_jove2019 and goCueTrigger_times")
    # Second column (isosbestic)
    axs[1, 1].plot(df_nph['times'], df_nph["isosbestic_jove2019"], linewidth=0.2, color="purple")
    for xc in df_trials['goCueTrigger_times']:
        axs[1, 1].axvline(x=xc, color='black', linewidth=0.15, alpha=0.8)
    axs[1, 1].set_title("isosbestic_jove2019 and goCueTrigger_times")
    # First column (calcium)
    axs[2, 0].plot(df_nph['times'], df_nph["calcium_mad"], linewidth=0.2, color="teal")
    for xc in df_trials['goCueTrigger_times']:
        axs[2, 0].axvline(x=xc, color='black', linewidth=0.15, alpha=0.8)
    axs[2, 0].set_title("calcium_mad and goCueTrigger_times")
    # Second column (isosbestic)
    axs[2, 1].plot(df_nph['times'], df_nph["isosbestic_mad"], linewidth=0.2, color="purple")
    for xc in df_trials['goCueTrigger_times']:
        axs[2, 1].axvline(x=xc, color='black', linewidth=0.15, alpha=0.8)
    axs[2, 1].set_title("isosbestic_mad and goCueTrigger_times")

    fig.suptitle(f'preprocess_{mouse}_{date}_{region}_{eid}', y=1, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'/mnt/h0/kb/data/psth_npy/Fig_01_preprocess_{mouse}_{date}_{region}_{eid}.png')
    plt.show()




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

    event_feedback = np.array(df_trials[EVENT]) #pick the feedback timestamps 

    feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

    psth_idx += feedback_idx

    photometry_feedback_1 = df_nph.calcium_photobleach.values[psth_idx] 
    np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_feedback_1)
    photometry_feedback_2 = df_nph.isosbestic_photobleach.values[psth_idx] 
    np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_feedback_2)
    photometry_feedback_3 = df_nph.calcium_jove2019.values[psth_idx] 
    np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_feedback_3)
    photometry_feedback_4 = df_nph.isosbestic_jove2019.values[psth_idx] 
    np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_feedback_4)
    photometry_feedback_5 = df_nph.calcium_mad.values[psth_idx] 
    plt.rcParams["figure.figsize"] = (8,8)
    plt.plot(photometry_feedback_5,alpha=0.1, linewidth=0.3, color='#15514A')
    average_values = np.mean(photometry_feedback_5, axis=1)
    plt.plot(average_values, linewidth=3, color='#15514A')
    plt.axvline(x=30, color="black", linestyle="dashed")
    plt.title(f'psthidx_calcium_mad_{EVENT}_{mouse}_{date}_{region}')
    plt.show() 
    np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_feedback_5)
    photometry_feedback_6 = df_nph.isosbestic_mad.values[psth_idx] 
    np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_feedback_6)

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

    fig.suptitle(f'calcium_mad_{mouse}_{date}_{region}_{eid}', y=1, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'/mnt/h0/kb/data/psth_npy/Fig02_{mouse}_{date}_{region}_{eid}.png')
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

"""delete(?most likely)"""
    # # Set up the plot
    # fig, axs = plt.subplots(3, 2, figsize=(16, 18), sharey='row')
    # # First column (calcium)
    # axs[0, 0].plot(df_nph['times'], df_nph["calcium_photobleach"], linewidth=0.2, color="#0e9594") 

    # axs[0, 0].plot(photometry_feedback_1,alpha=0.1, linewidth=0.2, color='#15514A')
    # average_values = np.mean(photometry_feedback_5, axis=1)
    # axs[0, 0].plot(average_values, linewidth=3, color='#15514A')
    # axs[0, 0].axvline(x=30, color="black", linestyle="dashed")
    # plt.title(f'psthidx_calcium_mad_{EVENT}_{mouse}_{date}_{region}')
    # plt.show() 

    # # Second column (isosbestic)
    # axs[0, 1].plot(df_nph['times'], df_nph["isosbestic_photobleach"], linewidth=0.2, color="purple")
    # axs[0, 1].set_title("isosbestic_photobleach and goCueTrigger_times")
    # # First column (calcium)
    # axs[1, 0].plot(df_nph['times'], df_nph["calcium_jove2019"], linewidth=0.2, color="teal")
    # axs[1, 0].set_title("calcium_jove2019 and goCueTrigger_times")
    # # Second column (isosbestic)
    # axs[1, 1].plot(df_nph['times'], df_nph["isosbestic_jove2019"], linewidth=0.2, color="purple")
    # axs[1, 1].set_title("isosbestic_jove2019 and goCueTrigger_times")
    # # First column (calcium)
    # axs[2, 0].plot(df_nph['times'], df_nph["calcium_mad"], linewidth=0.2, color="teal")
    # axs[2, 0].set_title("calcium_mad and goCueTrigger_times")
    # # Second column (isosbestic)
    # axs[2, 1].plot(df_nph['times'], df_nph["isosbestic_mad"], linewidth=0.2, color="purple")
    # axs[2, 1].set_title("isosbestic_mad and goCueTrigger_times")

    # fig.suptitle(f'preprocess_{mouse}_{date}_{region}_{eid}', y=1, fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f'/mnt/h0/kb/data/psth_npy/Fig_03_psth_{EVENT}_{mouse}_{date}_{region}_{eid}.png')
    # plt.show()

    df_trials.to_csv(f'/mnt/h0/kb/data/psth_npy/df_trials_{EVENT}_{mouse}_{date}_{region}_{eid}.csv') 
    df_nph.to_csv(f'/mnt/h0/kb/data/psth_npy/df_nph_{EVENT}_{mouse}_{date}_{region}_{eid}.csv') 




#%% 
# %% ##################################################################################################################
##################################################################################################################
##################################################################################################################

"""
18July2024
KB loop through the good sessions 
GOOD SESSIONS in excel MICE AND SESSIONS 

"""
test_02 = pd.read_csv('/home/ibladmin/Downloads/Mice_GOOD_sorted.csv') 
test_02['Date'] = pd.to_datetime(test_02['Date'], format='%m/%d/%Y')
test_03 = test_02[['Mouse', 'Date', 'NM', 'region']] 
EVENT = 'feedback_times'
# path = '/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_feedback_times_etc/' 
# path2 = path+(f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy')


EXCLUDES = []  

IMIN = 0

# Initialize empty containers
psth_combined = None
df_trials_combined = pd.DataFrame()

# test_04 = test_03[test_03["NM"]=="DA"].reset_index(drop=True)
# test_04 = test_03[test_03["NM"]=="5HT"].reset_index(drop=True) 
# test_04 = test_03[test_03["NM"]=="NE"].reset_index(drop=True)
test_04 = test_03[test_03["NM"]=="ACh"].reset_index(drop=True)

for i in range(len(test_04)): 
    if i < IMIN:
        continue
    if i in EXCLUDES:
        continue
    EVENT = "feedback_times"
    mouse = test_04.Mouse[i] 
    date = test_04.Date[i]
    if isinstance(date, pd.Timestamp):
        date = date.strftime('%Y-%m-%d')
    region = test_04.region[i]
    eid, df_trials = get_eid(mouse,date)
    print(f"{mouse} | {date} | {region} | {eid}")
    df_trials['trialNumber'] = range(1, len(df_trials) + 1) 
    df_trials["mouse"] = mouse
    df_trials["date"] = date
    df_trials["region"] = region
    df_trials["eid"] = eid 

    path_initial = '/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_feedback_times_etc/' 
    path = path_initial + f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy'

    # Load psth_idx from file
    psth_idx = np.load(path)

    # Concatenate psth_idx arrays
    if psth_combined is None:
        psth_combined = psth_idx
    else:
        psth_combined = np.hstack((psth_combined, psth_idx))

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

    # Concatenate df_trials DataFrames
    df_trials_combined = pd.concat([df_trials_combined, df_trials], axis=0)

    # Reset index of the combined DataFrame
    df_trials_combined.reset_index(drop=True, inplace=True)

    # Print shapes to verify
    print("Shape of psth_combined:", psth_combined.shape)
    print("Shape of df_trials_combined:", df_trials_combined.shape)
#%%
    ##################################################################################################
    # PLOT 
    psth_good = psth_idx[:,(df_trials.feedbackType == 1)]
    psth_error = psth_idx[:,(df_trials.feedbackType == -1)]
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
    plt.show()
    ##################################################################################################


#%%
np.save('/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_feedback_times_etc/RESULTS/jove2019_psth_combined_ACh.npy', psth_combined)
df_trials_combined.to_parquet('/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_feedback_times_etc/RESULTS/jove2019_df_trials_combined_ACh.pqt')























#%% 
##########################################################################################################################
##########################################################################################################################
""" 
FOR OLDER DATA, following the 01 02 03 04 05... 
"""
##########################################################################################################################


""" 
2024Aug21
""" 

#imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from ibldsp.utils import parabolic_max 
from iblphotometry.preprocessing import jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 
import iblphotometry.kcenia as kcenia
import ibldsp.utils
from pathlib import Path
from brainbox.io.one import SessionLoader 
import scipy.signal
import ibllib.plots
from one.api import ONE #always after the imports 
one = ONE(cache_dir="/mnt/h0/kb/data/one") 

#%% 
dtype = {'nph_file': int, 'nph_bnc': int, 'region': int}
df1 = pd.read_excel('/home/ibladmin/Downloads/Mice training tables (1).xlsx' , 'todelete',dtype=dtype)
# df1 = pd.read_excel('/home/ibladmin/Downloads/Mice training tables .xlsx' , 'todelete',dtype=dtype) 
# df1 = pd.read_excel('/home/ibladmin/Downloads/Mice training tables (3).xlsx' , 'todelete',dtype=dtype) 
# df1 = pd.read_excel('/home/ibladmin/Downloads/Mice training tables (4).xlsx' , 'todelete',dtype=dtype)

# Function to extract date from the bncfile column
def extract_date_from_bncfile(path):
    # Split the path by '/' and get the part where the date is located
    parts = path.split('/')
    for part in parts:
        # Check if the part matches the date format YYYY-MM-DD
        if len(part) == 10 and part[4] == '-' and part[7] == '-':
            return part
    return None

# Apply the function to the bncfile column and create the 'date' column
df1['date'] = df1['bncfile'].apply(extract_date_from_bncfile)

# List of columns to drop
columns_to_drop = ["Unnamed: 3", "Unnamed: 4", "Unnamed: 5", "Unnamed: 6", "Unnamed: 7"]
# columns_to_drop = ["Unnamed: 7", "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11"] 

# Drop the columns
df1 = df1.drop(columns=columns_to_drop) 

# Create a mapping dictionary
mapping = {
    "S5": "ZFM-04392",
    "D5": "ZFM-04019",
    "D6": "ZFM-04022",
    "D4": "ZFM-04026",
    "N1": "ZFM-04533",
    "N2": "ZFM-04534",
    "D1": "ZFM-03447",
    "D2": "ZFM-03448"
}

# Create the new 'Subject' column by mapping the 'Mouse' column using the dictionary
df1['Subject'] = df1['Mouse'].map(mapping) 
df1 = df1.rename(columns={"Mouse": "subject"})
df1 = df1.rename(columns={"Subject": "mouse"})
df1 = df1.rename(columns={"Patch cord": "region"})

def get_regions(rec): 
    regions = [f"Region{rec.region}G"] 
    return regions 
    
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
EXCLUDES = [15, 19, 20, 27, 34, 35, 36, 37, 41, 42, 43, 44, 45, 56] #5th 
#34 does not start at 0... 
IMIN = 56 

EXCLUDES = [5,10,14,15,18,27,29,33,56,61,68] #6th #07July2022 29July2022 
IMIN = 61

EXCLUDES = [2, 5, 6, 7, 10, 11, 12, 18, 19, 35, 38, 54, 64, 85, 86, 88, 89, 95, 100, 101, 105, 106, 110, 111, 120, 151, 153, 154, 155, 156, 179, 202, 203, 204, 206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 218, 219, 220, 222, 223, 224]

IMIN = 2 #to start from here when rerunning; from scratch: write 0 

#%%
EXCLUDES=[12, 20, 21, 41, 42, 61, 63, 64, 67, 73, 77, 78, 83, 84, 119, 127, 153, 
    9,13,17,28,37,39,40,43,44,49,53,54,55,56,58,69,71,72,99,105,110,112,115,120,139] #7th 28082024
IMIN = 153 #to start from here when rerunning; from scratch: write 0 


#%%
for i,rec in df1.iterrows(): 
    if i < IMIN:
        continue
    if i in EXCLUDES:
        continue
    #get data info
    regions = get_regions(rec) 
    region = regions[0]
    mouse = rec.mouse
    date = rec.date

    #get behav
    eid, df_trials = get_eid(rec.mouse,rec.date)

    tbpod = df_trials['stimOnTrigger_times'].values #bpod TTL times 

    try: 
        nph_path = '/mnt/h0/kb/data/one/mainenlab/Subjects/'+rec.mouse+'/'+rec.date+'/001/alf/'+region+'/raw_photometry.pqt' 
        df_nph = pd.read_parquet(nph_path)
    except: 
        nph_path = '/mnt/h0/kb/data/one/mainenlab/Subjects/'+rec.mouse+'/'+rec.date+'/002/alf/'+region+'/raw_photometry.pqt'
        df_nph = pd.read_parquet(nph_path)

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

    event_feedback = np.array(df_trials[EVENT]) #pick the feedback timestamps 

    feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

    psth_idx += feedback_idx

    try: 
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

    except: 
        print("#####################################CLIPPED PSTH#################################")
        # Clip the indices to be within the valid range, preserving the original shape
        psth_idx_clipped = np.clip(psth_idx, 0, len(df_nph.calcium_photobleach.values) - 1) 
        psth_idx = psth_idx_clipped 
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

# %%

























































































































































































# %% ##################################################################################################################
# %% ##################################################################################################################
# %% ##################################################################################################################
##################################################################################################################
##################################################################################################################

"""
# Load data for 2 examples 
"""
filename_1 = '/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-04022/2022-11-29/001/alf/Region3G/raw_photometry.pqt'
nph1 = pd.read_parquet(filename_1)
behav1 = pd.read_parquet('/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-04022/2022-11-29/001/alf/_ibl_trials.table.pqt')
nph2 = pd.read_parquet('/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-05236/2023-07-03/001/alf/Region4G/raw_photometry.pqt')
behav2 = pd.read_parquet('/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-05236/2023-07-03/001/alf/_ibl_trials.table.pqt')

mouse = filename_1[39:48]
date = filename_1[49:59] 
region = filename_1[68:76]
eid, test = get_eid(mouse, date)

# List of dataframes
nph_list = [nph1, nph2]
behav_list = [behav1, behav2]

# Loop through each nph and behav dataframe
PERIEVENT_WINDOW = [-1, 2]
EVENT = "feedback_times" 

selected_data_dict = {}
selected_data_dict_2 = {} 

"""cut the nph data around behav"""
for nph_index, nph in enumerate(nph_list, start=1):
    behav = behav_list[nph_index - 1]  # Assuming corresponding behav matches nph by index
    nph_sync_start = behav["intervals_0"].iloc[0] - 30  # Start time, 30 seconds before the first nph_sync value
    nph_sync_end = behav["intervals_1"].iloc[-1] + 30   # End time, 30 seconds after the last nph_sync value
    selected_data = nph[
        (nph['times'] >= nph_sync_start) &
        (nph['times'] <= nph_sync_end)]
    selected_data_dict[f"nph{nph_index}"] = selected_data.reset_index(drop=True) 

metadata = [
    {"mouse": mouse, "session": date, "nm": "DA", "region": region},
    {"mouse": "ZFM-05236", "session": "2023-07-03", "nm": "5HT", "region": "Region4G"}
]hallo mein name ist georg rasiser
    # Process the calcium signal and add to df  
    nph_j = preprocess_sliding_mad(data["raw_calcium"].values, data["times"].values, fs=fs)
    data["calcium"] = nph_j
    # Add metadata columns to the DataFrame
    index = int(key[-1]) - 1  # Extract the index from the key
    for col, val in metadata[index].items():
        data[col] = val
        data["nph_idx"] = key[3:4]
    # Update the dictionary with the processed data
    selected_data_dict[key] = data

""" 
now we have: 
    selected_data_dict["nph1"]
    and
    selected_data_dict["nph2"]
    behav1 and behav2 
"""
# %% ##################################################################################################################
# Loop through each nph and corresponding behav
for i in range(1, len(behav_list) + 1):
    df_nph = pd.DataFrame(selected_data_dict[f"nph{i}"])
    df_trials = pd.DataFrame(behav_list[i-1])

    array_timestamps_bpod = np.array(df_nph.times)  # pick the nph timestamps transformed to bpod clock 
    event_test = np.array(df_trials.intervals_0)  # pick the intervals_0 timestamps 
    idx_event = np.searchsorted(array_timestamps_bpod, event_test)  # check idx where they would be included, in a sorted way 

    # Create a column with the trial number in the nph df
    df_nph["trial_number"] = 0  # create a new column for the trial_number 
    df_nph.loc[idx_event, "trial_number"] = 1
    df_nph["trial_number"] = df_nph.trial_number.cumsum()  # sum the [i-1] to i in order to get the trial number 

    sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
    n_trials = df_trials.shape[0]

    psth_idx = np.tile(sample_window[:, np.newaxis], (1, n_trials))  # KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

    event_feedback = np.array(df_trials[hhhjjgjhghghguuyyhhxxxzz])  # pick the feedback timestamps 

    feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback)  # check idx where they would be included, in a sorted way 

    psth_idx += feedback_idx

    photometry_feedback = df_nph.calcium.values[psth_idx]

    np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_sliding_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_feedback)
    print("here: ", i, " and here: ", photometry_feedback)
    plt.plot(photometry_feedback, alpha=0.1, color='black', linewidth=0.3)
    plt.title(f'{mouse}_{date}_{region}') 
    plt.axvline(x=30)
    plt.show()







for nph_index, nph in enumerate(nph_list, start=1):
    behav = behav_list[nph_index - 1]  # Assuming corresponding behav matches nph by index
    for key, data in selected_data_dict.items():
        # Calculate the sampling frequency
        time_diffs = data["times"].diff().dropna() 
        fs = 1 / time_diffs.median() 

        photometry_feedback, idx_psth = psth(calcium=data.calcium.values, times=data.times.values, t_events=behav[EVENT].values, fs=fs, peri_event_window =PERIEVENT_WINDOW) 

# concatenated_df = pd.concat(selected_data_dict.values(), ignore_index=True) 







""" TO TEST """
for key, data in selected_data_dict.items(): 
    behav = behav_list[nph_index - 1]  # Assuming corresponding behav matches nph by index
    array_timestamps_bpod = np.array(data.times) #pick the nph timestamps transformed to bpod clock 
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

        event_feedback = np.array(df_trials[EVENT_NAME]) #pick the feedback timestamps 

        feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

        psth_idx += feedback_idx

        photometry_feedback = df_nph.calcium.values[psth_idx] 

        np.save(f'/home/kceniabougrova/Documents/results_for_OW/psthidx_{EVENT_NAME}_{mouse}_{date}_{region_number}_{eid}.npy', photometry_feedback) 

#%%

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












#%%









plt.figure(figsize=(20, 8))
plt.plot(nph.times, nph.calcium, c='teal', alpha=0.85, linewidth=0.2)
for i in behav.feedback_times: 
    plt.axvline(x=i, linewidth=0.2, color='black', alpha=0.75) 
plt.show() 


#create psth #OPTION1 
for session in sessions: 
    photometry_feedback, idx_psth = psth(
        calcium=nph.calcium.values,
        times=nph.times.values, t_events=behav[EVENT].values, fs=fs, peri_event_window =PERIEVENT_WINDOW) 

plt.figure(figsize=(15, 8))
plt.plot(photometry_feedback, color='black', linewidth=0.3, alpha=0.3) 
plt.axvline(x=30)
plt.show()

#check peak, time, mad 

ipeaks_1, maxis = parabolic_max(photometry_feedback.T)
tmax = ipeaks_1 / fs + PERIEVENT_WINDOW[0]
plt.figure(figsize=(10, 5))

plt.matshow(photometry_feedback)
plt.plot(np.arange(ipeaks_1.shape[0]), ipeaks_1, '*r')
def mad(arr, axis=None, keepdims=True):
    median = np.median(arr, axis=axis, keepdims=True)
    mad = np.median(np.abs(arr-median),axis=axis, keepdims=keepdims)
    return mad
ipeaks=[] 
mads=[]
for i in range(len(photometry_feedback)): 
    ipeaks.append(np.argmax(photometry_feedback[i])) #indices of max values along axis
    mads.append(mad(photometry_feedback[i]))

#%%
"""
KB 20240709 
adding some more behav columns 
""" 
#createtrialNumber
behav['trialNumber'] = range(1, len(behav) + 1)
idx=2 #column position 
new_col = behav['contrastLeft'].fillna(behav['contrastRight']) 
behav.insert(loc=idx, column='allContrasts', value=new_col) 
#create allUContrasts 
behav['allUContrasts'] = behav['allContrasts']
behav.loc[behav['contrastRight'].isna(), 'allUContrasts'] = behav['allContrasts'] * -1
behav.insert(loc=3, column='allUContrasts', value=behav.pop('allUContrasts'))
#create reactionTime 
reactionTime = np.array((behav["firstMovement_times"])-(behav["stimOn_times"]))
behav["reactionTime"] = reactionTime 

event_time = 30 

psth_fastest_100 = nph.calcium.values[idx_psth[:,((behav.reactionTime >= 0.25) & (behav.reactionTime < 0.3) & (behav.feedbackType == 1) & (behav.allContrasts == 1))]]
psth_fastest_25 = nph.calcium.values[idx_psth[:,((behav.reactionTime >= 0.25) & (behav.reactionTime < 0.3) & (behav.feedbackType == 1) & (behav.allContrasts == 0.25))]]
psth_fastest_12 = nph.calcium.values[idx_psth[:,((behav.reactionTime >= 0.25) & (behav.reactionTime < 0.3) & (behav.feedbackType == 1) & (behav.allContrasts == 0.125))]]
psth_fastest_6 = nph.calcium.values[idx_psth[:,((behav.reactionTime >= 0.25) & (behav.reactionTime < 0.3) & (behav.feedbackType == 1) & (behav.allContrasts == 0.0625))]]
psth_fastest_0 = nph.calcium.values[idx_psth[:,((behav.reactionTime >= 0.25) & (behav.reactionTime < 0.3) & (behav.feedbackType == 1) & (behav.allContrasts == 0))]]
# psth_fastest_100 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 1))]]
# psth_fastest_25 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.25))]]
# psth_fastest_12 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.125))]]
# psth_fastest_6 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.0625))]]
# psth_fastest_0 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0))]]


# %%
width_ratios = [1, 1]
FONTSIZE_1 = 30
FONTSIZE_2 = 25
FONTSIZE_3 = 15 

data = nph.calcium.values[idx_psth]
event_time = 30 

from numpy import nanmean
average_values = nanmean(data,axis=1)

plt.plot(average_values, color='black')
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
plt.grid(False) 


psth_error = nph.calcium.values[idx_psth[:,(behav.feedbackType == -1)]]
psth_good = nph.calcium.values[idx_psth[:,(behav.feedbackType == 1)]]

"""#########################################################################################################################################"""
""" WORKED 
Plot heatmap for correct and incorrect 
"""
psth_good_avg = psth_good.mean(axis=1) 
sem_A = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1]) 

ax = sns.heatmap(psth_good.T, cbar=True)
ax.invert_yaxis()
plt.axvline(x=30, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed") 
plt.title("Heatmap for correct trials")
plt.show()
 
ax = sns.heatmap(psth_error.T, cbar=True)
ax.invert_yaxis()
plt.axvline(x=30, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed") 
plt.title("Heatmap for incorrect trials")
plt.show() 


"""#########################################################################################################################################"""
"""##### scatter of max and min peaks divided by correct and incorrect ##########################################################"""
""" WORKED
Figure X1. where is the max peak around the event? 
"""
ipeaks_1, maxis = parabolic_max(psth_good.T)
tmax = ipeaks_1 / fs + PERIEVENT_WINDOW[0]

ipeaks_2, maxis_2 = parabolic_max(psth_error.T)
tmax_2 = ipeaks_2 / fs + PERIEVENT_WINDOW[0]

import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.scatter(tmax, maxis, color='#0f7173', alpha=0.5, label='correct') 
plt.scatter(tmax_2, maxis_2, color='#f05d5e', alpha=0.5, label='incorrect')
plt.axvline(x=0, linestyle='dashed', color='black')

plt.title("max peak values divided by correct and incorrect")
plt.legend()
plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
""" WORKED 
Figure X2. where is the min peak around the event? 
""" 
ipeaks_3, minis = parabolic_max(-(psth_good.T))
tmin = ipeaks_3 / fs + PERIEVENT_WINDOW[0]

ipeaks_4, minis_2 = parabolic_max(-(psth_error.T))
tmin_2 = ipeaks_4 / fs + PERIEVENT_WINDOW[0]

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.scatter(tmin, minis, color='#0f7173', alpha=0.5, label='correct') 
plt.scatter(tmin_2, minis_2, color='#f05d5e', alpha=0.5, label='incorrect')
plt.axvline(x=0, linestyle='dashed', color='black') 

plt.title("min peak values divided by correct and incorrect")
plt.legend()
plt.show() 
"""#################################################################################################"""

#%% 
""" WORKED 
heatmaps and lineplots under 
DOUBLE-CHECK AFTER CHATGPT 
and some editions... 
"""
psth_good = nph.calcium.values[idx_psth[:,(behav.feedbackType == 1)]]
psth_error = nph.calcium.values[idx_psth[:,(behav.feedbackType == -1)]]

# Calculate averages and SEM
psth_good_avg = psth_good.mean(axis=1)
sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
psth_error_avg = psth_error.mean(axis=1)
sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

# Create the figure and gridspec
fig = plt.figure(figsize=(5, 12))
gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 3, 1])

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
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
sns.heatmap(psth_error.T, cbar=False, ax=ax3) #, center = 0.0)
ax3.invert_yaxis()
ax3.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
ax3.set_title('Incorrect Trials')

ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
ax4.plot(psth_error_avg, color='#d62828', linewidth=3)
ax4.fill_between(range(len(psth_error_avg)), psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
ax4.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
ax4.set_ylabel('Average Value')
ax4.set_xlabel('Time')

plt.tight_layout()
plt.show()




































#%% 
""" 
WORKED 
Plot correct vs incorrect for different contrasts 
"""

EVENT_NAME = "feedback" 
CONTRAST_A = 1 
CONTRAST_B = 0.25
CONTRAST_C = 0.125
CONTRAST_D = 0.0625
CONTRAST_E = 0
CORRECT = 1
INCORRECT = -1

psth_A = nph.calcium.values[idx_psth[:,((behav.feedbackType == CORRECT) & (behav.allContrasts == CONTRAST_A))]]
psth_B = nph.calcium.values[idx_psth[:,((behav.feedbackType == CORRECT) & (behav.allContrasts == CONTRAST_B))]]
psth_C = nph.calcium.values[idx_psth[:,((behav.feedbackType == CORRECT) & (behav.allContrasts == CONTRAST_C))]]
psth_D = nph.calcium.values[idx_psth[:,((behav.feedbackType == CORRECT) & (behav.allContrasts == CONTRAST_D))]] 
psth_E = nph.calcium.values[idx_psth[:,((behav.feedbackType == CORRECT) & (behav.allContrasts == CONTRAST_E))]] 
psth_F = nph.calcium.values[idx_psth[:,((behav.feedbackType == INCORRECT) & (behav.allContrasts == CONTRAST_A))]]
psth_G = nph.calcium.values[idx_psth[:,((behav.feedbackType == INCORRECT) & (behav.allContrasts == CONTRAST_B))]]
psth_H = nph.calcium.values[idx_psth[:,((behav.feedbackType == INCORRECT) & (behav.allContrasts == CONTRAST_C))]]
psth_I = nph.calcium.values[idx_psth[:,((behav.feedbackType == INCORRECT) & (behav.allContrasts == CONTRAST_D))]] 
psth_J = nph.calcium.values[idx_psth[:,((behav.feedbackType == INCORRECT) & (behav.allContrasts == CONTRAST_E))]] 

# psth_fastest_0 = nph.calcium.values[idx_psth[:,((behav.reactionTime >= 0.25) & (behav.reactionTime < 0.3) & (behav.feedbackType == 1) & (behav.allContrasts == 0))]]

average_values_A = psth_A.mean(axis=1) 
average_values_B = psth_B.mean(axis=1) 
average_values_C = psth_C.mean(axis=1) 
average_values_D = psth_D.mean(axis=1) 
average_values_E = psth_D.mean(axis=1) 
average_values_F = psth_F.mean(axis=1)
average_values_G = psth_G.mean(axis=1)
average_values_H = psth_H.mean(axis=1)
average_values_I = psth_I.mean(axis=1)
average_values_J = psth_J.mean(axis=1)

sem_A = psth_A.std(axis=1) / np.sqrt(psth_A.shape[1])
sem_B = psth_B.std(axis=1) / np.sqrt(psth_B.shape[1])
sem_C = psth_C.std(axis=1) / np.sqrt(psth_C.shape[1]) 
sem_D = psth_D.std(axis=1) / np.sqrt(psth_D.shape[1]) 
sem_E = psth_E.std(axis=1) / np.sqrt(psth_E.shape[1]) 
sem_F = psth_F.std(axis=1) / np.sqrt(psth_F.shape[1])
sem_G = psth_G.std(axis=1) / np.sqrt(psth_G.shape[1])
sem_H = psth_H.std(axis=1) / np.sqrt(psth_H.shape[1])
sem_I = psth_I.std(axis=1) / np.sqrt(psth_I.shape[1])
sem_J = psth_J.std(axis=1) / np.sqrt(psth_J.shape[1]) 

# plt.plot(psth_fast, color='#2a9d8f', linewidth=0.5, alpha=0.2) 
# plt.plot(psth_slow, color='#fb8500', linewidth=0.5, alpha=0.2) 

#colors: 2f9c95, 40c9a2, 957fef, b79ced, 1f7a8c
plt.figure(figsize=(16, 12))
plt.plot(average_values_A, color='#40c9a2', linewidth=3, alpha=0.95, label='C 100 '+str(psth_A.shape[1])) 
plt.fill_between(range(len(average_values_A)), average_values_A - sem_A, average_values_A + sem_A, color='#40c9a2', alpha=0.1)
plt.plot(average_values_B, color='#40c9a2', linewidth=3, alpha=0.75, label='C 25 '+str(psth_B.shape[1])) 
plt.fill_between(range(len(average_values_B)), average_values_B - sem_B, average_values_B + sem_B, color='#40c9a2', alpha=0.1)
plt.plot(average_values_C, color='#40c9a2', linewidth=3, alpha=0.5, label='C 12 '+str(psth_C.shape[1]))
plt.fill_between(range(len(average_values_C)), average_values_C - sem_C, average_values_C + sem_C, color='#40c9a2', alpha=0.1)
plt.plot(average_values_D, color='#40c9a2', linewidth=3, alpha=0.3, label='C 6 '+str(psth_D.shape[1]))
plt.fill_between(range(len(average_values_D)), average_values_D - sem_D, average_values_D + sem_D, color='#40c9a2', alpha=0.1)
plt.plot(average_values_E, color='#40c9a2', linewidth=3, alpha=0.15, label='C 0 '+str(psth_E.shape[1]))
plt.fill_between(range(len(average_values_E)), average_values_E - sem_E, average_values_E + sem_E, color='#40c9a2', alpha=0.1)
plt.plot(average_values_F, color='#d90429', linewidth=3, alpha=0.95, label='I 100 '+str(psth_F.shape[1]))
plt.fill_between(range(len(average_values_F)), average_values_F - sem_F, average_values_F + sem_F, color='#d90429', alpha=0.1)
plt.plot(average_values_G, color='#d90429', linewidth=3, alpha=0.75, label='I 25 '+str(psth_G.shape[1]))
plt.fill_between(range(len(average_values_G)), average_values_G - sem_G, average_values_G + sem_G, color='#d90429', alpha=0.1)
plt.plot(average_values_H, color='#d90429', linewidth=3, alpha=0.5, label='I 12 '+str(psth_H.shape[1]))
plt.fill_between(range(len(average_values_H)), average_values_H - sem_H, average_values_H + sem_H, color='#d90429', alpha=0.1)
plt.plot(average_values_I, color='#d90429', linewidth=3, alpha=0.3, label='I 6 '+str(psth_I.shape[1]))
plt.fill_between(range(len(average_values_I)), average_values_I - sem_I, average_values_I + sem_I, color='#d90429', alpha=0.1)
plt.plot(average_values_J, color='#d90429', linewidth=3, alpha=0.15, label='I 0 '+str(psth_J.shape[1]))
plt.fill_between(range(len(average_values_J)), average_values_J - sem_J, average_values_J + sem_J, color='#d90429', alpha=0.1)


plt.suptitle("NM response at "+EVENT_NAME+" for different reaction times", fontsize=FONTSIZE_1)
plt.title("feedback outcome = 1, reactionTime", fontsize=FONTSIZE_2, pad=20)
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('Neuromodulator activity', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed", label=EVENT_NAME)
plt.xticks(fontsize=FONTSIZE_3)
plt.yticks(fontsize=FONTSIZE_3) 
plt.legend(fontsize=FONTSIZE_3, frameon=False) 

# transformed_x = [(x / 30) - 1 for x in range(len(psth_idx))]
plt.grid(False)
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_xticks([0, 15, 30, 45, 60, 75, 90])

for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)

ax.spines['bottom'].set_color("black")
ax.spines['left'].set_color("black") 
# plt.ylim(-0.005, 0.0075) 
plt.show()






#%% 

EVENT_NAME = "feedback"
REACTIONTIME_1 = 0.3 
psth_A = nph.calcium.values[idx_psth[:,((behav.reactionTime <0.5) & (behav.feedbackType == 1) & (behav.allContrasts == 1))]]
psth_B = nph.calcium.values[idx_psth[:,((behav.reactionTime >1.2) & (behav.feedbackType == 1) & (behav.allContrasts == 1))]]
psth_C = nph.calcium.values[idx_psth[:,((behav.reactionTime <1) & (behav.feedbackType == 1) & (behav.allContrasts == 0))]]
psth_D = nph.calcium.values[idx_psth[:,((behav.reactionTime >1.2) & (behav.feedbackType == 1) & (behav.allContrasts == 0))]]
# psth_fastest_0 = nph.calcium.values[idx_psth[:,((behav.reactionTime >= 0.25) & (behav.reactionTime < 0.3) & (behav.feedbackType == 1) & (behav.allContrasts == 0))]]

average_values_fast0 = psth_A.mean(axis=1) 
average_values_fast = psth_B.mean(axis=1) 
average_values_slow = psth_C.mean(axis=1) 
average_values_medium = psth_D.mean(axis=1) 

sem_fast0 = psth_A.std(axis=1) / np.sqrt(psth_A.shape[1])
sem_fast = psth_B.std(axis=1) / np.sqrt(psth_B.shape[1])
sem_slow = psth_C.std(axis=1) / np.sqrt(psth_C.shape[1]) 
sem_medium = psth_D.std(axis=1) / np.sqrt(psth_D.shape[1])

# plt.plot(psth_fast, color='#2a9d8f', linewidth=0.5, alpha=0.2) 
# plt.plot(psth_slow, color='#fb8500', linewidth=0.5, alpha=0.2) 
plt.figure(figsize=(16, 12))
plt.plot(average_values_fast0, color='#2f9c95', linewidth=3, alpha=0.95, label='<0.3 RT C=1 '+str(psth_A.shape[1])) 
plt.fill_between(range(len(average_values_fast0)), average_values_fast0 - sem_fast0, average_values_fast0 + sem_fast0, color='#2f9c95', alpha=0.15)
plt.plot(average_values_fast, color='#40c9a2', linewidth=3, alpha=0.8, label='>1.2 RT C=1 '+str(psth_B.shape[1])) 
plt.fill_between(range(len(average_values_fast)), average_values_fast - sem_fast, average_values_fast + sem_fast, color='#40c9a2', alpha=0.15)
plt.plot(average_values_slow, color='#957fef', linewidth=3, alpha=0.7, label='<0.3 RT C=0 '+str(psth_C.shape[1]))
plt.fill_between(range(len(average_values_slow)), average_values_slow - sem_slow, average_values_slow + sem_slow, color='#957fef', alpha=0.15)
plt.plot(average_values_medium, color='#b79ced', linewidth=3, alpha=0.6, label='>1.2 RT C=0 '+str(psth_D.shape[1]))
plt.fill_between(range(len(average_values_medium)), average_values_medium - sem_medium, average_values_medium + sem_medium, color='#b79ced', alpha=0.15)
plt.suptitle("NM response at "+EVENT_NAME+" for different reaction times", fontsize=FONTSIZE_1)
plt.title("feedback outcome = 1, reactionTime", fontsize=FONTSIZE_2, pad=20)
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('Neuromodulator activity', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed", label=EVENT_NAME)
plt.xticks(fontsize=FONTSIZE_3)
plt.yticks(fontsize=FONTSIZE_3) 
plt.legend(fontsize=FONTSIZE_3, frameon=False) 

# transformed_x = [(x / 30) - 1 for x in range(len(psth_idx))]
plt.grid(False)
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_xticks([0, 15, 30, 45, 60, 75, 90])

for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)

ax.spines['bottom'].set_color("black")
ax.spines['left'].set_color("black") 
# plt.ylim(-0.005, 0.0075) 
plt.show()
# %%
