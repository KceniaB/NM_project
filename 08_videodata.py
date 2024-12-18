"""
22November2024
KB
videos processed with LP by MW 

  mouse   |    date    |      eid
ZFM-03059	2021-08-22	b105bea6-b3d4-46e2-af41-4e7e53277f27 
ZFM-03059	2021-08-24	3bf6c7e2-3eb1-43be-b409-4fed80e46dde
ZFM-04019	2022-01-28	a60b059d-6553-4a46-be03-eec202f4097d

""" 

#%%
#imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import glob
import os 
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
import scipy.signal
import ibllib.plots
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

# %%
""" 
video
V I D E O   D A T A

""" 
eid = '3bf6c7e2-3eb1-43be-b409-4fed80e46dde' 
ref = one.eid2ref(eid)
mouse = ref.subject
date = str(ref.date) 
a = one.load_object(eid, 'trials')
df_trials = a.to_df() 
idx = 2
new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
# create allSContrasts 
df_trials['allSContrasts'] = df_trials['allContrasts']
df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))
# create reactionTime
df_trials["stimOnScreenT"] = df_trials["feedback_times"] - df_trials["stimOn_times"]

tbpod = df_trials["feedback_times"].values


# Define the base path, mouse, and date
base_path = f'/mnt/h0/kb/data/one/mainenlab/Subjects/{mouse}/{date}/'
# Use glob to find the file path with the unknown intermediate directory
file_pattern = os.path.join(base_path, '*/raw_photometry.pqt')
file_paths = glob.glob(file_pattern)

# Check if a file was found and load it
if file_paths:
    df_nph = pd.read_parquet(file_paths[0])  # Take the first match
    print(f"Loaded file: {file_paths[0]}")
else:
    print("No matching file found.")

nph_path = '/mnt/h0/kb/data/external_drive/HCW_S2_23082021/HCW_S3_24Aug2021/PhotometryData_M1_M4_HCW_S3_24Aug2021.csv'
df_nph = pd.read_csv(nph_path)
bpod_path = '/mnt/h0/kb/data/external_drive/HCW_S2_23082021/HCW_S3_24Aug2021/DI0_M1_M4_HCW_S3_24Aug2021.csv' 
df_bpod = pd.read_csv(bpod_path) 

regions = 'Region4G'


def get_ttl(df_DI0): 
    if 'Value.Value' in df_DI0.columns: #for the new ones
        df_DI0 = df_DI0.rename(columns={"Value.Seconds": "Seconds", "Value.Value": "Value"})
    elif 'Timestamp' in df_DI0.columns: 
        df_DI0["Timestamp"] = df_DI0["Timestamp"] #for the old ones #KB added 20082024
    else:
        df_DI0["Timestamp"] = df_DI0["Seconds"] #for the old ones
    #use Timestamp from this part on, for any of the files
    raw_phdata_DI0_true = df_DI0[df_DI0.Value==True]
    df_raw_phdata_DI0_T_timestamp = pd.DataFrame(raw_phdata_DI0_true, columns=["Timestamp"])
    # raw_phdata_DI0_true = pd.DataFrame(df_DI0.Timestamp[df_DI0.Value==True], columns=['Timestamp'])
    df_raw_phdata_DI0_T_timestamp = df_raw_phdata_DI0_T_timestamp.reset_index(drop=True) 
    tph = df_raw_phdata_DI0_T_timestamp.values[:, 0] 
    return tph 

tph = get_ttl(df_bpod)
iup = tph 
fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True) #interpolation 
if len(tph)/len(tbpod) < .9: 
    print("mismatch in sync, will try to add ITI duration to the sync")
    tbpod = np.sort(np.r_[
        df_trials['intervals_0'].values,
        df_trials['intervals_1'].values - 1,  # here is the trick
        df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
    )
    fcn_nph_to_bpod_times, drift_ppm = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True)
    if len(tph)/len(tbpod) > .9:
        print("still mismatch, maybe this is an old session")
        tbpod = np.sort(np.r_[df_trials['stimOnTrigger_times'].values])
        fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, linear=True, return_indices=True) 
        assert len(iph)/len(tbpod) > .9
        print("recovered from sync mismatch, continuing #2")
assert abs(drift_ppm) < 100, "drift is more than 100 ppm"

df_nph["bpod_frame_times"] = fcn_nph_to_bpod_times(df_nph["Timestamp"]) 

fcn_nph_to_bpod_times(df_nph["Timestamp"])

df_nph["Timestamp"] 


# Assuming tph contains the timestamps in seconds
tbpod_start = tbpod[0] - 30  # Start time, 100 seconds before the first tph value
tbpod_end = tbpod[-1] + 30   # End time, 100 seconds after the last tph value

# Select data within the specified time range
selected_data = df_nph[
    (df_nph['bpod_frame_times'] >= tbpod_start) &
    (df_nph['bpod_frame_times'] <= tbpod_end)
]

# Now, selected_data contains the rows of df_ph within the desired time range 
selected_data 

df_nph = selected_data


#===========================================================================
#      4. FUNCTIONS TO LOAD DATA AND ADD SOME VARIABLES (BEHAVIOR)
#===========================================================================
df_ph = df_nph 
df_ph = df_ph.reset_index(drop=True)
df_ph = kcenia.LedState_or_Flags(df_ph)

""" 4.1.2 Check for LedState/previous Flags bugs """ 
""" 4.1.2.1 Length """
# Verify the length of the data of the 2 different LEDs
df_470, df_415 = kcenia.verify_length(df_ph)
""" 4.1.2.2 Verify if there are repeated flags """ 
kcenia.verify_repetitions(df_ph["LedState"])
""" 4.1.3 Remove "weird" data (flag swap, huge signal) """ 
df_ph_1 = df_ph

# Remove rows with LedState 1 at both ends if present
if df_ph_1['LedState'].iloc[0] == 1 and df_ph_1['LedState'].iloc[-1] == 1:
    df_ph_1 = df_ph_1.iloc[1:]

# Remove rows with LedState 2 at both ends if present
if df_ph_1['LedState'].iloc[0] == 2 and df_ph_1['LedState'].iloc[-1] == 2:
    df_ph_1 = df_ph_1.iloc[:-2]

# Filter data for LedState 2 (470nm)
df_470 = df_ph_1[df_ph_1['LedState'] == 2]

# Filter data for LedState 1 (415nm)
df_415 = df_ph_1[df_ph_1['LedState'] == 1]

# Check if the lengths of df_470 and df_415 are equal
assert len(df_470) == len(df_415), "Sync arrays are of different lengths"

# Plot the data
plt.rcParams["figure.figsize"] = (8, 5)
plt.plot(df_470[regions], c='#279F95', linewidth=0.5)
plt.plot(df_415[regions], c='#803896', linewidth=0.5)
# session_info = one.eid2ref(sl.eid)
plt.title(f'Cropped signal {ref.subject} {str(ref.date)}')
plt.legend(["GCaMP", "isosbestic"], frameon=False)
sns.despine(left=False, bottom=False)
plt.show(block=False)
plt.close() 
# Print counts
print("470 =", df_470['LedState'].count(), " 415 =", df_415['LedState'].count())

df_ph = df_ph_1.reset_index(drop=True)  
df_470 = df_ph[df_ph.LedState==2] 
df_470 = df_470.reset_index(drop=True)
df_415 = df_ph[df_ph.LedState==1] 
df_415 = df_415.reset_index(drop=True) 
#================================================
""" 4.1.4 FRAME RATE """ 
acq_FR = kcenia.find_FR(df_470["Timestamp"]) 

raw_reference = df_415[regions] #isosbestic 
raw_signal = df_470[regions] #GCaMP signal 
raw_timestamps_bpod = df_470["bpod_frame_times"]
raw_timestamps_nph_470 = df_470["Timestamp"]
raw_timestamps_nph_415 = df_415["Timestamp"]
raw_TTL_bpod = tbpod
raw_TTL_nph = tph

my_array = np.c_[raw_timestamps_bpod, raw_reference, raw_signal]

df = pd.DataFrame(my_array, columns=['times', 'raw_isosbestic', 'raw_calcium'])

df_nph = df



time_diffs = df_nph["times"].diff().dropna() 
fs = 1 / time_diffs.median()
test=df_nph
test['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
test['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
test['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
test['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
test['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
test['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)
df_nph=test 








array_timestamps_bpod = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(array_timestamps_bpod, event_test) #check idx where they would be included, in a sorted way 
# print(idx_event) 

""" create a column with the trial number in the nph df """
df_nph["trial_number"] = 0 #create a new column for the trial_number 
df_nph.loc[idx_event,"trial_number"]=1
df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
SAMPLING_RATE = round(1 / np.mean(np.diff(array_timestamps_bpod)), 0)  # Round to 2 decimal places
EVENT = "feedback_times"
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


# %%
video_data = pd.read_parquet("/home/ibladmin/Downloads/13319506-3b45-4e94-85c6-b1080dc7b10a__ibl_leftCamera.lightningPose.pqt")
video_times = np.load("/home/ibladmin/Downloads/_ibl_leftCamera.times.1daf7fae-9b0b-4cb6-8985-bba05e1d0e66.npy")
video_data["times"] = video_times

# for the new 2 videos 12Aug2024
# test = one.load_object(eid[0], 'leftCamera', attribute=['lightningPose', 'times'])
test = one.load_object(eid, 'leftCamera', attribute=['lightningPose', 'times'], query_type='remote')
video_data = pd.DataFrame(test['lightningPose']) 
video_data["times"] = test.times







ref = one.eid2ref(eid)
ref




# %%
""" plot with the correlation matrix to all column names """
video_column_names = video_data.columns
for name in video_column_names: 
    window_size = 10
    video_variable = name

    # Ensure times are sorted and create rolling means
    df_nph_smoothed = df_nph['calcium_jove2019'].rolling(window=window_size).mean()
    video_data_smoothed = video_data[video_variable].rolling(window=window_size).mean()

    # Drop NaN values created by rolling mean
    combined_data = pd.DataFrame({
        'calcium_jove2019': df_nph_smoothed,
        video_variable: video_data_smoothed
    }).dropna()

    # Calculate correlation
    correlation_matrix = combined_data.corr()

    # Plotting the smoothed data
    fig, ax1 = plt.figure(figsize=(12, 3)), plt.gca()

    # First plot
    ax1.plot(df_nph.times, df_nph_smoothed, linewidth=1, color="teal", alpha=0.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Calcium (jove 2019)', color='teal')
    ax1.tick_params(axis='y', labelcolor='teal')
    # ax1.set_ylim(-0.0015, 0.003)

    # Creating the second y-axis
    ax2 = ax1.twinx()
    ax2.plot(video_data.times, video_data_smoothed, linewidth=1, color='orange', alpha=0.5)
    ax2.set_ylabel(video_variable, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    # ax2.set_ylim(170, 190)

    plt.xlim(-5, 81)
    plt.title('Calcium and Nose Tip X over Time')
    plt.show()

    # Plotting the correlation matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

#%%
video_data["v0_diameter"] = video_data["pupil_top_r_y"] - video_data["pupil_bottom_r_y"] 

fig, ax1 = plt.figure(figsize=(12, 3)), plt.gca()
window_size = 1000
video_data_smoothed = video_data["v0_diameter"].rolling(window=window_size).mean()
ax1.plot(video_data.times, video_data_smoothed, linewidth=0.5, color='darkblue', alpha=0.5) 
nph_smoothed = df_nph["calcium_jove2019"].rolling(window=100).mean()
ax2 = ax1.twinx()
ax2.plot(df_nph.times, nph_smoothed, linewidth=0.5, color='teal', alpha=0.5) 
ax2.set_ylim(-0.01, 0.01)
plt.xlim(0,6000)
plt.show()
# %%
""" 3 plots for parts of the session - pupil, serotonin, correct incorrect """
video_data["v0_diameter"] = video_data["pupil_top_r_y"] - video_data["pupil_bottom_r_y"] 

fig, ax1 = plt.figure(figsize=(12, 3)), plt.gca()
window_size = 150
video_data_smoothed = video_data["v0_diameter"].rolling(window=window_size).mean()
ax1.plot(video_data.times, video_data_smoothed, linewidth=2, color='brown') 
nph_smoothed = df_nph["calcium_jove2019"].rolling(window=150).mean()
ax2 = ax1.twinx()
ax2.plot(df_nph.times, nph_smoothed, linewidth=2, color='teal') 
for xc, xv in zip(df_trials.stimOn_times, df_trials.feedbackType):
    if xv == 1: 
        ax2.axvline(x=xc, color='blue', linewidth=0.3)
    elif xv == -1: 
        ax2.axvline(x=xc, color='red', linewidth=0.3)
ax2.set_ylim(-0.002, 0.002)
plt.xlim(0,500)
plt.show()




fig, ax1 = plt.figure(figsize=(12, 3)), plt.gca()
window_size = 150
video_data_smoothed = video_data["v0_diameter"].rolling(window=window_size).mean()
ax1.plot(video_data.times, video_data_smoothed, linewidth=2, color='brown') 
nph_smoothed = df_nph["calcium_jove2019"].rolling(window=150).mean()
ax2 = ax1.twinx()
ax2.plot(df_nph.times, nph_smoothed, linewidth=2, color='teal') 
# Correctly zipping `feedback_times` and `feedbackType`
for xc, xv in zip(df_trials.stimOn_times, df_trials.feedbackType):
    if xv == 1: 
        ax2.axvline(x=xc, color='blue', linewidth=0.3)
    elif xv == -1: 
        ax2.axvline(x=xc, color='red', linewidth=0.3)
ax2.set_ylim(-0.002, 0.002)
plt.xlim(4950,5300)
plt.show()



window_size = 150
video_data_smoothed = video_data["v0_diameter"].rolling(window=window_size).mean()
ax1.plot(video_data.times, video_data_smoothed, linewidth=2, color='brown') 

nph_smoothed = df_nph["calcium_jove2019"].rolling(window=150).mean()
ax2 = ax1.twinx()
ax2.plot(df_nph.times, nph_smoothed, linewidth=2, color='teal') 
ax2.set_ylim(-0.002, 0.002)

# Correctly zipping `feedback_times` and `feedbackType`
for xc, xv in zip(df_trials.stimOn_times, df_trials.feedbackType):
    if xv == 1: 
        ax2.axvline(x=xc, color='blue', linewidth=0.3)
    elif xv == -1: 
        ax2.axvline(x=xc, color='red', linewidth=0.3)

plt.xlim(5500, 6000)
plt.show()

#%%
""" WHEEL MOVEMENT - shadow areas are wheel movement """ 
wheel = one.load_object(eid, 'wheel', collection='alf')
try:
    # Warning: Some older sessions may not have a wheelMoves dataset
    wheel_moves = one.load_object(eid, 'wheelMoves', collection='alf')
except AssertionError:
    wheel_moves = extract_wheel_moves(wheel.timestamps, wheel.position) 

fig, ax1 = plt.subplots(figsize=(12, 3))
window_size = 100
video_data_smoothed = video_data["v0_diameter"].rolling(window=window_size).mean()
ax1.plot(video_data.times, video_data_smoothed, linewidth=2, color='brown', label='pupil') 
plt.legend()

nph_smoothed = df_nph["calcium_jove2019"].rolling(window=150).mean()
ax2 = ax1.twinx()
ax2.plot(df_nph.times, nph_smoothed, linewidth=2, color='teal', label='5-HT')

# Add vertical lines for feedback times
for xc, xv in zip(df_trials.feedback_times, df_trials.feedbackType):
    if xv == 1: 
        ax2.axvline(x=xc, color='blue', linewidth=0.3)
    elif xv == -1: 
        ax2.axvline(x=xc, color='red', linewidth=0.3)

# Add shaded areas for wheel movements
for interval in wheel_moves['intervals']:
    ax1.axvspan(interval[0], interval[1], color='gray', alpha=0.5)

ax2.set_ylim(-0.002, 0.002)
plt.legend()
plt.xlim(5000, 5900)
plt.show()

# %%
""" creating new video vars """ 
"""
['nose_tip_x', 'nose_tip_y', 'nose_tip_likelihood', 'pupil_top_r_x',
       'pupil_top_r_y', 'pupil_top_r_likelihood', 'pupil_top_r_zscore',
       'pupil_right_r_x', 'pupil_right_r_y', 'pupil_right_r_likelihood',
       'pupil_right_r_zscore', 'pupil_bottom_r_x', 'pupil_bottom_r_y',
       'pupil_bottom_r_likelihood', 'pupil_bottom_r_zscore', 'pupil_left_r_x',
       'pupil_left_r_y', 'pupil_left_r_likelihood', 'pupil_left_r_zscore',
       'paw_l_x', 'paw_l_y', 'paw_l_likelihood', 'paw_r_x', 'paw_r_y',
       'paw_r_likelihood', 'paw_l_zscore', 'paw_r_zscore', 'tube_top_x',
       'tube_top_y', 'tube_top_likelihood', 'tube_bottom_x', 'tube_bottom_y',
       'tube_bottom_likelihood', 'tongue_end_l_x', 'tongue_end_l_y',
       'tongue_end_l_likelihood', 'tongue_end_r_x', 'tongue_end_r_y',
       'tongue_end_r_likelihood', 'times', 'v0_diameter']
""" 

""" 1. try to recalculate the diameter and correlate it with 5-HT """ 
video_data["v1_diameter"] = ((video_data.pupil_top_r_y - video_data.pupil_bottom_r_y) + (video_data.pupil_left_r_y - video_data.pupil_right_r_y))/2

""" 2. Nose Tip """ 
video_data["nose_s"] = (video_data.nose_tip_x + video_data.nose_tip_y) 
video_data["nose_m"] = (video_data.nose_tip_x * video_data.nose_tip_y) 

""" 3. Paws """
video_data["paw_l_s"] = (video_data.paw_l_x + video_data.paw_l_y) 
video_data["paw_l_m"] = (video_data.paw_l_x * video_data.paw_l_y) 
video_data["paw_r_s"] = (video_data.paw_r_x + video_data.paw_r_y) 
video_data["paw_r_m"] = (video_data.paw_r_x * video_data.paw_r_y) 
video_data["paw_lr_s"] = ((video_data.paw_l_x + video_data.paw_l_y) + (video_data.paw_r_x + video_data.paw_r_y)) 
video_data["paw_lr_m"] = ((video_data.paw_l_x * video_data.paw_l_y) + (video_data.paw_r_x * video_data.paw_r_y)) 
video_data["paw_lr_y_s"] = (video_data.paw_l_y + video_data.paw_r_y) 
video_data["paw_lr_y_m"] = (video_data.paw_l_y * video_data.paw_r_y)


############################################################################################################
#%% 
""" GLM, Random Forest and GradientBoostingRegressor on the video&photometry data """ 

import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Assuming df_nph and video_data are pre-loaded pandas DataFrames

# Merge dataframes on a common time axis
combined_data = pd.DataFrame({
    'times': df_nph.times,  # assuming df_nph.times exists and is aligned
    'calcium_jove2019': df_nph['calcium_jove2019']
})

# Merge with video_data on time
combined_data = pd.merge_asof(combined_data.sort_values('times'), video_data.sort_values('times'), on='times')

# Select relevant columns for regression analysis
features = video_data.columns.difference(['times'])
target = 'calcium_jove2019'

# Drop rows with missing values
combined_data = combined_data.dropna(subset=[target] + list(features))

# Normalize or standardize features
scaler = StandardScaler()
combined_data[features] = scaler.fit_transform(combined_data[features])

#%%
# Add a constant term for the intercept
X = sm.add_constant(combined_data[features])
y = combined_data[target]

# Fit GLM model
model = sm.GLM(y, X, family=sm.families.Gaussian()).fit()

# Print summary of the model
print(model.summary())

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize correlations
corr_matrix = combined_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)

#%%
from sklearn.linear_model import LassoCV

# Use LassoCV for feature selection
lasso = LassoCV(cv=5).fit(combined_data[features], combined_data[target])

# Get the features that have non-zero coefficients
selected_features = combined_data[features].columns[(lasso.coef_ != 0)]

print("Selected features:", selected_features)

#%% 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assuming combined_data is your DataFrame and target is 'calcium_jove2019'
selected_features = ['paw_l_m', 'paw_l_s', 'paw_r_m', 'paw_r_y', 'pupil_bottom_r_y', 'pupil_left_r_zscore', 'v0_diameter', 'v1_diameter']
X = combined_data[selected_features]
y = combined_data['calcium_jove2019']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

#%%
""" 2m """ 
# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=25, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest MSE: {mse_rf}')
print(f'Random Forest R^2: {r2_rf}')

# Feature importance
feature_importances_rf = pd.DataFrame({'feature': selected_features, 'importance': rf_model.feature_importances_})
print(feature_importances_rf.sort_values(by='importance', ascending=False))

#%% 
""" 2m """
# Initialize and train the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_gb = gb_model.predict(X_test)

# Evaluate the model
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f'Gradient Boosting MSE: {mse_gb}')
print(f'Gradient Boosting R^2: {r2_gb}')

# Feature importance
feature_importances_gb = pd.DataFrame({'feature': selected_features, 'importance': gb_model.feature_importances_})
print(feature_importances_gb.sort_values(by='importance', ascending=False)) 



################################################################################################################################# 

#%%
""" continue plotting pupil 5-HT paw """ 
# Plotting data with smoothed lines
fig, ax1 = plt.subplots(figsize=(12, 3))
window_size = 100

# Smoothed v1_diameter data
video_data_smoothed = video_data["v1_diameter"].rolling(window=window_size).mean()
ax1.plot(video_data.times, video_data_smoothed, linewidth=2, color='brown', label='pupil') 
# Adding legend for the first axis
ax1.legend(loc='upper left')
# Smoothed calcium_jove2019 data
nph_smoothed = df_nph["calcium_jove2019"].rolling(window=150).mean()
ax2 = ax1.twinx()
ax2.plot(df_nph.times, nph_smoothed, linewidth=2, color='teal', label='5-HT')
# Setting y-axis limits for the second axis
ax2.set_ylim(-0.002, 0.002)
# Adding legend for the second axis
ax2.legend(loc='upper right')

video_data_smoothed2 = video_data["test_2"].rolling(window=150).mean() 
ax3=ax1.twinx()
ax3.plot(video_data.times, video_data_smoothed2, linewidth=1, color='orange', label='paw') 
ax3.legend(loc='upper left') 

# Correctly zipping `feedback_times` and `feedbackType`
for xc, xv in zip(df_trials.stimOn_times, df_trials.feedbackType):
    if xv == 1: 
        ax2.axvline(x=xc, color='blue', linewidth=0.3)
    elif xv == -1: 
        ax2.axvline(x=xc, color='red', linewidth=0.3)


plt.xlim(5500, 5900)
plt.show() 

#%% 
# Step 1: Calculate the difference and store it in 'test'
video_data['test'] = video_data['paw_r_y'].diff()

# Step 2: Smooth the 'test' column and store the result in 'test1'
smoothing_window = 100
video_data['test1'] = video_data['test'].rolling(window=smoothing_window).mean()
video_data['test1'] = abs(video_data["test1"]) 

# # Step 3: Perform the comparison to generate 'test_2'
# video_data.loc[abs(video_data['test1']) > 0.15, 'test_2'] = 1
# video_data.loc[abs(video_data['test1']) < 0.15, 'test_2'] = 0

# Plotting data with smoothed lines
fig, ax1 = plt.subplots(figsize=(12, 3))
window_size = 100

# Smoothed v1_diameter data
video_data_smoothed = video_data["v1_diameter"].rolling(window=window_size).mean()
ax1.plot(video_data.times, video_data_smoothed, linewidth=2, color='brown', label='pupil') 
# Adding legend for the first axis
ax1.legend(loc='upper left')

# Smoothed calcium_jove2019 data
nph_smoothed = df_nph["calcium_jove2019"].rolling(window=150).mean()
ax2 = ax1.twinx()
ax2.plot(df_nph.times, nph_smoothed, linewidth=2, color='teal', label='5-HT')
# Setting y-axis limits for the second axis
ax2.set_ylim(-0.002, 0.002)
# Adding legend for the second axis
ax2.legend(loc='upper right')

# Smoothed paw_r_y data
video_data_smoothed2 = video_data["test1"].rolling(window=50).mean() 
ax3 = ax1.twinx()
ax3.plot(video_data.times, video_data_smoothed2, linewidth=2, color='orange', label='paw', alpha=0.5) 
ax3.legend(loc='upper left') 

# Correctly zipping `feedback_times` and `feedbackType`
for xc, xv in zip(df_trials.feedback_times, df_trials.feedbackType):
    if xv == 1: 
        ax2.axvline(x=xc, color='blue', linewidth=0.3)
    elif xv == -1: 
        ax2.axvline(x=xc, color='red', linewidth=0.3)

# # Adding shaded regions where video_data.test_2 is 0
# for i in range(len(video_data) - 1):
#     if video_data["test_2"].iloc[i] == 0:
#         ax1.axvspan(video_data.times.iloc[i], video_data.times.iloc[i + 1], color='grey', alpha=0.3)

plt.xlim(5000, 5800)
# Displaying the plot
plt.show()
# %%
""" WORKS 4 PLOTS ALONG THE SESSION """ 

# Step 1: Calculate the difference and store it in 'test'
video_data['test'] = video_data['nose_s'].diff()
# Step 2: Smooth the 'test' column and store the result in 'test1'
smoothing_window = 100
video_data['test1'] = video_data['test'].rolling(window=smoothing_window).mean()
video_data['test1'] = abs(video_data["test1"]) 
fig, ax1 = plt.subplots(figsize=(12, 3))
window_size = 100

# Smoothed v1_diameter data
video_data_smoothed = video_data["v1_diameter"].rolling(window=window_size).mean()
ax1.plot(video_data.times, video_data_smoothed, linewidth=2, color='brown', label='pupil') 
# Adding legend for the first axis
ax1.legend()

# Smoothed calcium_jove2019 data
nph_smoothed = df_nph["calcium_jove2019"].rolling(window=100).mean()
ax2 = ax1.twinx()
ax2.plot(df_nph.times, nph_smoothed, linewidth=2, color='teal', label='DA')
# Setting y-axis limits for the second axis
ax2.set_ylim(-0.003, 0.0035)
# Adding legend for the second axis
ax2.legend(loc='upper right')

# Smoothed paw_r_y data
video_data_smoothed2 = video_data["test1"].rolling(window=100).mean() 
ax3 = ax1.twinx()
ax3.plot(video_data.times, video_data_smoothed2, linewidth=2, color='orange', label='nose_s', alpha=0.8) 
ax3.legend(loc='upper left') 

# Correctly zipping `feedback_times` and `feedbackType`
for xc, xv in zip(df_trials.feedback_times, df_trials.feedbackType):
    if xv == 1: 
        ax2.axvline(x=xc, color='blue', linewidth=0.3)
    elif xv == -1: 
        ax2.axvline(x=xc, color='red', linewidth=0.3)
plt.xlim(0, 550)
plt.show()


# Step 1: Calculate the difference and store it in 'test'
video_data['test'] = video_data['nose_s'].diff()
# Step 2: Smooth the 'test' column and store the result in 'test1'
smoothing_window = 100
video_data['test1'] = video_data['test'].rolling(window=smoothing_window).mean()
video_data['test1'] = abs(video_data["test1"]) 
fig, ax1 = plt.subplots(figsize=(12, 3))
window_size = 100

# Smoothed v1_diameter data
video_data_smoothed = video_data["v1_diameter"].rolling(window=window_size).mean()
ax1.plot(video_data.times, video_data_smoothed, linewidth=2, color='brown', label='pupil') 
# Adding legend for the first axis
ax1.legend()

# Smoothed calcium_jove2019 data
nph_smoothed = df_nph["calcium_jove2019"].rolling(window=150).mean()
ax2 = ax1.twinx()
ax2.plot(df_nph.times, nph_smoothed, linewidth=2, color='teal', label='DA')
# Setting y-axis limits for the second axis
ax2.set_ylim(-0.003, 0.0035)
# Adding legend for the second axis
ax2.legend(loc='upper right')

# Smoothed paw_r_y data
video_data_smoothed2 = video_data["test1"].rolling(window=100).mean() 
ax3 = ax1.twinx()
ax3.plot(video_data.times, video_data_smoothed2, linewidth=2, color='orange', label='nose_s', alpha=0.8) 
ax3.legend(loc='upper left') 

# Correctly zipping `feedback_times` and `feedbackType`
for xc, xv in zip(df_trials.feedback_times, df_trials.feedbackType):
    if xv == 1: 
        ax2.axvline(x=xc, color='blue', linewidth=0.3)
    elif xv == -1: 
        ax2.axvline(x=xc, color='red', linewidth=0.3)
plt.xlim(1000, 1550)
plt.show()

# Step 1: Calculate the difference and store it in 'test'
video_data['test'] = video_data['nose_s'].diff()
# Step 2: Smooth the 'test' column and store the result in 'test1'
smoothing_window = 100
video_data['test1'] = video_data['test'].rolling(window=smoothing_window).mean()
video_data['test1'] = abs(video_data["test1"]) 

fig, ax1 = plt.subplots(figsize=(12, 3))
window_size = 100

# Smoothed v1_diameter data
video_data_smoothed = video_data["v1_diameter"].rolling(window=window_size).mean()
ax1.plot(video_data.times, video_data_smoothed, linewidth=2, color='brown', label='pupil') 
# Adding legend for the first axis
ax1.legend()

# Smoothed calcium_jove2019 data
nph_smoothed = df_nph["calcium_jove2019"].rolling(window=150).mean()
ax2 = ax1.twinx()
ax2.plot(df_nph.times, nph_smoothed, linewidth=2, color='teal', label='DA')
# Setting y-axis limits for the second axis
ax2.set_ylim(-0.003, 0.0035)
# Adding legend for the second axis
ax2.legend(loc='upper right')

# Smoothed paw_r_y data
video_data_smoothed2 = video_data["test1"].rolling(window=100).mean() 
ax3 = ax1.twinx()
ax3.plot(video_data.times, video_data_smoothed2, linewidth=2, color='orange', label='nose_s', alpha=0.8) 
ax3.legend(loc='upper left') 

# Correctly zipping `feedback_times` and `feedbackType`
for xc, xv in zip(df_trials.feedback_times, df_trials.feedbackType):
    if xv == 1: 
        ax2.axvline(x=xc, color='blue', linewidth=0.3)
    elif xv == -1: 
        ax2.axvline(x=xc, color='red', linewidth=0.3)
plt.xlim(2500, 3050)
plt.show()



# Step 1: Calculate the difference and store it in 'test'
video_data['test'] = video_data['nose_s'].diff()
# Step 2: Smooth the 'test' column and store the result in 'test1'
smoothing_window = 100
video_data['test1'] = video_data['test'].rolling(window=smoothing_window).mean()
video_data['test1'] = abs(video_data["test1"]) 

fig, ax1 = plt.subplots(figsize=(12, 3))
window_size = 100

# Smoothed v1_diameter data
video_data_smoothed = video_data["v1_diameter"].rolling(window=window_size).mean()
ax1.plot(video_data.times, video_data_smoothed, linewidth=2, color='brown', label='pupil') 
# Adding legend for the first axis
ax1.legend()

# Smoothed calcium_jove2019 data
nph_smoothed = df_nph["calcium_jove2019"].rolling(window=150).mean()
ax2 = ax1.twinx()
ax2.plot(df_nph.times, nph_smoothed, linewidth=2, color='teal', label='DA')
# Setting y-axis limits for the second axis
ax2.set_ylim(-0.003, 0.0035)
# Adding legend for the second axis
ax2.legend(loc='upper right')

# Smoothed paw_r_y data
video_data_smoothed2 = video_data["test1"].rolling(window=100).mean() 
ax3 = ax1.twinx()
ax3.plot(video_data.times, video_data_smoothed2, linewidth=2, color='orange', label='nose_s', alpha=0.8) 
ax3.legend(loc='upper left') 

# Correctly zipping `feedback_times` and `feedbackType`
for xc, xv in zip(df_trials.feedback_times, df_trials.feedbackType):
    if xv == 1: 
        ax2.axvline(x=xc, color='blue', linewidth=0.3)
    elif xv == -1: 
        ax2.axvline(x=xc, color='red', linewidth=0.3)
plt.xlim(4000, 4550)
plt.show() 


#%% 

""" 16August2024 """
"""
PLOT CC CE EC EE 
""" 
import pandas as pd

# Assuming df_trials is your DataFrame
# Example: df_trials = pd.DataFrame({'feedbackType': [1, 1, -1, -1, 1, -1, 1, 1]})

# Shift feedbackType column to get the previous feedback type
prev_feedback = df_trials['feedbackType'].shift(-1)

# cc: Current feedbackType is 1 and previous feedbackType is 1
df_trials_cc = df_trials[(prev_feedback == 1) & (df_trials['feedbackType'] == 1)]

# ce: Current feedbackType is -1 and previous feedbackType is 1
df_trials_ce = df_trials[(prev_feedback == 1) & (df_trials['feedbackType'] == -1)]

# ec: Current feedbackType is 1 and previous feedbackType is -1
df_trials_ec = df_trials[(prev_feedback == -1) & (df_trials['feedbackType'] == 1)]

# ee: Current feedbackType is -1 and previous feedbackType is -1
df_trials_ee = df_trials[(prev_feedback == -1) & (df_trials['feedbackType'] == -1)]

# Output the results
print("df_trials_cc:")
print(df_trials_cc)
print("\ndf_trials_ce:")
print(df_trials_ce)
print("\ndf_trials_ec:")
print(df_trials_ec)
print("\ndf_trials_ee:")
print(df_trials_ee)



selected_df_nph = df_nph
nph_times = np.array(selected_df_nph.times) #pick the nph timestamps transformed to bpod clock 
event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(nph_times, event_test) #check idx where they would be included, in a sorted way 
# print(idx_event) 

selected_df_nph["trial_number"] = 0 #create a new column for the trial_number 
selected_df_nph.loc[idx_event,"trial_number"]=1
selected_df_nph["trial_number"] = selected_df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
SAMPLING_RATE = int(1/np.mean(np.diff(nph_times))) #not a constant: print(1/np.mean(np.diff(nph_times))) #sampling rate #acq_FR
EVENT_NAME = "feedback_times" 

sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_trials.shape[0]

psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

event_feedback = np.array(df_trials[EVENT_NAME]) #pick the feedback timestamps 

feedback_idx = np.searchsorted(nph_times, event_feedback) #check idx where they would be included, in a sorted way 

psth_idx += feedback_idx

photometry_feedback = selected_df_nph.calcium_jove2019.values[psth_idx] 

photometry_feedback_avg = np.mean(photometry_feedback, axis=1)
# plt.plot(photometry_feedback_avg) 

import numpy as np
import matplotlib.pyplot as plt

def avg_sem(data):
    avg = data.mean(axis=1)
    sem = data.std(axis=1) / np.sqrt(data.shape[1])
    return avg, sem

# Assuming df_trials, prev_feedback, and photometry_feedback are already defined
psth_combined = photometry_feedback
psth_combined_400 = photometry_feedback

# Define the trial types
df_trials_cc = df_trials[(prev_feedback == 1) & (df_trials['feedbackType'] == 1)]
df_trials_ee = df_trials[(prev_feedback == -1) & (df_trials['feedbackType'] == -1)]
df_trials_ce = df_trials[(prev_feedback == 1) & (df_trials['feedbackType'] == -1)]
df_trials_ec = df_trials[(prev_feedback == -1) & (df_trials['feedbackType'] == 1)]

# PSTH for each trial type
psth_correct = psth_combined[:, ((prev_feedback == 1) & (df_trials['feedbackType'] == 1))]
psth_correct_400 = psth_combined_400[:, ((prev_feedback == -1) & (df_trials['feedbackType'] == -1))]
psth_ce = psth_combined[:, ((prev_feedback == 1) & (df_trials['feedbackType'] == -1))]
psth_ec = psth_combined[:, ((prev_feedback == -1) & (df_trials['feedbackType'] == 1))]

# Compute avg and sem for each trial type
avg_cc, sem_cc = avg_sem(psth_correct)
avg_ee, sem_ee = avg_sem(psth_correct_400)
avg_ce, sem_ce = avg_sem(psth_ce)
avg_ec, sem_ec = avg_sem(psth_ec)

# Create the figure and gridspec
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 3])

# Plot for cc trials
color = "#218380"
plt.plot(avg_cc, color=color, linewidth=2, label='cc trials')
plt.fill_between(range(len(avg_cc)), avg_cc - sem_cc, avg_cc + sem_cc, color=color, alpha=0.18)

# Plot for ee trials
color = "#aa3e98"
plt.plot(avg_ee, color=color, linewidth=2, label="ee trials")
plt.fill_between(range(len(avg_ee)), avg_ee - sem_ee, avg_ee + sem_ee, color=color, alpha=0.18)

# Plot for ce trials
color = "#f28e2b"
plt.plot(avg_ce, color=color, linewidth=2, label="ce trials")
plt.fill_between(range(len(avg_ce)), avg_ce - sem_ce, avg_ce + sem_ce, color=color, alpha=0.18)

# Plot for ec trials
color = "#76b041"
plt.plot(avg_ec, color=color, linewidth=2, label="ec trials")
plt.fill_between(range(len(avg_ec)), avg_ec - sem_ec, avg_ec + sem_ec, color=color, alpha=0.18)

# Adding a vertical line, labels, title, and legend
plt.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
plt.ylabel('Average Value')
plt.xlabel('Time')
mouse = "ZFM-06275"  # Change here
title = "psth aligned to feedback "
plt.title(title + ' mouse ' + mouse, fontsize=16)

# Adding legend outside the plots
plt.legend(fontsize=14)
fig.suptitle('Neuromodulator activity for different trial types in 1 mouse', y=1.02, fontsize=18)
plt.tight_layout()

# Show the plot
plt.show()
# %%














































































#%% 
""" 28-October-2024 """
""" LOAD EID, MOUSE, DATE, TRIALS WITH MORE FEATURES, AND PHOTOMETRY, COMBINE THEM ALL AND PLOT """
#%%
#imports 
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from ibldsp.utils import parabolic_max 
from iblphotometry.preprocessing import preprocessing_alejandro, jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 

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

# Get the list of good sessions and their info 
df_goodsessions = pd.read_csv('/home/ibladmin/Downloads/Mice_GOOD_sorted.csv') 
df_goodsessions['Date'] = pd.to_datetime(df_goodsessions['Date'], format='%m/%d/%Y')
df_gs = df_goodsessions[['Mouse', 'Date', 'NM', 'region']] 

# Edit the event! 
EVENT = 'feedback_times'

# Initialize empty containers
psth_combined = None
df_trials_combined = pd.DataFrame() 
df_nph_combined = pd.DataFrame()

EXCLUDES = []  
IMIN = 0

# Choose the NM
NM="DA" #"DA", "5HT", "NE", "ACh"
df_goodsessions = df_gs[df_gs["NM"]==NM].reset_index(drop=True)





for i in range(len(df_gs[0:6])): 
    mouse = df_gs.Mouse[i] 
    date = df_gs.Date[i]
    if isinstance(date, pd.Timestamp):
        date = date.strftime('%Y-%m-%d')
    region = df_gs.region[i]
    eid, df_trials = get_eid(mouse,date)

    """ LOAD TRIALS """
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
        df_trials["quiescenceTime"] = df_trials["stimOnTrigger_times"] - df_trials["intervals_0"] 
        df_trials["trialTime"] = df_trials["intervals_1"] - df_trials["intervals_0"]  

        try: 
            dataset_task_settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json')  
            values = dataset_task_settings.get('LEN_BLOCKS', 'Key not found') 
            # values gives the block length 
            # example for eid = 'be3208c9-43de-44dc-bdc6-ff8963464f98'
            # [90, 27, 82, 50, 30, 30, 31, 78, 64, 83, 24, 42, 74, 72, 34, 41, 52, 56, 68, 39, 45, 88, 37, 35, 29, 69, 85, 52, 37, 78, 80, 28, 68, 95, 34, 36, 42] 

            values_sum = np.cumsum(values) 

            # Initialize a new column 'probL' with NaN values
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
            # plt.title(f'behavior_{subject}_{session_date}_{eid}')
            # plt.show() 
        except: 
            pass 

        df_trials["trialNumber"] = range(1, len(df_trials) + 1) 
        return df_trials, subject, session_date

    df_trials, subject, session_date = load_trials_updated(eid) 
    mouse = subject
    date = session_date
    region = df_gs.region[i]
    eid, df_trials2 = get_eid(mouse,date)

    region = f'Region{region}G'
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

        


    time_diffs = (df_nph["times"]).diff().dropna() 
    fs = 1 / time_diffs.median() 
    
    df_nph[["subject", "date", "eid"]] = [subject, session_date, eid]    
    df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
    df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
    df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
    df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
    df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
    df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)
    df_nph['calcium_alex'] = preprocessing_alejandro(df_nph["raw_calcium"], fs=fs) 
    df_nph['isos_alex'] = preprocessing_alejandro(df_nph['raw_isosbestic'], fs=fs)




    plt.figure(figsize=(20, 6))
    plt.plot(df_nph['times'][1000:2000], df_nph['calcium_mad'][1000:2000], linewidth=1.25, alpha=0.8, color='teal') 
    plt.plot(df_nph['times'][1000:2000], df_nph['isosbestic_mad'][1000:2000], linewidth=1.25, alpha=0.8, color='purple') 
    plt.show() 

    """ SELECT THE EVENT AND WHAT INTERVAL TO PLOT IN THE PSTH """ 
    EVENT = "feedback_times" 
    time_bef = -1
    time_aft = 2
    PERIEVENT_WINDOW = [time_bef,time_aft]
    SAMPLING_RATE = int(1/np.mean(np.diff(df_nph.times))) 



    nph_times = df_nph['times'].values

    # Step 1: Identify the last row index to keep in df_trials
    last_index_to_keep = None

    for index, row in df_trials.iterrows():
        if row['intervals_1'] >= nph_times.max():  # Check if intervals_1 is >= any nph times
            last_index_to_keep = index - 1  # Store the index just before the current one
            break

    # If no row meets the condition, keep all
    if last_index_to_keep is None:
        filtered_df_trials = df_trials
    else:
        filtered_df_trials = df_trials.iloc[:last_index_to_keep + 1]

    df_trials_original = df_trials
    df_trials = filtered_df_trials

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








    # path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/' 
    # path = path_initial + f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy'    

    
    # # Load psth_idx from file
    # psth_idx = np.load(path)

    # Concatenate psth_idx arrays
    if psth_combined is None:
        psth_combined = psth_idx
    else: 
        psth_combined = np.hstack((psth_combined, psth_idx))


    # Concatenate df_trials DataFrames
    df_trials_combined = pd.concat([df_trials_combined, df_trials], axis=0)

    # Reset index of the combined DataFrame
    df_trials_combined.reset_index(drop=True, inplace=True)

    # Concatenate df_trials DataFrames
    df_nph_combined = pd.concat([df_nph_combined, df_nph], axis=0)

    # Reset index of the combined DataFrame
    df_nph_combined.reset_index(drop=True, inplace=True)

    # Print shapes to verify
    print("Shape of psth_combined:", psth_combined.shape)
    print("Shape of df_trials_combined:", df_trials_combined.shape)
    print("Shape of df_nph_combined:", df_nph_combined.shape)


    # ##################################################################################################
    # # PLOT heatmap and correct vs incorrect 
    # def plot_heatmap_psth(preprocessingtype=df_nph.calcium_mad, psth = psth_combined, trials = df_trials_combined): 
    #     psth_good = preprocessingtype.values[psth[:,(trials["feedbackType"] == 1)]]
    #     psth_error = preprocessingtype.values[psth[:,(trials["feedbackType"] == -1)]]
    #     # Calculate averages and SEM
    #     psth_good_avg = psth_good.mean(axis=1)
    #     sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
    #     psth_error_avg = psth_error.mean(axis=1)
    #     sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

    #     # Create the figure and gridspec
    #     fig = plt.figure(figsize=(10, 12))
    #     gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    #     # Plot the heatmap and line plot for correct trials
    #     ax1 = fig.add_subplot(gs[0, 0])
    #     sns.heatmap(psth_good.T, cbar=False, ax=ax1) #, center = 0.0)
    #     ax1.invert_yaxis()
    #     ax1.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    #     ax1.set_title('Correct Trials')

    #     ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    #     ax2.plot(psth_good_avg, color='#2f9c95', linewidth=3) 
    #     # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
    #     ax2.fill_between(range(len(psth_good_avg)), psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
    #     ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    #     ax2.set_ylabel('Average Value')
    #     ax2.set_xlabel('Time')

    #     # Plot the heatmap and line plot for incorrect trials
    #     ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
    #     sns.heatmap(psth_error.T, cbar=False, ax=ax3) #, center = 0.0)
    #     ax3.invert_yaxis()
    #     ax3.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    #     ax3.set_title('Incorrect Trials')

    #     ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax2)
    #     ax4.plot(psth_error_avg, color='#d62828', linewidth=3)
    #     ax4.fill_between(range(len(psth_error_avg)), psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
    #     ax4.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    #     ax4.set_ylabel('Average Value')
    #     ax4.set_xlabel('Time')

    #     fig.suptitle(f'calcium_mad_{EVENT}_{subject}_{session_date}_{region}_{eid}', y=1, fontsize=14)
    #     plt.tight_layout()
    #     # plt.savefig(f'/mnt/h0/kb/data/psth_npy/Fig02_{EVENT}_{mouse}_{date}_{region}_{eid}.png')
    #     plt.show() 

    # # plot_heatmap_psth(df_nph.calcium_mad) 
    # plot_heatmap_psth(df_nph.calcium_mad, psth_combined, df_trials_combined)

################################################################################################## 



#%%
""" just one of the sessions, depending on the index chosen from the date filtering from the general table """ 


test = df_trials_combined[df_trials_combined.date == '2023-01-19']

indices = test.index  # Assuming `test` and `df_trials_combined` have matching indices

# Step 2: Select the columns in `psth_combined` using these indices
psth_combined_test = psth_combined[:, indices]

# Verify the shape of psth_combined_test
print("Shape of psth_combined_test:", psth_combined_test.shape)

##################################################################################################
# PLOT heatmap and correct vs incorrect 
# psth_good = psth_combined_test[:,(test.feedbackType == 1)]
# psth_error = psth_combined_test[:,(test.feedbackType == -1)] 
# psth_good = df_nph.calcium_mad.values[psth_idx[:,(df_trials.feedbackType == 1)]]
# psth_error = df_nph.calcium_mad.values[psth_idx[:,(df_trials.feedbackType == -1)]]
psth_good = df_nph_combined.calcium_mad.values[psth_combined[:,(df_trials_combined.feedbackType == 1)]]
psth_error = df_nph_combined.calcium_mad.values[psth_combined[:,(df_trials_combined.feedbackType == -1)]]
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

# MIGHT BE WRONG!!!!! 
# ticks = np.linspace(0, len(psth_good_avg), 4)  # Assuming 91 points, set 4 tick marks
# tick_labels = [-1, 0, 1, 2]    # Labels corresponding to time from -1 to 2 seconds

fig.suptitle(f'calcium_mad_{EVENT}_{mouse}_{date}_{region}_{NM}_{eid}', y=1, fontsize=14)
plt.tight_layout()
plt.show()




#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

correlation = []

# Step 1: Calculate mean calcium_mad values within each interval, as before
mean_calcium_mad = []

start_time_name = 'feedback_times'
end_time_name = 'intervals_1' 
varTime = 'quiescenceTime'
for _, row in df_trials.iterrows():
    # Define the time interval for each trial
    start_time = row[start_time_name]
    end_time = row[end_time_name]
    
    # Filter df_nph to get calcium_mad values within this interval
    in_interval = df_nph[(df_nph['times'] >= start_time) & (df_nph['times'] <= end_time)]
    mean_value = in_interval['calcium_mad'].mean()  # Calculate the mean for the interval
    
    mean_calcium_mad.append(mean_value)  # Append to the list

# Step 2: Add the mean calcium values to df_trials
df_trials['mean_calcium_mad'] = mean_calcium_mad

# Step 3: Calculate the correlation for each trial
# Calculate the correlation between mean calcium and quiescence times as a single value for each trial
correlations = df_trials['mean_calcium_mad'].corr(df_trials[varTime])

# Assuming `mean_calcium_mad` and `quiescenceTimes` are in `df_trials`
plt.figure(figsize=(10, 6))

# Scatter plot of mean_calcium_mad vs. quiescenceTimes for each trial
plt.scatter(df_trials[varTime], df_trials['mean_calcium_mad'], color='blue', alpha=0.3)
plt.title(f"Mean Calcium MAD vs {start_time_name} and {end_time_name}")
plt.xlabel(f"{varTime} (s)")  # Adjust units if necessary
plt.ylabel("Mean Calcium MAD")

# Add the correlation coefficient to the plot
correlation = df_trials['mean_calcium_mad'].corr(df_trials[varTime])
plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', ha='left', va='top', transform=plt.gca().transAxes)

# Show the plot
plt.grid(True)
plt.show() 




""" CORRELATE THE MEAN PHOTOMETRY SIGNAL BY TRIAL WITHIN A SPECIFIC TIME INTERVAL TO THE OTHER COLUMNS IN DF_TRIALS """ 
import seaborn as sns
import matplotlib.pyplot as plt

mean_calcium_mad = []

start_time_name = 'firstMovement_times'
end_time_name = 'feedback_times' 
for _, row in df_trials.iterrows():
    # Define the time interval for each trial
    start_time = row[start_time_name]
    end_time = row[end_time_name]
    
    # Filter df_nph to get calcium_mad values within this interval
    in_interval = df_nph[(df_nph['times'] >= start_time) & (df_nph['times'] <= end_time)]
    mean_value = in_interval['calcium_mad'].mean()  # Calculate the mean for the interval
    
    mean_calcium_mad.append(mean_value)  # Append to the list

# Step 2: Add the mean calcium values to df_trials
df_trials['mean_calcium_mad'] = mean_calcium_mad
# Step 2: Select the relevant columns for correlation
columns_to_correlate = [
    'mean_calcium_mad',  # Include the calcium signal mean values
    'stimOnTrigger_times', 'goCueTrigger_times', 'allContrasts',
    'allSContrasts', 'stimOff_times', 'stimOffTrigger_times',
    'quiescencePeriod', 'goCue_times', 'response_times', 'choice',
    'stimOn_times', 'contrastLeft', 'contrastRight', 'feedback_times',
    'feedbackType', 'rewardVolume', 'probabilityLeft',
    'firstMovement_times', 'intervals_0', 'intervals_1', 
    'reactionTime', 'responseTime', 'quiescenceTime', 'trialTime',
    'probL', 'trialNumber'
]

# Step 3: Calculate the correlation matrix
correlation_matrix = df_trials[columns_to_correlate].corr()

# Step 4: Plot the correlation matrix as a heatmap
plt.figure(figsize=(20, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True,
            cbar_kws={"shrink": .8}, annot_kws={"size": 8})
plt.title(f"Correlation Matrix Heatmap between Calcium Signal {start_time_name} {end_time_name} and Time Columns")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

plt.show()














































#%% WORKS FOR MULTIPLE SESSIONS OF 1  NM AND SO ON 
NM="5HT" #"DA", "5HT", "NE", "ACh"
df_goodsessions = df_gs[df_gs["NM"]==NM].reset_index(drop=True)

for i in range(len(df_goodsessions)): 
    mouse = df_goodsessions.Mouse[i] 
    date = df_goodsessions.Date[i]
    if isinstance(date, pd.Timestamp):
        date = date.strftime('%Y-%m-%d')
    region = df_goodsessions.region[i]
    eid, df_trials = get_eid(mouse,date)

    """ LOAD TRIALS """
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
        df_trials["quiescenceTime"] = df_trials["stimOnTrigger_times"] - df_trials["intervals_0"] 
        df_trials["trialTime"] = df_trials["intervals_1"] - df_trials["intervals_0"]  

        try: 
            dataset_task_settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json')  
            values = dataset_task_settings.get('LEN_BLOCKS', 'Key not found') 
            # values gives the block length 
            # example for eid = 'be3208c9-43de-44dc-bdc6-ff8963464f98'
            # [90, 27, 82, 50, 30, 30, 31, 78, 64, 83, 24, 42, 74, 72, 34, 41, 52, 56, 68, 39, 45, 88, 37, 35, 29, 69, 85, 52, 37, 78, 80, 28, 68, 95, 34, 36, 42] 

            values_sum = np.cumsum(values) 

            # Initialize a new column 'probL' with NaN values
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
            # plt.title(f'behavior_{subject}_{session_date}_{eid}')
            # plt.show() 
        except: 
            pass 

        df_trials["trialNumber"] = range(1, len(df_trials) + 1) 
        return df_trials, subject, session_date

    df_trials, subject, session_date = load_trials_updated(eid) 
    mouse = subject
    date = session_date
    region = df_gs.region[i]
    eid, df_trials2 = get_eid(mouse,date)
    try: 
        region = f'Region{region}G'
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

            


        time_diffs = (df_nph["times"]).diff().dropna() 
        fs = 1 / time_diffs.median() 
        
        df_nph[["subject", "date", "eid"]] = [subject, session_date, eid]    
        df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
        df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
        df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
        df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
        df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
        df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)
        df_nph['calcium_alex'] = preprocessing_alejandro(df_nph["raw_calcium"], fs=fs) 
        df_nph['isos_alex'] = preprocessing_alejandro(df_nph['raw_isosbestic'], fs=fs)




        plt.figure(figsize=(20, 6))
        plt.plot(df_nph['times'][1000:2000], df_nph['calcium_mad'][1000:2000], linewidth=1.25, alpha=0.8, color='teal') 
        plt.plot(df_nph['times'][1000:2000], df_nph['isosbestic_mad'][1000:2000], linewidth=1.25, alpha=0.8, color='purple') 
        plt.show() 

        """ SELECT THE EVENT AND WHAT INTERVAL TO PLOT IN THE PSTH """ 
        EVENT = "feedback_times" 
        time_bef = -1
        time_aft = 2
        PERIEVENT_WINDOW = [time_bef,time_aft]
        SAMPLING_RATE = int(1/np.mean(np.diff(df_nph.times))) 



        nph_times = df_nph['times'].values

        # Step 1: Identify the last row index to keep in df_trials
        last_index_to_keep = None

        for index, row in df_trials.iterrows():
            if row['intervals_1'] >= nph_times.max():  # Check if intervals_1 is >= any nph times
                last_index_to_keep = index - 1  # Store the index just before the current one
                break

        # If no row meets the condition, keep all
        if last_index_to_keep is None:
            filtered_df_trials = df_trials
        else:
            filtered_df_trials = df_trials.iloc[:last_index_to_keep + 1]

        df_trials_original = df_trials
        df_trials = filtered_df_trials

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








        # path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/' 
        # path = path_initial + f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy'    

        
        # # Load psth_idx from file
        # psth_idx = np.load(path)

        # Concatenate psth_idx arrays
        if psth_combined is None:
            psth_combined = psth_idx
        else: 
            psth_combined = np.hstack((psth_combined, psth_idx))


        # Concatenate df_trials DataFrames
        df_trials_combined = pd.concat([df_trials_combined, df_trials], axis=0)

        # Reset index of the combined DataFrame
        df_trials_combined.reset_index(drop=True, inplace=True)

        # Concatenate df_trials DataFrames
        df_nph_combined = pd.concat([df_nph_combined, df_nph], axis=0)

        # Reset index of the combined DataFrame
        df_nph_combined.reset_index(drop=True, inplace=True)

        # Print shapes to verify
        print("Shape of psth_combined:", psth_combined.shape)
        print("Shape of df_trials_combined:", df_trials_combined.shape)
        print("Shape of df_nph_combined:", df_nph_combined.shape)


        # ##################################################################################################
        # # PLOT heatmap and correct vs incorrect 
        # def plot_heatmap_psth(preprocessingtype=df_nph.calcium_mad, psth = psth_combined, trials = df_trials_combined): 
        #     psth_good = preprocessingtype.values[psth[:,(trials["feedbackType"] == 1)]]
        #     psth_error = preprocessingtype.values[psth[:,(trials["feedbackType"] == -1)]]
        #     # Calculate averages and SEM
        #     psth_good_avg = psth_good.mean(axis=1)
        #     sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
        #     psth_error_avg = psth_error.mean(axis=1)
        #     sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

        #     # Create the figure and gridspec
        #     fig = plt.figure(figsize=(10, 12))
        #     gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

        #     # Plot the heatmap and line plot for correct trials
        #     ax1 = fig.add_subplot(gs[0, 0])
        #     sns.heatmap(psth_good.T, cbar=False, ax=ax1) #, center = 0.0)
        #     ax1.invert_yaxis()
        #     ax1.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
        #     ax1.set_title('Correct Trials')

        #     ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        #     ax2.plot(psth_good_avg, color='#2f9c95', linewidth=3) 
        #     # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
        #     ax2.fill_between(range(len(psth_good_avg)), psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
        #     ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        #     ax2.set_ylabel('Average Value')
        #     ax2.set_xlabel('Time')

        #     # Plot the heatmap and line plot for incorrect trials
        #     ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
        #     sns.heatmap(psth_error.T, cbar=False, ax=ax3) #, center = 0.0)
        #     ax3.invert_yaxis()
        #     ax3.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
        #     ax3.set_title('Incorrect Trials')

        #     ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax2)
        #     ax4.plot(psth_error_avg, color='#d62828', linewidth=3)
        #     ax4.fill_between(range(len(psth_error_avg)), psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
        #     ax4.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        #     ax4.set_ylabel('Average Value')
        #     ax4.set_xlabel('Time')

        #     fig.suptitle(f'calcium_mad_{EVENT}_{subject}_{session_date}_{region}_{eid}', y=1, fontsize=14)
        #     plt.tight_layout()
        #     # plt.savefig(f'/mnt/h0/kb/data/psth_npy/Fig02_{EVENT}_{mouse}_{date}_{region}_{eid}.png')
        #     plt.show() 

        # # plot_heatmap_psth(df_nph.calcium_mad) 
        # plot_heatmap_psth(df_nph.calcium_mad, psth_combined, df_trials_combined) 
    except: 
        pass

################################################################################################## 



#%%
""" just one of the sessions, depending on the index chosen from the date filtering from the general table """ 


test = df_trials_combined[df_trials_combined.date == '2023-01-19']

indices = test.index  # Assuming `test` and `df_trials_combined` have matching indices

# Step 2: Select the columns in `psth_combined` using these indices
psth_combined_test = psth_combined[:, indices]

# Verify the shape of psth_combined_test
print("Shape of psth_combined_test:", psth_combined_test.shape)

##################################################################################################
# PLOT heatmap and correct vs incorrect 
# psth_good = psth_combined_test[:,(test.feedbackType == 1)]
# psth_error = psth_combined_test[:,(test.feedbackType == -1)] 
# psth_good = df_nph.calcium_mad.values[psth_idx[:,(df_trials.feedbackType == 1)]]
# psth_error = df_nph.calcium_mad.values[psth_idx[:,(df_trials.feedbackType == -1)]]
psth_good = df_nph_combined.calcium_mad.values[psth_combined[:,(df_trials_combined.feedbackType == 1)]]
psth_error = df_nph_combined.calcium_mad.values[psth_combined[:,(df_trials_combined.feedbackType == -1)]]
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

# MIGHT BE WRONG!!!!! 
# ticks = np.linspace(0, len(psth_good_avg), 4)  # Assuming 91 points, set 4 tick marks
# tick_labels = [-1, 0, 1, 2]    # Labels corresponding to time from -1 to 2 seconds

fig.suptitle(f'calcium_mad_{EVENT}_{mouse}_{date}_{region}_{NM}_{eid}', y=1, fontsize=14)
plt.tight_layout()
plt.show()


















































































#%%
""" THIS ONE LAST VERSION 29October2024 """
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from ibldsp.utils import parabolic_max 
from iblphotometry.preprocessing import preprocessing_alejandro, jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 

from one.api import ONE #always after the imports 
one = ONE(cache_dir="/mnt/h0/kb/data/one") 

psth_combined = None
df_trials_combined = None
df_nph_combined = None

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

# Get the list of good sessions and their info 
df_goodsessions = pd.read_csv('/home/ibladmin/Downloads/Mice_GOOD_sorted.csv') 
df_goodsessions['Date'] = pd.to_datetime(df_goodsessions['Date'], format='%m/%d/%Y')
df_gs = df_goodsessions[['Mouse', 'Date', 'NM', 'region']] 

# Edit the event! 
EVENT = 'feedback_times'

# Initialize empty containers
psth_combined = None
df_trials_combined = pd.DataFrame() 
df_nph_combined = pd.DataFrame()





NM="ACh" #"DA", "5HT", "NE", "ACh"
df_goodsessions = df_gs[df_gs["NM"]==NM].reset_index(drop=True)

for i in range(len(df_goodsessions)): 
    mouse = df_goodsessions.Mouse[i] 
    date = df_goodsessions.Date[i]
    if isinstance(date, pd.Timestamp):
        date = date.strftime('%Y-%m-%d')
    region = df_goodsessions.region[i]
    eid, df_trials = get_eid(mouse,date)

    """ LOAD TRIALS """
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
        df_trials["quiescenceTime"] = df_trials["stimOnTrigger_times"] - df_trials["intervals_0"] 
        df_trials["trialTime"] = df_trials["intervals_1"] - df_trials["intervals_0"]  

        try: 
            dataset_task_settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json')  
            values = dataset_task_settings.get('LEN_BLOCKS', 'Key not found') 
            # values gives the block length 
            # example for eid = 'be3208c9-43de-44dc-bdc6-ff8963464f98'
            # [90, 27, 82, 50, 30, 30, 31, 78, 64, 83, 24, 42, 74, 72, 34, 41, 52, 56, 68, 39, 45, 88, 37, 35, 29, 69, 85, 52, 37, 78, 80, 28, 68, 95, 34, 36, 42] 

            values_sum = np.cumsum(values) 

            # Initialize a new column 'probL' with NaN values
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
            # plt.title(f'behavior_{subject}_{session_date}_{eid}')
            # plt.show() 
        except: 
            pass 

        df_trials["trialNumber"] = range(1, len(df_trials) + 1) 
        return df_trials, subject, session_date

    df_trials, subject, session_date = load_trials_updated(eid) 
    mouse = subject
    date = session_date
    # eid, df_trials2 = get_eid(mouse,date)

    region = f'Region{region}G'
    try: 
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


        session_start = df_trials.intervals_0.values[0] - 10  # Start time, 100 seconds before the first tph value
        session_end = df_trials.intervals_1.values[-1] + 10   # End time, 100 seconds after the last tph value

        # Select data within the specified time range
        selected_data = df_nph[
            (df_nph['times'] >= session_start) &
            (df_nph['times'] <= session_end)
        ] 
        df_nph = selected_data.reset_index(drop=True) 
            


        time_diffs = (df_nph["times"]).diff().dropna() 
        fs = 1 / time_diffs.median() 
        
        df_nph[["subject", "date", "eid"]] = [subject, session_date, eid]    
        df_nph['calcium_photobleach'] = photobleaching_lowpass(df_nph["raw_calcium"].values, fs=fs) #KB
        df_nph['isosbestic_photobleach'] = photobleaching_lowpass(df_nph["raw_isosbestic"], fs=fs)
        df_nph['calcium_jove2019'] = jove2019(df_nph["raw_calcium"], df_nph["raw_isosbestic"], fs=fs) 
        df_nph['isosbestic_jove2019'] = jove2019(df_nph["raw_isosbestic"], df_nph["raw_calcium"], fs=fs)
        df_nph['calcium_mad'] = preprocess_sliding_mad(df_nph["raw_calcium"].values, df_nph["times"].values, fs=fs)
        df_nph['isosbestic_mad'] = preprocess_sliding_mad(df_nph["raw_isosbestic"].values, df_nph["times"].values, fs=fs)
        df_nph['calcium_alex'] = preprocessing_alejandro(df_nph["raw_calcium"], fs=fs) 
        df_nph['isos_alex'] = preprocessing_alejandro(df_nph['raw_isosbestic'], fs=fs)




        plt.figure(figsize=(20, 6))
        plt.plot(df_nph['times'][1000:2000], df_nph['calcium_mad'][1000:2000], linewidth=1.25, alpha=0.8, color='teal') 
        plt.plot(df_nph['times'][1000:2000], df_nph['isosbestic_mad'][1000:2000], linewidth=1.25, alpha=0.8, color='purple') 
        plt.show() 

        """ SELECT THE EVENT AND WHAT INTERVAL TO PLOT IN THE PSTH """ 
        EVENT = "feedback_times" 
        time_bef = -1
        time_aft = 2
        PERIEVENT_WINDOW = [time_bef,time_aft]
        SAMPLING_RATE = int(1/np.mean(np.diff(df_nph.times))) 



        nph_times = df_nph['times'].values

        # Step 1: Identify the last row index to keep in df_trials
        last_index_to_keep = None

        for index, row in df_trials.iterrows():
            if row['intervals_1'] >= nph_times.max():  # Check if intervals_1 is >= any nph times
                last_index_to_keep = index - 1  # Store the index just before the current one
                break

        # If no row meets the condition, keep all
        if last_index_to_keep is None:
            filtered_df_trials = df_trials
        else:
            filtered_df_trials = df_trials.iloc[:last_index_to_keep + 1]

        df_trials_original = df_trials
        df_trials = filtered_df_trials

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








        # path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/' 
        # path = path_initial + f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy'    

        
        # # Load psth_idx from file
        # psth_idx = np.load(path)

        # Concatenate psth_idx arrays
        if psth_combined is None:
            psth_combined = psth_idx
        else: 
            psth_combined = np.hstack((psth_combined, psth_idx))


        # Concatenate df_trials DataFrames
        df_trials_combined = pd.concat([df_trials_combined, df_trials], axis=0)

        # Reset index of the combined DataFrame
        df_trials_combined.reset_index(drop=True, inplace=True)

        # Concatenate df_trials DataFrames
        df_nph_combined = pd.concat([df_nph_combined, df_nph], axis=0)

        # Reset index of the combined DataFrame
        df_nph_combined.reset_index(drop=True, inplace=True)

        # Print shapes to verify
        print("Shape of psth_combined:", psth_combined.shape)
        print("Shape of df_trials_combined:", df_trials_combined.shape)
        print("Shape of df_nph_combined:", df_nph_combined.shape)
    except: 
        pass

##################################################################################################
# PLOT heatmap and correct vs incorrect 
psth_good = df_nph_combined.calcium_mad.values[psth_combined[:,(df_trials_combined.feedbackType == 1)]]
psth_error = df_nph_combined.calcium_mad.values[psth_combined[:,(df_trials_combined.feedbackType == -1)]]
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

# MIGHT BE WRONG!!!!! 
# ticks = np.linspace(0, len(psth_good_avg), 4)  # Assuming 91 points, set 4 tick marks
# tick_labels = [-1, 0, 1, 2]    # Labels corresponding to time from -1 to 2 seconds

fig.suptitle(f'calcium_mad_{EVENT}_{mouse}_{date}_{region}_{NM}_{eid}', y=1, fontsize=14)
plt.tight_layout()
plt.show()

#%%
columns = df_nph.columns 
for column in columns[6:len(columns)-1]: 
    fig = plt.figure(figsize=(20, 8))
    plt.plot(df_nph[column], linewidth=0.25, alpha=0.5) 
    # plt.xlim(25000,50500)
    plt.title(column)
    plt.show()
# %%


















































#%% 
#################################################################################
#################################################################################
# NEW VIDEOS HABITUATION CHOICE WORLD HCW 2024-11-22 

eid = '13319506-3b45-4e94-85c6-b1080dc7b10a' 
# eid = 'b105bea6-b3d4-46e2-af41-4e7e53277f27' 
ref = one.eid2ref(eid)
ref

video_data = pd.read_parquet('/home/ibladmin/Downloads/_ibl_leftCamera.lightningPose.c10bfdf3-aecd-4a77-a13e-fe78335d45f8.pqt')
test = one.load_object(eid, 'leftCamera', attribute=['times']) 
video_data["times"] = test.times
plt.rcParams["figure.figsize"] = (20,3)
for column in video_data.columns: 
    plt.plot(video_data[column], linewidth=0.5, alpha=0.85)
    plt.legend()
    plt.title(column)
    plt.show() 



#%%
a = one.load_object(eid, 'trials')
df_trials = a.to_df() 

#%% 
video_data["moving_avg_top_r_x"] = video_data["pupil_top_r_x"].rolling(window=30).mean()
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(video_data["times"], video_data["moving_avg_top_r_x"], linewidth=1, alpha=0.85, color='#582f0e')
for x in df_trials.feedback_times: 
    plt.axvline(x, linewidth=0.5, alpha=1, color = '#2978a0')
plt.legend()
plt.title(column)
plt.show() 
# %%







