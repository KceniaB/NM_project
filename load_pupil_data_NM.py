#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from brainbox.behavior.dlc import likelihood_threshold
from brainbox.behavior.dlc import get_speed
from scipy.signal import convolve
import seaborn as sns
# from functions_nm import load_trials 
import iblphotometry.kcenia as kcenia 
import ibldsp.utils 
from pathlib import Path
import iblphotometry.plots
import iblphotometry.dsp 
from brainbox.io.one import SessionLoader
# import functions_nm 
import scipy.signal
import ibllib.plots
from one.api import ONE #always after the imports 
one = ONE()

eids = one.search(subject="ZFM-03059") 
eid = eids[70]
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
df=trials 
label = 'left' # 'left', 'right' or 'body'

video_features = one.load_object(eid, f'{label}Camera', collection='alf') 

# Set values with likelihood below chosen threshold to NaN

dlc = likelihood_threshold(video_features['dlc'], threshold=0.9) 





# Compute the speed of the right paw
feature = 'paw_r'
dlc_times = video_features['times']
paw_r_speed = get_speed(dlc, dlc_times, label, feature=feature) 






video_features_dlc = video_features.dlc


diameters = []
# Get the x,y coordinates of the four pupil points
top, bottom, left, right = [np.vstack((dlc[f'pupil_{point}_r_x'], dlc[f'pupil_{point}_r_y']))
                            for point in ['top', 'bottom', 'left', 'right']]
# First compute direct diameters
diameters.append(np.linalg.norm(top - bottom, axis=0))
diameters.append(np.linalg.norm(left - right, axis=0))

# For non-crossing edges, estimate diameter via circle assumption
for pair in [(top, left), (top, right), (bottom, left), (bottom, right)]:
    diameters.append(np.linalg.norm(pair[0] - pair[1], axis=0) * 2 ** 0.5) 




plt.scatter(dlc.pupil_top_r_x,dlc.pupil_top_r_y)
plt.scatter(dlc.pupil_right_r_x,dlc.pupil_right_r_y)
plt.scatter(dlc.pupil_bottom_r_x,dlc.pupil_bottom_r_y)
plt.scatter(dlc.pupil_left_r_x,dlc.pupil_left_r_y)



diameters = []
# Get the x,y coordinates of the four pupil points
top, bottom, left, right = [np.vstack((dlc[f'pupil_{point}_r_x'], dlc[f'pupil_{point}_r_y']))
                            for point in ['top', 'bottom', 'left', 'right']]




diameters.append(np.linalg.norm(top - bottom, axis=0))
diameters.append(np.linalg.norm(left - right, axis=0))

for pair in [(top, left), (top, right), (bottom, left), (bottom, right)]:
    diameters.append(np.linalg.norm(pair[0] - pair[1], axis=0) * 2 ** 0.5)

a =     np.nanmedian(diameters, axis=0) 
df = pd.DataFrame(video_features["times"], columns=["times"])
df["diameter"] = a


def smooth_array(arr, window_size):
    kernel = np.ones(window_size) / window_size
    smoothed_arr = convolve(arr, kernel, mode='same')
    return smoothed_arr

# Define the figure size
fig = plt.figure(figsize=(15, 6))

# Plot the original data
# plt.plot(df.times, df.diameter, linewidth=0.5)
xcoords = trials.feedback_times
for xc in zip(xcoords):
    plt.axvline(x=xc, color='gray', linewidth=0.3)
plt.xlim(1000, 1500)
plt.ylim(7.5, 15)

# Plot the smoothed line
window_size = 100  # Adjust the window size for smoothing
smoothed_diameter = smooth_array(df.diameter, window_size)
plt.plot(df.times, smoothed_diameter, color='red', linewidth=1.3)

# Show the plot
plt.show()


















# %%
""" Try an example session """


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from brainbox.behavior.dlc import likelihood_threshold
from brainbox.behavior.dlc import get_speed
from scipy.signal import convolve


from one.api import ONE
one = ONE()
source_folder = (f"/home/kceniabougrova/Documents/nph/2021-10-14/")
df_nph = pd.read_csv(source_folder+f"PhotometryData_M1_M4_TCW_S35_14Oct2021.csv") 
df_nphttl = pd.read_csv(source_folder+f"DI0_M1_M4_TCW_S35_14Oct2021.csv") 
region = "Region1G"




eids = one.search(subject="ZFM-03059") 
eid = eids[70]
ref = one.eid2ref(eid)
print(eid)
print(ref) 

a = one.load_object(eid, 'trials')
df_trials = a.to_df()



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
df=trials 
label = 'left' # 'left', 'right' or 'body'

video_features = one.load_object(eid, f'{label}Camera', collection='alf') 

# Set values with likelihood below chosen threshold to NaN

dlc = likelihood_threshold(video_features['dlc'], threshold=0.9) 



# Compute the speed of the right paw
feature = 'paw_r'
dlc_times = video_features['times']
paw_r_speed = get_speed(dlc, dlc_times, label, feature=feature) 






video_features_dlc = video_features.dlc


diameters = []
# Get the x,y coordinates of the four pupil points
top, bottom, left, right = [np.vstack((dlc[f'pupil_{point}_r_x'], dlc[f'pupil_{point}_r_y']))
                            for point in ['top', 'bottom', 'left', 'right']]
# First compute direct diameters
diameters.append(np.linalg.norm(top - bottom, axis=0))
diameters.append(np.linalg.norm(left - right, axis=0))

# For non-crossing edges, estimate diameter via circle assumption
for pair in [(top, left), (top, right), (bottom, left), (bottom, right)]:
    diameters.append(np.linalg.norm(pair[0] - pair[1], axis=0) * 2 ** 0.5) 






plt.scatter(dlc.pupil_top_r_x,dlc.pupil_top_r_y)
plt.scatter(dlc.pupil_right_r_x,dlc.pupil_right_r_y)
plt.scatter(dlc.pupil_bottom_r_x,dlc.pupil_bottom_r_y)
plt.scatter(dlc.pupil_left_r_x,dlc.pupil_left_r_y)



diameters = []
# Get the x,y coordinates of the four pupil points
top, bottom, left, right = [np.vstack((dlc[f'pupil_{point}_r_x'], dlc[f'pupil_{point}_r_y']))
                            for point in ['top', 'bottom', 'left', 'right']]




diameters.append(np.linalg.norm(top - bottom, axis=0))
diameters.append(np.linalg.norm(left - right, axis=0))

for pair in [(top, left), (top, right), (bottom, left), (bottom, right)]:
    diameters.append(np.linalg.norm(pair[0] - pair[1], axis=0) * 2 ** 0.5)

a =     np.nanmedian(diameters, axis=0) 
df = pd.DataFrame(video_features["times"], columns=["times"])
df["diameter"] = a


def smooth_array(arr, window_size):
    kernel = np.ones(window_size) / window_size
    smoothed_arr = convolve(arr, kernel, mode='same')
    return smoothed_arr

# Define the figure size
fig = plt.figure(figsize=(15, 6))

# Plot the original data
# plt.plot(df.times, df.diameter, linewidth=0.5)
xcoords = trials.feedback_times
for xc in zip(xcoords):
    plt.axvline(x=xc, color='gray', linewidth=0.3)
plt.xlim(1000, 1500)
plt.ylim(7.5, 15)

# Plot the smoothed line
window_size = 100  # Adjust the window size for smoothing
smoothed_diameter = smooth_array(df.diameter, window_size)
plt.plot(df.times, smoothed_diameter, color='red', linewidth=1.3)

# Show the plot
plt.show()

df_pupil = df


#%% 
""" nph """
df_DI0 = df_nphttl
if 'Value.Value' in df_DI0.columns: #for the new ones
    df_DI0 = df_DI0.rename(columns={"Value.Seconds": "Seconds", "Value.Value": "Value"})
else:
    df_DI0["Timestamp"] = df_DI0["Timestamp"] #for the old ones
#use Timestamp from this part on, for any of the files
raw_phdata_DI0_true = df_DI0[df_DI0.Value==True]
df_raw_phdata_DI0_T_timestamp = pd.DataFrame(raw_phdata_DI0_true, columns=["Timestamp"])
# raw_phdata_DI0_true = pd.DataFrame(df_DI0.Timestamp[df_DI0.Value==True], columns=['Timestamp'])
df_raw_phdata_DI0_T_timestamp = df_raw_phdata_DI0_T_timestamp.reset_index(drop=True) 
tph = df_raw_phdata_DI0_T_timestamp.values[:, 0] 
# tbpod = np.sort(np.r_[df_trials['stimOnTrigger_times'].values])
tbpod = np.sort(np.r_[df_trials['intervals_0'].values, df_trials['intervals_1'].values, df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values])

try:
    tbpod = np.sort(np.r_[
        df_trials['intervals_0'].values,
        df_trials['intervals_1'].values,
        df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
    )
    fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
    assert len(iph)/len(tbpod) > .9
except AssertionError:
    print("mismatch in sync, will try to add ITI duration to the sync")
    try:
        tbpod = np.sort(np.r_[
            df_trials['intervals_0'].values,
            df_trials['intervals_1'].values - 1,  # here is the trick
            df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
        )
        fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
        assert len(iph)/len(tbpod) > .9
        print("recovered from sync mismatch, continuing")
    except AssertionError:
        print("mismatch, maybe this is an old session")
        tbpod = np.sort(np.r_[df_trials['stimOnTrigger_times'].values])
        fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
        assert len(iph)/len(tbpod) > .9
        print("recovered from sync mismatch, continuing #2")

fig, axs = plt.subplots(2, 1)
axs[0].plot(np.diff(tph))
axs[0].plot(np.diff(tbpod))
axs[0].legend(['ph', 'pbod'])
print('max deviation:', np.max(np.abs(fcn_nph_to_bpod_times(tph[iph]) - tbpod[ibpod]) * 1e6), 'drift: ', drift_ppm, 'ppm')
#fcn_nph_to_bpod_times  # apply this function to whatever photometry timestamps
axs[1].plot(np.diff(fcn_nph_to_bpod_times(tph[iph])))
axs[1].plot(np.diff(tbpod[ibpod]))
axs[1].legend(['ph', 'pbod'])
df_PhotometryData = df_nph

# transform the nph TTL times into bpod times 
nph_sync = fcn_nph_to_bpod_times(tph[iph]) 
bpod_sync = tbpod[ibpod] #same bpod_sync = tbpod
fig1, ax = plt.subplots()
ax.set_box_aspect(1)
plt.plot(nph_sync, bpod_sync) 
plt.show(block=False)
plt.close() 

df_PhotometryData["bpod_frame_times_feedback_times"] = fcn_nph_to_bpod_times(df_PhotometryData["Timestamp"])

regions=region

# Assuming nph_sync contains the timestamps in seconds
nph_sync_start = nph_sync[0] - 30  # Start time, 100 seconds before the first nph_sync value
nph_sync_end = nph_sync[-1] + 30   # End time, 100 seconds after the last nph_sync value

# Select data within the specified time range
selected_data = df_PhotometryData[
    (df_PhotometryData['bpod_frame_times_feedback_times'] >= nph_sync_start) &
    (df_PhotometryData['bpod_frame_times_feedback_times'] <= nph_sync_end)
]

# Now, selected_data contains the rows of df_PhotometryData within the desired time range 
selected_data 

# Plotting the new filtered data 
plt.figure(figsize=(20, 10))
plt.plot(selected_data.bpod_frame_times_feedback_times, selected_data[regions],color = "#25a18e") 
xcoords = nph_sync
for xc in zip(xcoords):
    plt.axvline(x=xc, color='blue',linewidth=0.3)
plt.title("Entire signal, raw data")
plt.legend(["GCaMP","isosbestic"],frameon=False)
sns.despine(left = False, bottom = False) 
# plt.axvline(x=init_idx) 
# plt.axvline(x=end_idx) 
plt.show(block=False)
plt.close()

df_PhotometryData = selected_data

#===========================================================================
#      4. FUNCTIONS TO LOAD DATA AND ADD SOME VARIABLES (BEHAVIOR)
#===========================================================================
df_PhotometryData = df_PhotometryData.reset_index(drop=True)
df_PhotometryData = kcenia.LedState_or_Flags(df_PhotometryData)

""" 4.1.2 Check for LedState/previous Flags bugs """ 
""" 4.1.2.1 Length """
# Verify the length of the data of the 2 different LEDs
df_470, df_415 = kcenia.verify_length(df_PhotometryData)
""" 4.1.2.2 Verify if there are repeated flags """ 
kcenia.verify_repetitions(df_PhotometryData["LedState"])
""" 4.1.3 Remove "weird" data (flag swap, huge signal) """ 
# session_day=rec.date
# plot_outliers(df_470,df_415,region,mouse,session_day) 

df_PhotometryData_1 = df_PhotometryData

# Remove rows with LedState 1 at both ends if present
if df_PhotometryData_1['LedState'].iloc[0] == 1 and df_PhotometryData_1['LedState'].iloc[-1] == 1:
    df_PhotometryData_1 = df_PhotometryData_1.iloc[1:]

# Remove rows with LedState 2 at both ends if present
if df_PhotometryData_1['LedState'].iloc[0] == 2 and df_PhotometryData_1['LedState'].iloc[-1] == 2:
    df_PhotometryData_1 = df_PhotometryData_1.iloc[:-2]

# Filter data for LedState 2 (470nm)
df_470 = df_PhotometryData_1[df_PhotometryData_1['LedState'] == 2]

# Filter data for LedState 1 (415nm)
df_415 = df_PhotometryData_1[df_PhotometryData_1['LedState'] == 1]

# Check if the lengths of df_470 and df_415 are equal
assert len(df_470) == len(df_415), "Sync arrays are of different lengths"







# Plot the data
plt.rcParams["figure.figsize"] = (8, 5)
plt.plot(df_470[regions], c='#279F95', linewidth=0.5)
plt.plot(df_415[regions], c='#803896', linewidth=0.5)
plt.title("Cropped signal")
plt.legend(["GCaMP", "isosbestic"], frameon=False)
sns.despine(left=False, bottom=False)
plt.show(block=False)
plt.close() 
# Print counts
print("470 =", df_470['LedState'].count(), " 415 =", df_415['LedState'].count())

df_PhotometryData = df_PhotometryData_1.reset_index(drop=True)  
df_470 = df_PhotometryData[df_PhotometryData.LedState==2] 
df_470 = df_470.reset_index(drop=True)
df_415 = df_PhotometryData[df_PhotometryData.LedState==1] 
df_415 = df_415.reset_index(drop=True) 
#================================================
""" 4.1.4 FRAME RATE """ 
acq_FR = kcenia.find_FR(df_470["Timestamp"]) 

raw_reference = df_415[regions] #isosbestic 
raw_signal = df_470[regions] #GCaMP signal 
raw_timestamps_bpod = df_470["bpod_frame_times_feedback_times"]
raw_timestamps_nph_470 = df_470["Timestamp"]
raw_timestamps_nph_415 = df_415["Timestamp"]
raw_TTL_bpod = bpod_sync
raw_TTL_nph = nph_sync

my_array = np.c_[raw_timestamps_bpod, raw_reference, raw_signal]

df = pd.DataFrame(my_array, columns=['times', 'raw_isosbestic', 'raw_calcium'])

df_photometry = iblphotometry.dsp.baseline_correction_dataframe(df)



#%%
#run once: 
df_original = df
df_pupil_original = df_pupil
df = df_original[3000:len(df_original)]
df_pupil = df_pupil_original[3000:len(df_pupil_original)]

#%%

def smooth_array(arr, window_size):
    kernel = np.ones(window_size) / window_size
    smoothed_arr = np.convolve(arr, kernel, mode='same')
    return smoothed_arr

fig, ax1 = plt.subplots(figsize=(15, 6))

# Plot the original data
# plt.plot(df.times, df.diameter, linewidth=0.5) 
# Plot the events 
xcoords = trials.feedback_times
for xc in zip(xcoords):
    ax1.axvline(x=xc, color='#38040e', alpha=0.3, linewidth=0.75)

# Plot the smoothed line
window_size = 250  # Adjust the window size for smoothing
smoothed_diameter = smooth_array(df_pupil.diameter, window_size)
ax1.plot(df_pupil.times, smoothed_diameter, color='black', linewidth=1.5)
ax1.set_xlabel('Time')
ax1.set_ylabel('Pupil Diameter', color='black')

# Create another y-axis on the right for df.calcium 
smoothed_calcium = smooth_array(df.calcium, window_size)
ax2 = ax1.twinx()
ax2.plot(df.times, smoothed_calcium, color='teal', linewidth=1.5)
ax2.set_ylabel('Calcium', color='teal')
plt.title("5-HT pupil and calcium oscillations")
plt.xlim(3000,3800)

# Remove gridlines
ax1.grid(False)
ax2.grid(False)

# Show the plot
plt.show()










# %%
array_timestamps_bpod = np.array(df.times) #pick the nph timestamps transformed to bpod clock 
FONTSIZE_1 = 30
FONTSIZE_2 = 25
FONTSIZE_3 = 15
def test_test_test(df_alldata=trials, a="feedback_times", df=df):
    PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
    SAMPLING_RATE = 15 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR
    sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
    n_trials = df_alldata.shape[0]

    # psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
    psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials-1))

    event_feedback = np.array(df_alldata[a]) #pick the feedback timestamps 
    event_feedback = event_feedback[0:len(event_feedback)-1] #KB added 20240327 CHECK WITH OW

    feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

    psth_idx += feedback_idx

    event_time=15
    def create_heatmap(data, ax, ax_label, event_time):
        sns.heatmap(data.T, cbar=False, ax=ax, linewidths=0)
        ax.axvline(x=event_time, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax.set_xlabel('time since event (s)', fontsize=FONTSIZE_2)
        ax.set_ylabel(ax_label, fontsize=FONTSIZE_2)

    # Function to create the line plot
    def create_line_plot(data, ax, event_time):
        average_values = data.mean(axis=1)
        ax.plot(average_values, color='black')
        ax.set_xlabel('time since event (s)', fontsize=FONTSIZE_2)
        ax.set_ylabel('zdFF', fontsize=FONTSIZE_2)
        ax.axvline(x=event_time, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax.grid(False)

    # fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, gridspec_kw={'height_ratios': [4, 1]}, figsize=(20, 15), sharex=True) 
    fig, ((ax1), (ax5)) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(6, 12), sharex=True) 

    # create_heatmap(df.raw_isosbestic.values[psth_idx], ax1, 'Raw Isosbestic', event_time)
    # create_heatmap(df.raw_calcium.values[psth_idx], ax2, 'Raw Calcium', event_time)
    create_heatmap(df.calcium.values[psth_idx], ax1, 'Calcium', event_time)
    # create_heatmap(df.zdFF.values[psth_idx], ax4, 'zdFF', event_time)

    # create_line_plot(df.raw_isosbestic.values[psth_idx], ax5, event_time)
    create_line_plot(df.raw_calcium.values[psth_idx], ax5, event_time)
    # create_line_plot(df.calcium.values[psth_idx], ax7, event_time)
    # create_line_plot(df.zdFF.values[psth_idx], ax8, event_time) 
    # plt.suptitle(a+" "+nm+" "+mouse+" "+date+" "+region+" "+nphfile_number+" "+bncfile_number, fontsize=FONTSIZE_1) 
    for ax in [ax1, ax5]:
        ax.tick_params(axis='x', labelsize=FONTSIZE_3)
        ax.tick_params(axis='y', labelsize=FONTSIZE_3)

    plt.tight_layout()
    path = '/home/kceniabougrova/Documents/results/figures'
    # path_fig = 'fig1_'+mouse+'_'+date+'_'+region+'_'+a 
    # plt.savefig(os.path.join(path, path_fig + '.png')) 
    # plt.savefig(os.path.join(path, path_fig + '.pdf')) #not really working 
    plt.show() 

test_test_test(df_alldata=trials, a="feedback_times", df=df) 

# %%
