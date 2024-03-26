

#%%
import pandas as pd 
from brainbox.io.one import SessionLoader
from one.api import ONE
one = ONE()
import matplotlib.pyplot as plt 
import seaborn as sns
# from functions_nm import load_trials 
from functions_nm import *

#%%
""" PHOTOMETRY """
date = "2024-01-19"
mouse = "ZFM-06275"
nphfile_number = "1"
bncfile_number = "1"
region = "Region6G"

source_folder = ("/home/kceniabougrova/Documents/nph/"+date+"/")
df_nph = pd.read_csv(source_folder+"raw_photometry"+nphfile_number+".csv") 
df_nphttl = pd.read_csv(source_folder+"bonsai_DI"+bncfile_number+nphfile_number+".csv")

"""
********** EXPLAINING df_nph **********
    LedState        - ignore anything which is not a 2 or 1; process the data in order to start with one of them and end with another; 1 is isosbestic and 2 is GCaMP 
    Input0, Input1  - whenever "1" is when the TTL was sent, but the Timestamp corresponds to the time when the LED turned on (so it is most likely that the TTL arrived in between the previous row and the "current" row) 
                    - use df_nphttl for this 
    RegionXG        - recorded "brightness" of the ROI

TAKE INTO ACCOUNT: 
LedState, GCaMP and isos, is interleaved - so we need to extract 2 df's from there 

********** EXPLAINING df_nphttl ********** 
    Value.Seconds       - used to align 
    Value.Value         - filter only for True (it's when the TTL was received) 

"""

# %% 
""" BEHAVIOR """
eids = one.search(subject=mouse, date=date) 
len(eids)
eid = eids[0]
ref = one.eid2ref(eid)
print(ref) 

# %% 
a = one.load_object(eid, 'trials')
trials = a.to_df()


# %%
df_DI0 = df_nphttl
if 'Value.Value' in df_DI0.columns: #for the new ones
    df_DI0 = df_DI0.rename(columns={"Value.Seconds": "Seconds", "Value.Value": "Value"})
else:
    df_DI0["Timestamp"] = df_DI0["Seconds"] #for the old ones
#use Timestamp from this part on, for any of the files
raw_phdata_DI0_true = df_DI0[df_DI0.Value==True]
df_raw_phdata_DI0_T_timestamp = pd.DataFrame(raw_phdata_DI0_true, columns=["Timestamp"])
df_raw_phdata_DI0_T_timestamp = df_raw_phdata_DI0_T_timestamp.reset_index(drop=True)

#%%
import numpy as np
df_trials = df
tph = df_raw_phdata_DI0_T_timestamp.values[:, 0]
tbpod = np.sort(np.r_[df_trials['intervals_0'].values, df_trials['intervals_1'].values, df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values])

#%%
import neurodsp.utils 
# plotting tnph and tbpod
fig, axs = plt.subplots(2, 1)
axs[0].plot(np.diff(tph))
axs[0].plot(np.diff(tbpod))
axs[0].legend(['ph', 'pbod'])

fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = neurodsp.utils.sync_timestamps(tph, tbpod, return_indices=True)

print('max deviation:', np.max(np.abs(fcn_nph_to_bpod_times(tph[iph]) - tbpod[ibpod]) * 1e6), 'drift: ', drift_ppm, 'ppm')

#fcn_nph_to_bpod_times  # apply this function to whatever photometry timestamps

axs[1].plot(np.diff(fcn_nph_to_bpod_times(tph[iph])))
axs[1].plot(np.diff(tbpod[ibpod]))
axs[1].legend(['ph', 'pbod']) 

#%% 
intervals_0_event = np.sort(np.r_[df_trials['intervals_0'].values]) 
len(intervals_0_event)

# transform the nph TTL times into bpod times 
df_PhotometryData = df_nph
nph_sync = fcn_nph_to_bpod_times(tph[iph]) 
bpod_times = tbpod[ibpod] 
bpod_sync = bpod_times
bpod_1event_times = tbpod[ibpod]
nph_to_bpod = nph_sync
fig1, ax = plt.subplots()

ax.set_box_aspect(1)
plt.plot(nph_to_bpod, bpod_1event_times) 
plt.show()

df_PhotometryData["bpod_frame_times_feedback_times"] = fcn_nph_to_bpod_times(df_PhotometryData["Timestamp"]) 
#%% 
plt.figure(figsize=(20, 10))
plt.plot(df_PhotometryData.bpod_frame_times_feedback_times, df_PhotometryData[region],color = "#25a18e") 

xcoords = df_alldata.feedback_times
for xc in zip(xcoords):
    plt.axvline(x=xc, color='blue',linewidth=0.3)
plt.title("Entire signal, raw data")
plt.legend(["GCaMP","isosbestic"],frameon=False)
sns.despine(left = False, bottom = False) 
# plt.axvline(x=init_idx) 
# plt.axvline(x=end_idx) 
plt.show()

# %%
# Assuming nph_sync contains the timestamps in seconds
nph_sync_start = nph_sync[0] - 60  # Start time, 100 seconds before the first nph_sync value
nph_sync_end = nph_sync[-1] + 60   # End time, 100 seconds after the last nph_sync value

# Select data within the specified time range
selected_data = df_PhotometryData[
    (df_PhotometryData['bpod_frame_times_feedback_times'] >= nph_sync_start) &
    (df_PhotometryData['bpod_frame_times_feedback_times'] <= nph_sync_end)
]

# Now, selected_data contains the rows of df_PhotometryData within the desired time range 
selected_data 

# Plotting the new filtered data 
plt.figure(figsize=(20, 10))
plt.plot(selected_data.bpod_frame_times_feedback_times, selected_data[region],color = "#25a18e") 

xcoords = nph_sync
for xc in zip(xcoords):
    plt.axvline(x=xc, color='blue',linewidth=0.3)
plt.title("Entire signal, raw data")
plt.legend(["GCaMP","isosbestic"],frameon=False)
sns.despine(left = False, bottom = False) 
# plt.axvline(x=init_idx) 
# plt.axvline(x=end_idx) 
plt.show()
# %%
df_PhotometryData = selected_data

#%% 
#===========================================================================
#      4. FUNCTIONS TO LOAD DATA AND ADD SOME VARIABLES (BEHAVIOR)
#===========================================================================
df_PhotometryData = df_PhotometryData.reset_index(drop=True)
df_PhotometryData = LedState_or_Flags(df_PhotometryData)

""" 4.1.2 Check for LedState/previous Flags bugs """ 
""" 4.1.2.1 Length """
# Verify the length of the data of the 2 different LEDs
df_470, df_415 = verify_length(df_PhotometryData)
""" 4.1.2.2 Verify if there are repeated flags """ 
verify_repetitions(df_PhotometryData["LedState"])
""" 4.1.3 Remove "weird" data (flag swap, huge signal) """ 
session_day=date
plot_outliers(df_470,df_415,region,mouse,session_day) 
# %%
# Select a subset of df_PhotometryData and reset the index
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
plt.plot(df_470[region], c='#279F95', linewidth=0.5)
plt.plot(df_415[region], c='#803896', linewidth=0.5)
plt.title("Cropped signal, what to use next")
plt.legend(["GCaMP", "isosbestic"], frameon=False)
sns.despine(left=False, bottom=False)
plt.show()

# Print counts
print("470 =", df_470['LedState'].count(), " 415 =", df_415['LedState'].count())


#%%
# Select a subset of df_PhotometryData and reset the index
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
plt.plot(df_470[region], c='#279F95', linewidth=0.5)
plt.plot(df_415[region], c='#803896', linewidth=0.5)
plt.title("Cropped signal, what to use next")
plt.legend(["GCaMP", "isosbestic"], frameon=False)
sns.despine(left=False, bottom=False)
plt.show()

# Print counts
print("470 =", df_470['LedState'].count(), " 415 =", df_415['LedState'].count())

#%%
# %% 
df_PhotometryData = df_PhotometryData_1.reset_index(drop=True)  
df_470 = df_PhotometryData[df_PhotometryData.LedState==2] 
df_470 = df_470.reset_index(drop=True)
df_415 = df_PhotometryData[df_PhotometryData.LedState==1] 
df_415 = df_415.reset_index(drop=True) 
#================================================
""" 4.1.4 FRAME RATE """ 
acq_FR = find_FR(df_470["Timestamp"]) 


#%%
import scipy.signal

raw_reference = df_415[region] #isosbestic 
raw_signal = df_470[region] #GCaMP signal 



sos = scipy.signal.butter(**{'N': 3, 'Wn': 0.05, 'btype': 'highpass'}, output='sos')
butt = scipy.signal.sosfiltfilt(sos, raw_signal) 

plt.plot(butt)
butt = scipy.signal.sosfiltfilt(sos, raw_reference) 
plt.plot(butt,alpha=0.5)
plt.show()

# %%
raw_reference = df_415[region] #isosbestic 
raw_signal = df_470[region] #GCaMP signal 
raw_timestamps_bpod = df_470["bpod_frame_times_feedback_times"]
raw_timestamps_nph_470 = df_470["Timestamp"]
raw_timestamps_nph_415 = df_415["Timestamp"]
raw_TTL_bpod = bpod_sync
raw_TTL_nph = nph_sync

plt.plot(raw_signal[:],color="#60d394")
plt.plot(raw_reference[:],color="#c174f2") 
plt.legend(["signal","isosbestic"],fontsize=15, loc="best")
plt.show() 
# %%

my_array = np.c_[raw_timestamps_bpod, raw_reference, raw_signal]

df = pd.DataFrame(my_array, columns=['times', 'raw_isosbestic', 'raw_calcium'])


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import iblphotometry.plots
import iblphotometry.dsp


df_photometry = iblphotometry.dsp.baseline_correction_dataframe(df)


fig, ax = iblphotometry.plots.plot_raw_data_df(df_photometry)