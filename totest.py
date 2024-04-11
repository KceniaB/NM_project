"""
KB Photometry pre-processing 2024 
inputs: 
    d, m, n, b, r

outputs: 
    save csv photometry
    save csv behavior
    save figures 

"""

#%%
import numpy as np
import pandas as pd 
from brainbox.io.one import SessionLoader
from one.api import ONE
one = ONE()
import matplotlib.pyplot as plt 
import seaborn as sns
# from functions_nm import load_trials 
from functions_nm import * 
import neurodsp.utils 
from pathlib import Path
import iblphotometry.plots
import iblphotometry.dsp


#%%
""" PHOTOMETRY """
date = "2024-03-22"
mouse = "ZFM-06948"
nphfile_number = "0"
bncfile_number = "0"
region = "Region6G" 

if mouse == "ZFM-06948" or mouse == "ZFM-06305":
    nm = "ACh"

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

#%%
"""
Alternative to extract from the raw data
""" 
session_path_behav = '/home/kceniabougrova/Documents/nph/Behav_2024Mar20/ZFM-06948/2024-03-22/001/' 

from ibllib.io.extractors.biased_trials import extract_all 
df_alldata = extract_all(session_path_behav)

# Extract the 'table' key from the first OrderedDict in df_alldata
table_data = df_alldata[0]['table']

# Convert the 'table' data into a DataFrame
df_table = pd.DataFrame(table_data) 
df_alldata = df_table 
trials = df_table


# %%
df_DI0 = df_nphttl
if 'Value.Value' in df_DI0.columns: #for the new ones
    df_DI0 = df_DI0.rename(columns={"Value.Seconds": "Seconds", "Value.Value": "Value"})
else:
    df_DI0["Timestamp"] = df_DI0["Seconds"] #for the old ones
#use Timestamp from this part on, for any of the files
raw_phdata_DI0_true = df_DI0[df_DI0.Value==True]
df_raw_phdata_DI0_T_timestamp = pd.DataFrame(raw_phdata_DI0_true, columns=["Timestamp"])
# raw_phdata_DI0_true = pd.DataFrame(df_DI0.Timestamp[df_DI0.Value==True], columns=['Timestamp'])
df_raw_phdata_DI0_T_timestamp = df_raw_phdata_DI0_T_timestamp.reset_index(drop=True)

#%%
df_trials = trials
tph = df_raw_phdata_DI0_T_timestamp.values[:, 0]
tbpod = np.sort(np.r_[df_trials['intervals_0'].values, df_trials['intervals_1'].values, df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values])





#%%
""" from old code """ 
import neurodsp.utils
try:
    tbpod = np.sort(np.r_[
    df_trials['intervals_0'].values,
    df_trials['intervals_1'].values,
    df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
    )
    fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = neurodsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
    assert len(iph)/len(tbpod) > .9
except AssertionError:
    print("mismatch in sync, will try to add ITI duration to the sync")
    tbpod = np.sort(np.r_[
    df_trials['intervals_0'].values,
    df_trials['intervals_1'].values - 1,  # here is the trick
    df_trials.loc[df_trials['feedbackType'] == 1, 'feedback_times'].values]
    )
    fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = neurodsp.utils.sync_timestamps(tph, tbpod, return_indices=True) 
    assert len(iph)/len(tbpod) > .9
    print("recovered from sync mismatch, continuing")








#%%
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
df_alldata=df_trials
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
# # Select a subset of df_PhotometryData and reset the index
# df_PhotometryData_1 = df_PhotometryData

# # Remove rows with LedState 1 at both ends if present
# if df_PhotometryData_1['LedState'].iloc[0] == 1 and df_PhotometryData_1['LedState'].iloc[-1] == 1:
#     df_PhotometryData_1 = df_PhotometryData_1.iloc[1:]

# # Remove rows with LedState 2 at both ends if present
# if df_PhotometryData_1['LedState'].iloc[0] == 2 and df_PhotometryData_1['LedState'].iloc[-1] == 2:
#     df_PhotometryData_1 = df_PhotometryData_1.iloc[:-2]

# # Filter data for LedState 2 (470nm)
# df_470 = df_PhotometryData_1[df_PhotometryData_1['LedState'] == 2]

# # Filter data for LedState 1 (415nm)
# df_415 = df_PhotometryData_1[df_PhotometryData_1['LedState'] == 1]

# # Check if the lengths of df_470 and df_415 are equal
# assert len(df_470) == len(df_415), "Sync arrays are of different lengths"

# # Plot the data
# plt.rcParams["figure.figsize"] = (8, 5)
# plt.plot(df_470[region], c='#279F95', linewidth=0.5)
# plt.plot(df_415[region], c='#803896', linewidth=0.5)
# plt.title("Cropped signal, what to use next")
# plt.legend(["GCaMP", "isosbestic"], frameon=False)
# sns.despine(left=False, bottom=False)
# plt.show()

# # Print counts
# print("470 =", df_470['LedState'].count(), " 415 =", df_415['LedState'].count())

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
# import scipy.signal

# raw_reference = df_415[region] #isosbestic 
# raw_signal = df_470[region] #GCaMP signal 



# sos = scipy.signal.butter(**{'N': 3, 'Wn': 0.05, 'btype': 'highpass'}, output='sos')
# butt = scipy.signal.sosfiltfilt(sos, raw_signal) 

# plt.plot(butt)
# butt = scipy.signal.sosfiltfilt(sos, raw_reference) 
# plt.plot(butt,alpha=0.5)
# plt.show()

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

df_photometry = iblphotometry.dsp.baseline_correction_dataframe(df)


fig, ax = iblphotometry.plots.plot_raw_data_df(df_photometry)




# %%
""" search the idx where "event_test" fits in "df_test" in a sorted way """
array_timestamps_bpod = np.array(df.times) #pick the nph timestamps transformed to bpod clock 
event_test = np.array(df_alldata.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(array_timestamps_bpod, event_test) #check idx where they would be included, in a sorted way 
print(idx_event)


#%%
""" create a column with the trial number in the nph df """
df["trial_number"] = 0 #create a new column for the trial_number 
# for i in idx_event: 
#     df["trial_number"][i] = 1 #add an 1 whenever that event occurred (intervals_0) 
df.loc[idx_event,"trial_number"]=1

df["trial_number"] = df.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

# df.to_parquet('/home/kcenia/Desktop/testserver_2023_08_31/photometry20230831.parquet') 

#%% 
PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR
sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_alldata.shape[0]

# psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials-1))

event_feedback = np.array(df_alldata.feedback_times) #pick the feedback timestamps 
event_feedback = event_feedback[0:len(event_feedback)-1] #KB added 20240327 CHECK WITH OW

feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

psth_idx += feedback_idx

df.calcium.values[psth_idx] 

#%%
sns.heatmap(df.calcium.values[psth_idx])
# sns.heatmap(df.zdFF.values[psth_idx].T)

plt.axhline(y=30, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed")
# plt.savefig('/home/kceniabougrova/Documents/nph/2024-01-19/test1.png')
plt.show() 

# %%
# behav_value = df_alldata.contrastLeft.values
# trial_index = np.lexsort((np.arange(n_trials), behav_value))
sns.heatmap(df.calcium.values[psth_idx].T, cbar=True)
plt.axvline(x=30, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed")



# %%



#%%
""" ALTERNATIVE #2 """
""" https://colab.research.google.com/github/katemartian/Photometry_data_processing/blob/master/Photometry_data_processing.ipynb#scrollTo=rQ8dS2Da5ykE """
'''
get_zdFF.py calculates standardized dF/F signal based on calcium-idependent 
and calcium-dependent signals commonly recorded using fiber photometry calcium imaging

Reference:
  (1) Martianova, E., Aronson, S., Proulx, C.D. Multi-Fiber Photometry 
      to Record Neural Activity in Freely Moving Animal. J. Vis. Exp. 
      (152), e60278, doi:10.3791/60278 (2019)
      https://www.jove.com/video/60278/multi-fiber-photometry-to-record-neural-activity-freely-moving

'''

def get_zdFF(reference,signal,smooth_win=10,remove=0,lambd=5e4,porder=1,itermax=50): 
  '''
  Calculates z-score dF/F signal based on fiber photometry calcium-idependent 
  and calcium-dependent signals
  
  Input
      reference: calcium-independent signal (usually 405-420 nm excitation), 1D array
      signal: calcium-dependent signal (usually 465-490 nm excitation for 
                   green fluorescent proteins, or ~560 nm for red), 1D array
      smooth_win: window for moving average smooth, integer
      remove: the beginning of the traces with a big slope one would like to remove, integer
      Inputs for airPLS:
      lambd: parameter that can be adjusted by user. The larger lambda is,  
              the smoother the resulting background, z
      porder: adaptive iteratively reweighted penalized least squares for baseline fitting
      itermax: maximum iteration times
  Output
      zdFF - z-score dF/F, 1D numpy array
  '''
  
  import numpy as np
  from sklearn.linear_model import Lasso

 # Smooth signal
  reference = smooth_signal(reference, smooth_win)
  signal = smooth_signal(signal, smooth_win)
  
 # Remove slope using airPLS algorithm
  r_base=airPLS(reference,lambda_=lambd,porder=porder,itermax=itermax)
  s_base=airPLS(signal,lambda_=lambd,porder=porder,itermax=itermax) 

 # Remove baseline and the begining of recording
  reference = (reference[remove:] - r_base[remove:])
  signal = (signal[remove:] - s_base[remove:])   

 # Standardize signals    
  reference = (reference - np.median(reference)) / np.std(reference)
  signal = (signal - np.median(signal)) / np.std(signal)
  
 # Align reference signal to calcium signal using non-negative robust linear regression
  lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
              positive=True, random_state=9999, selection='random')
  n = len(reference)
  lin.fit(reference.reshape(n,1), signal.reshape(n,1))
  reference = lin.predict(reference.reshape(n,1)).reshape(n,)

 # z dFF    
  zdFF = (signal - reference)
 
  return zdFF

def smooth_signal(x,window_len=10,window='flat'):

    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    The code taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                'flat' window will produce a moving average smoothing.

    output:
        the smoothed signal        
    """

    import numpy as np

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': # Moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[(int(window_len/2)-1):-int(window_len/2)]

from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    i=np.arange(0,m)
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,
                 the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z

# %%
df 

# %%
# Adjust these lines depending on your dataframe
raw_reference = df['raw_isosbestic'][0:]
raw_signal = df['raw_calcium'][0:]

#%%
fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(raw_signal,'blue',linewidth=0.25)
ax2 = fig.add_subplot(212)
ax2.plot(raw_reference,'purple',linewidth=0.25)
# %%
zdFF = get_zdFF(raw_reference,raw_signal) 

df["zdFF"] = zdFF
# %%
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.plot(zdFF,'black',linewidth=0.25) 

# %% 

"""PLOTS""" 

######################""" TO PLOT IN A LOOP HEATMAP AND LINEPLOT ALIGNED TO EVENT """#################### 

# Define width ratios for subplots
width_ratios = [1, 1]
FONTSIZE_1 = 30
FONTSIZE_2 = 25
FONTSIZE_3 = 15

# Create subplots
fig, axes = plt.subplots(4, 2, figsize=(25, 20), gridspec_kw={'width_ratios': width_ratios})
fig.suptitle('Isosbestic and GCaMP traces', fontsize=FONTSIZE_1)  # Increase main title font size

# Define start and end indices for x-axis limits
start_index = df.times[int(len(df.times)/2)]
end_index = df.times[int(len(df.times)/2)+250]

# raw_isosbestic
sns.lineplot(ax=axes[0,0], x=df['times'], y=df['raw_isosbestic'], linewidth=0.1, color="#9d4edd")
axes[0,0].set_title("raw_isosbestic", fontsize=FONTSIZE_2)  # Increase title font size

sns.lineplot(ax=axes[0,1], x=df['times'], y=df['raw_isosbestic'], linewidth=2, color="#9d4edd")
axes[0,1].set_xlim(start_index, end_index)
axes[0,1].set_title("Zoomed raw_isosbestic", fontsize=FONTSIZE_2)  # Increase title font size

# raw_calcium
sns.lineplot(ax=axes[1,0], x=df['times'], y=df['raw_calcium'], linewidth=0.1, color="#43aa8b")
axes[1,0].set_title("raw_calcium", fontsize=FONTSIZE_2)  # Increase title font size

sns.lineplot(ax=axes[1,1], x=df['times'], y=df['raw_calcium'], linewidth=2, color="#43aa8b")
axes[1,1].set_xlim(start_index, end_index)
axes[1,1].set_title("Zoomed raw_calcium", fontsize=FONTSIZE_2)  # Increase title font size

# calcium
sns.lineplot(ax=axes[2,0], x=df['times'], y=df['calcium'], linewidth=0.1, color="#0081a7")
axes[2,0].set_title("calcium", fontsize=FONTSIZE_2)  # Increase title font size

sns.lineplot(ax=axes[2,1], x=df['times'], y=df['calcium'], linewidth=2, color="#0081a7")
axes[2,1].set_xlim(start_index, end_index)
axes[2,1].set_title("Zoomed calcium", fontsize=FONTSIZE_2)  # Increase title font size

# zdFF
sns.lineplot(ax=axes[3,0], x=df['times'], y=df['zdFF'], linewidth=0.1, color="#0081a7")
axes[3,0].set_title("zdFF", fontsize=FONTSIZE_2)  # Increase title font size

sns.lineplot(ax=axes[3,1], x=df['times'], y=df['zdFF'], linewidth=2, color="#0081a7")
axes[3,1].set_xlim(start_index, end_index)
axes[3,1].set_title("Zoomed zdFF", fontsize=FONTSIZE_2)  # Increase title font size

# Increase axis label font size
for ax in axes.flat:
    ax.set_xlabel('Time', fontsize=FONTSIZE_2)
    ax.set_ylabel('Values', fontsize=FONTSIZE_2) 
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_3)

plt.tight_layout()
plt.show()

#%% #####################################################################################################
######################""" TO PLOT IN A LOOP HEATMAP AND LINEPLOT ALIGNED TO EVENT """#################### 

a = ["intervals_0", "intervals_1", "goCue_times", "stimOn_times", "firstMovement_times", "feedback_times", "intervals_1"]
def test_test_test(df_alldata=df_alldata, a="feedback_times", df=df):
    PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
    SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR
    sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
    n_trials = df_alldata.shape[0]

    # psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
    psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials-1))

    event_feedback = np.array(df_alldata[a]) #pick the feedback timestamps 
    event_feedback = event_feedback[0:len(event_feedback)-1] #KB added 20240327 CHECK WITH OW

    feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

    psth_idx += feedback_idx

    event_time=30
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

    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, gridspec_kw={'height_ratios': [4, 1]}, figsize=(20, 15), sharex=True) 

    create_heatmap(df.raw_isosbestic.values[psth_idx], ax1, 'Raw Isosbestic', event_time)
    create_heatmap(df.raw_calcium.values[psth_idx], ax2, 'Raw Calcium', event_time)
    create_heatmap(df.calcium.values[psth_idx], ax3, 'Calcium', event_time)
    create_heatmap(df.zdFF.values[psth_idx], ax4, 'zdFF', event_time)

    create_line_plot(df.raw_isosbestic.values[psth_idx], ax5, event_time)
    create_line_plot(df.raw_calcium.values[psth_idx], ax6, event_time)
    create_line_plot(df.calcium.values[psth_idx], ax7, event_time)
    create_line_plot(df.zdFF.values[psth_idx], ax8, event_time) 
    plt.suptitle(a+" "+nm+" "+mouse+" "+date+" "+region+" "+nphfile_number+" "+bncfile_number, fontsize=FONTSIZE_1) 
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.tick_params(axis='x', labelsize=FONTSIZE_3)
        ax.tick_params(axis='y', labelsize=FONTSIZE_3)

    plt.tight_layout()
    path = '/home/kceniabougrova/Documents/results/figures'
    path_fig = 'fig1_'+mouse+'_'+date+'_'+region+'_'+a 
    plt.savefig(os.path.join(path, path_fig + '.png')) 
    # plt.savefig(os.path.join(path, path_fig + '.pdf')) #not really working 
    plt.show() 

for i in a: 
    test_test_test(df_alldata=df_alldata, a=i, df=df)

# %% #############################################################
######################""" TO SAVE THE DATA """#################### 

import os 

# Define the path
path = '/home/kceniabougrova/Documents/results'
path_nph = 'nph_'+mouse+'_'+date+'_'+region 
path_behav = 'behav_'+mouse+'_'+date+'_'+region 

# Ensure the directory exists
os.makedirs(path, exist_ok=True) 

# Save photometry
# to CSV file
df.to_csv(os.path.join(path, path_nph+'.csv'), index=False)
# Parquet file
df.to_parquet(os.path.join(path, path_nph+'.parquet'), index=False) 
# Save behavior
# to CSV file
df_alldata.to_csv(os.path.join(path, path_behav+'.csv'), index=False)
# Parquet file
df_alldata.to_parquet(os.path.join(path, path_behav+'.parquet'), index=False) 


# ######################""" TO READ THE DATA """#################### 
# # Read the CSV file
# df_csv = pd.read_csv(os.path.join(path, path_nph+'.csv'))
# # Read the Parquet file
# df_parquet = pd.read_parquet(os.path.join(path, path_nph+'.parquet')) 





#%%
#%% #####################################################################################################
######################""" TO PLOT IN A LOOP HEATMAP AND LINEPLOT ALIGNED TO EVENT SPLIT """#################### 

# a = ["intervals_0", "intervals_1", "goCue_times", "stimOn_times", "firstMovement_times", "feedback_times", "intervals_1"]
def test_test_test(df_alldata=df_alldata, a="feedback_times", df=df):
    PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
    SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR
    sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
    n_trials = df_alldata.shape[0]

    # psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
    psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials-1))

    event_feedback = np.array(df_alldata[a]) #pick the feedback timestamps 
    event_feedback = event_feedback[0:len(event_feedback)-1] #KB added 20240327 CHECK WITH OW

    feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

    psth_idx += feedback_idx

    event_time=30
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

    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, gridspec_kw={'height_ratios': [4, 1]}, figsize=(20, 15), sharex=True) 

    create_heatmap(df.raw_isosbestic.values[psth_idx], ax1, 'Raw Isosbestic', event_time)
    create_heatmap(df.raw_calcium.values[psth_idx], ax2, 'Raw Calcium', event_time)
    create_heatmap(df.calcium.values[psth_idx], ax3, 'Calcium', event_time)
    create_heatmap(df.zdFF.values[psth_idx], ax4, 'zdFF', event_time)

    create_line_plot(df.raw_isosbestic.values[psth_idx], ax5, event_time)
    create_line_plot(df.raw_calcium.values[psth_idx], ax6, event_time)
    create_line_plot(df.calcium.values[psth_idx], ax7, event_time)
    create_line_plot(df.zdFF.values[psth_idx], ax8, event_time) 
    plt.suptitle(a+" "+nm+" "+mouse+" "+date+" "+region+" "+nphfile_number+" "+bncfile_number, fontsize=FONTSIZE_1) 
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.tick_params(axis='x', labelsize=FONTSIZE_3)
        ax.tick_params(axis='y', labelsize=FONTSIZE_3)

    plt.tight_layout()
    path = '/home/kceniabougrova/Documents/results/figures'
    path_fig = 'fig1_'+mouse+'_'+date+'_'+region+'_'+a 
    plt.savefig(os.path.join(path, path_fig + '.png')) 
    # plt.savefig(os.path.join(path, path_fig + '.pdf')) #not really working 
    plt.show() 

test_test_test(df_alldata=df_alldata, a="feedback_times", df=df) 


#%%
""" JUST THE LINEPLOT """ 
data = df.calcium.values[psth_idx] 
event_time = 30

average_values = data.mean(axis=1)
plt.plot(average_values, color='black')
plt.xlabel('time since event (s)', fontsize=FONTSIZE_3)
plt.ylabel('zdFF', fontsize=FONTSIZE_3)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=1.5, linestyle="dashed")
plt.grid(False)

plt.plot 




# %%

tph_1 = tph[15:len(tph)] 
""" Join the 3 behav events that are associated to the TTL that is sent, with tph and tbpod """
# Step 1: Create DataFrames for each event type
feedbackType_1 = df_trials[df_trials["feedbackType"]==1]
feedbackType_1=feedbackType_1.reset_index(drop=True) 

df_intervals_0 = pd.DataFrame({'times': df_trials['intervals_0'], 'event_name': 'intervals_0'})
df_intervals_1 = pd.DataFrame({'times': df_trials['intervals_1'], 'event_name': 'intervals_1'}) 
df_feedback_times = pd.DataFrame({'times': feedbackType_1['feedback_times'], 'event_name': 'feedback_times'}) 

# Step 2: Concatenate DataFrames
df_concatenated = pd.concat([df_intervals_0, df_intervals_1, df_feedback_times])

# Step 3: Sort by 'times'
df_concatenated.sort_values(by='times', inplace=True)
df_concatenated["tph"] = tph_1
df_concatenated["tbpod"] = tbpod

# Step 4: Extract times related to intervals_0
times_related_to_intervals_0 = df_concatenated[df_concatenated['event_name'] == 'intervals_0']['times'].values 

tphtimes_related_to_intervals_0 = pd.DataFrame(df_concatenated[df_concatenated['event_name'] == 'intervals_0']['tph'].values, columns=["intervals_0_tph"]) 
df_events = pd.concat([tphtimes_related_to_intervals_0, df_trials], axis=1)

df_events['trial_number'] = df_events.reset_index().index + 1 


table_x = ["intervals_0", "goCue_times", "stimOn_times", "firstMovement_times", "feedback_times", "intervals_1"]

table_y = pd.concat([df_events['choice'], df_events['contrastLeft'], df_events['contrastRight'], df_events['feedbackType'], 
                     df_events['rewardVolume'], df_events['probabilityLeft'], df_events['trial_number']], axis=1) 





onetime_allnontime={} 
for x in table_x: 
    for i in range(0, len(table_x)): 
        onetime_allnontime["{0}".format(x)] = pd.concat([df_events["{0}".format(x)], 
                                table_y], axis=1) #join df_events of each table_x to the entire table_y
        onetime_allnontime["{0}".format(x)]["name"] = "{0}".format(x) #names with "name" the column to which table_x time name it is associated to
        onetime_allnontime["{0}".format(x)] = onetime_allnontime["{0}".format(x)].rename(columns={"{0}".format(x): 'times'}) #renames the new created column with "times"

onetime_allnontime_2=pd.DataFrame(onetime_allnontime["intervals_0"]) #create a df with the data of the previous loop's first time event
for x in table_x[1:len(table_x)]: 
    onetime_allnontime_2 = pd.concat([onetime_allnontime_2,(onetime_allnontime["{0}".format(x)])], ignore_index=True) 
onetime_allnontime_2 = onetime_allnontime_2.reset_index(drop=True) #reset the index
df_events_sorted = onetime_allnontime_2.sort_values(by=['times']) #sort all the rows by the time of the events
# to check what are the nans: 
#test = df_events_sorted[df_events_sorted['times'].isna()]
#test.name.unique()
df_events_sorted = df_events_sorted.dropna(subset=['times']) #drop the nan rows - may be associated to the stimFreeze_times, stimFreezeTrigger_times, errorCueTrigger_times
df_events_sorted = df_events_sorted.reset_index() 
# %%
