

#%%
from one.api import ONE
from pathlib import Path
from generator import Generator, make_data_js
one = one = ONE(cache_dir="/mnt/h0/kb/data/one")
SAVE_PATH = Path("/mnt/h0/kb/viz_figures_tests")
SAVE_PATH.mkdir(exist_ok=True, parents=True)

session = Path('/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-04022/2022-12-30/001')
eid = one.path2eid(session)
g = Generator(eid, one=one, data_path=session, cache_path=SAVE_PATH)
g.make_all_plots(nums=(1,2,3,4))




"""##################################################################################################################################################"""
# %% 
""" 
2024June28 
To load the data and find the peak value, time of the peak value and mad 

"""
#imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from ibldsp.utils import parabolic_max 
from iblphotometry.preprocessing import jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 

PERIEVENT_WINDOW = [-1, 2]
EVENT = "feedback_times"
#load data 
# nph = pd.read_parquet('/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-04022/2022-11-29/001/alf/Region4G/raw_photometry.pqt')
# behav = pd.read_parquet('/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-04022/2022-11-29/001/alf/_ibl_trials.table.pqt') 
nph = pd.read_parquet('/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-05236/2023-07-03/001/alf/Region4G/raw_photometry.pqt')
behav = pd.read_parquet('/mnt/h0/kb/data/one/mainenlab/Subjects/ZFM-05236/2023-07-03/001/alf/_ibl_trials.table.pqt') 


#preprocess data 
fs = 1 / np.median(np.diff(nph.times.values))
nph_j = jove2019(nph.raw_calcium, nph.raw_isosbestic, fs=fs) 
# nph_j = preprocess_sliding_mad(nph.raw_calcium.values, nph.times.values, fs=fs)
nph["calcium"] = nph_j 

plt.figure(figsize=(20, 8))
plt.plot(nph.times, nph.calcium, c='teal', alpha=0.85, linewidth=0.2)
for i in behav.feedback_times: 
    plt.axvline(x=i, linewidth=0.2, color='black', alpha=0.75) 
plt.show() 


#create psth #OPTION1 

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
