
#%%
import pandas as pd 
import matplotlib.pyplot as plt 

from one.api import ONE #always after the imports 
one = ONE()



# ######################""" TO READ THE DATA """#################### 
# # Read the CSV file
# df_csv = pd.read_csv(os.path.join(path, path_nph+'.csv'))
# # Read the Parquet file 

path = '/home/kceniabougrova/Documents/results_for_OW/' 
mouse = 'ZFM-04019' 
date = '2023-03-22'
region_number = '4'
region = f'Region{region_number}G' 

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

eid,df_trials = get_eid(mouse,date)
path_nph = f'demux_nph_{mouse}_{date}_{region_number}_{eid}' 
df_nph = pd.read_parquet(os.path.join(path, path_nph+'.pqt')) 

#%% behavior data division into times vs no-times 
#trialNumber
df_trials['trialNumber'] = range(1, len(df_trials) + 1)

df_trials_times = df_trials[[ 
                            #  'trialNumber', 
                             'intervals_0', 
                             'stimOnTrigger_times', 
                             'stimOn_times', #don't look at this column... 
                             'goCueTrigger_times', 
                             'goCue_times', 
                             'firstMovement_times', 
                             'response_times', 
                             'feedback_times', 
                             'intervals_1'
                            ]] 
df_trials_events = df_trials[[ 
                             'trialNumber', 
                             'contrastLeft', 
                             'contrastRight', 
                             'choice', 
                             'feedbackType',
                             'rewardVolume', 
                             'probabilityLeft', 
                             'quiescencePeriod' 
                             ]] 
#create allContrasts 
idx = 2
new_col = df_trials_events['contrastLeft'].fillna(df_trials_events['contrastRight']) 
df_trials_events.insert(loc=idx, column='allContrasts', value=new_col) 
#create allUContrasts 
df_trials_events['allUContrasts'] = df_trials_events['allContrasts']
df_trials_events.loc[df_trials_events['contrastRight'].isna(), 'allUContrasts'] = df_trials_events['allContrasts'] * -1
df_trials_events.insert(loc=3, column='allUContrasts', value=df_trials_events.pop('allUContrasts'))

#create reactionTime
reactionTime = np.array((df_trials_times["firstMovement_times"])-(df_trials_times["stimOnTrigger_times"]))
df_trials_events["reactionTime"] = reactionTime 





df_all = pd.concat([df_nph,df_trials]).sort_values(by='times') 














































































# %%
""" 21-May-2024 """
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


# ######################""" TO READ THE DATA """#################### 
# # Read the CSV file
# df_csv = pd.read_csv(os.path.join(path, path_nph+'.csv'))
# # Read the Parquet file 
path = '/home/kceniabougrova/Documents/results_for_OW/' 

# mouse = 'ZFM-04019' 
# date = '2023-03-22'
# region_number = '4'
# region = f'Region{region_number}G' 
# # nphfile_number = "2"
# # bncfile_number = '0'

mouse = 'ZFM-04022' 
date = '2022-12-30'
region_number = '4'
region = f'Region{region_number}G' 

mouse = 'ZFM-04022' 
date = '2022-12-30'
region_number = '3'
region = f'Region{region_number}G' 

mouse = 'ZFM-04019' 
date = '2023-01-12'
region_number = '3'
region = f'Region{region_number}G' 





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
prefix = "demux_nph_ZFM-04019_"
################################################
file_list = list_files_in_folder(path, prefix)

for file_name in file_list: 
    mouse = file_name[10:19]
    date = file_name[20:30] 
    region_number = file_name[31:32]
    print(mouse, date, region_number)







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
eid,df_trials = get_eid(mouse,date) 
path_nph = f'demux_nph_{mouse}_{date}_{region_number}_{eid}' 
df_nph = pd.read_parquet(os.path.join(path, path_nph+'.pqt')) 

#def get_lalala(): 
    # source_folder = ("/home/kceniabougrova/Documents/nph/"+date+"/")
    # df_nph = pd.read_csv(source_folder+"raw_photometry"+nphfile_number+".csv") 
    # df_nphttl = pd.read_csv(source_folder+"bonsai_DI"+bncfile_number+nphfile_number+".csv")



    # df_trials['trialNumber'] = range(1, len(df_trials) + 1)

    # df_trials_times = df_trials[[ 
    #                             #  'trialNumber', 
    #                             'intervals_0', 
    #                             'stimOnTrigger_times', 
    #                             'stimOn_times', #don't look at this column... 
    #                             'goCueTrigger_times', 
    #                             'goCue_times', 
    #                             'firstMovement_times', 
    #                             'response_times', 
    #                             'feedback_times', 
    #                             'intervals_1'
    #                             ]] 
    # df_trials_events = df_trials[[ 
    #                             'trialNumber', 
    #                             'contrastLeft', 
    #                             'contrastRight', 
    #                             'choice', 
    #                             'feedbackType',
    #                             'rewardVolume', 
    #                             'probabilityLeft', 
    #                             'quiescencePeriod' 
    #                             ]] 
#create allContrasts 
idx = 2
new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
#create allUContrasts 
df_trials['allUContrasts'] = df_trials['allContrasts']
df_trials.loc[df_trials['contrastRight'].isna(), 'allUContrasts'] = df_trials['allContrasts'] * -1
df_trials.insert(loc=3, column='allUContrasts', value=df_trials.pop('allUContrasts'))

#create reactionTime
reactionTime = np.array((df_trials["firstMovement_times"])-(df_trials["stimOnTrigger_times"]))
df_trials["reactionTime"] = reactionTime 


a = one.load_object(eid, 'trials')
trials = a.to_df()

df_trials = df_trials[0:len(df_trials)-1] #to avoid the last trial not having photometry data 



"""
now I should have df_trials and df_nph
"""
# df = df_nph
# df_alldata = df_trials
# %% 
"""
PLOT THE FIGURES 
""" 

fig, ax = iblphotometry.plots.plot_raw_data_df(df_nph) 

array_timestamps_bpod = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(array_timestamps_bpod, event_test) #check idx where they would be included, in a sorted way 
print(idx_event)

""" create a column with the trial number in the nph df """
df_nph["trial_number"] = 0 #create a new column for the trial_number 
df_nph.loc[idx_event,"trial_number"]=1
df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

# df.to_parquet('/home/kcenia/Desktop/testserver_2023_08_31/photometry20230831.parquet') 

PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR

################################################
""" CHANGE HERE """
EVENT_NAME = "feedback_times"
################################################

sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_trials.shape[0]

psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
# psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials-1))

event_feedback = np.array(df_trials[EVENT_NAME]) #pick the feedback timestamps 
# event_feedback = event_feedback[0:len(event_feedback)-1] #KB added 20240327 CHECK WITH OW

feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

psth_idx += feedback_idx

df_nph.calcium.values[psth_idx] 

sns.heatmap(df_nph.calcium.values[psth_idx].T, cbar=True)
plt.axvline(x=30, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed")

#%%
width_ratios = [1, 1]
FONTSIZE_1 = 30
FONTSIZE_2 = 25
FONTSIZE_3 = 15 

data = df_nph.calcium.values[psth_idx]
event_time = 30 

average_values = data.mean(axis=1)
plt.plot(average_values, color='black')
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
plt.grid(False)



















#%%
""" 
1.a. heatmap for split correct vs incorrect for window = -3s to 5s 
"""
PERIEVENT_WINDOW = [-3,5] #never to be changed!!! "constant" 
SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR
sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_trials.shape[0]

psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
# psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials-1))

event_feedback = np.array(df_trials.feedback_times) #pick the feedback timestamps 
# event_feedback = event_feedback[0:len(event_feedback)-1] #KB added 20240327 CHECK WITH OW

feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

psth_idx += feedback_idx

psth_error = df_nph.calcium.values[psth_idx[:,(df_trials.feedbackType == -1)]]
psth_good = df_nph.calcium.values[psth_idx[:,(df_trials.feedbackType == 1)]]

sns.heatmap(psth_good.T, cbar=True)
plt.axvline(x=90, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed") 
plt.show()

sns.heatmap(psth_error.T, cbar=True)
plt.axvline(x=90, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed") 
plt.show()

#%%
""" 
1.b. heatmap for split correct vs incorrect for window = -1s to 2s 
"""
PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR
sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_trials.shape[0]
psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
event_feedback = np.array(df_trials.feedback_times) #pick the feedback timestamps 
feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 
psth_idx += feedback_idx
psth_error = df_nph.calcium.values[psth_idx[:,(df_trials.feedbackType == -1)]]
psth_good = df_nph.calcium.values[psth_idx[:,(df_trials.feedbackType == 1)]]

#correct 
sns.heatmap(psth_good.T, cbar=True)
plt.axvline(x=30, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed") 
plt.show()
#incorrect
sns.heatmap(psth_error.T, cbar=True)
plt.axvline(x=30, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed") 
plt.show()

#%% 
##############################################################################################
##############################################################################################
##############################################################################################

""" 
1.b. lineplots for split correct vs incorrect for window = -1s to 2s 
"""
plt.rcParams["figure.figsize"] = (8,6)
width_ratios = [1, 1]
FONTSIZE_1 = 16
FONTSIZE_2 = 15
FONTSIZE_3 = 15 
event_time = 30 

psth_50 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1))]]
psth_20 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == 1))]]
psth_80 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == 1))]]

# plt.plot(psth_50, color='#cdb4db', linewidth=1, alpha=0.2) 
# plt.plot(psth_20, color='#ffafcc', linewidth=1, alpha=0.2) 
# plt.plot(psth_80, color='#a2d2ff', linewidth=1, alpha=0.2) 
average_values_50 = psth_50.mean(axis=1) 
average_values_20 = psth_20.mean(axis=1) 
average_values_80 = psth_80.mean(axis=1) 
plt.plot(average_values_50, color='#cdb4db', linewidth=3) 
plt.plot(average_values_20, color='#ffafcc', linewidth=3) 
plt.plot(average_values_80, color='#a2d2ff', linewidth=3) 
plt.title("NM response at feedback outcome = correct, for the 3 blocks", fontsize=FONTSIZE_1)
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
plt.legend(("0.5","0.2","0.8"), fontsize=FONTSIZE_3, frameon=False) 

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
plt.show() 


psth_50 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == -1))]]
psth_20 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == -1))]]
psth_80 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == -1))]] 
average_values_50 = psth_50.mean(axis=1) 
average_values_20 = psth_20.mean(axis=1) 
average_values_80 = psth_80.mean(axis=1) 
plt.plot(average_values_50, color='#cdb4db', linewidth=3) 
plt.plot(average_values_20, color='#ffafcc', linewidth=3) 
plt.plot(average_values_80, color='#a2d2ff', linewidth=3) 
plt.title("NM response at feedback outcome = incorrect, for the 3 blocks", fontsize=FONTSIZE_1)
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
plt.legend(("0.5","0.2","0.8"), fontsize=FONTSIZE_3, frameon=False) 
plt.grid(False) 
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color("black")
ax.spines['left'].set_color("black") 
plt.show()
############################################################################################## 
##############################################################################################
##############################################################################################











#%% 
#####################################################################################
plt.rcParams["figure.figsize"] = (8,6)
width_ratios = [1, 1]
FONTSIZE_1 = 16
FONTSIZE_2 = 15
FONTSIZE_3 = 15 

event_time = 30 

psth_50 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 1))]]
psth_20 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0))]]
psth_80 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.1250))]]
psth_10 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.0625))]]


average_values_50 = psth_50.mean(axis=1) 
average_values_20 = psth_20.mean(axis=1) 
average_values_80 = psth_80.mean(axis=1) 
average_values_10 = psth_10.mean(axis=1) 
plt.plot(average_values_50, color='#220901', linewidth=2.5, label="1") 
plt.plot(average_values_20, color='#f6aa1c', linewidth=2.5) 
plt.plot(average_values_80, color='#941b0c', linewidth=2.5) 
plt.plot(average_values_10, color='#bc3908', linewidth=2.5) 
plt.title("NM response at feedback outcome = correct, cR = 1, for the 3 blocks", fontsize=FONTSIZE_1)
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
plt.legend(("1","0","0.1250","0.0625"), fontsize=FONTSIZE_3, frameon=False) 
plt.grid(False)
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color("black")
ax.spines['left'].set_color("black") 
plt.show()

#####################################################################################






#%%
#####################################################################################
plt.rcParams["figure.figsize"] = (14,8)
plt.figure(dpi=300)
width_ratios = [1, 1]
FONTSIZE_1 = 16
FONTSIZE_2 = 15
FONTSIZE_3 = 15 
event_time = 30 

psth_100 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 1))]]
psth_25 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.25))]] 
psth_12 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.1250))]]
psth_06 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.0625))]]
psth_00 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0))]]

average_values_100 = psth_100.mean(axis=1) 
average_values_25 = psth_25.mean(axis=1) 
average_values_12 = psth_12.mean(axis=1) 
average_values_06 = psth_06.mean(axis=1) 
average_values_00 = psth_00.mean(axis=1) 

plt.plot(average_values_100, color='#370617', linewidth=2.5) 
plt.plot(average_values_25, color='#d00000', linewidth=2.5) 
plt.plot(average_values_12, color='#e85d04', linewidth=2.5) 
plt.plot(average_values_06, color='#ffba08', linewidth=2.5) 
plt.plot(average_values_00, color='#d8e2dc', linewidth=2.5) 

plt.title("NM response at feedback outcome = correct, block = 0.5", fontsize=FONTSIZE_1)
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
plt.legend(("1","0.25","0.1250","0.0625","0")) 
plt.grid(False) 
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color("black")
ax.spines['left'].set_color("black") 
plt.show()
#####################################################################################


# %% 
#####################################################################################
plt.rcParams["figure.figsize"] = (14,8)
plt.figure(dpi=300)
width_ratios = [1, 1]
FONTSIZE_1 = 16
FONTSIZE_2 = 15
FONTSIZE_3 = 15 
event_time = 30 

# psth_50 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == -1))]]
# psth_20 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == -1))]]
# psth_80 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == -1))]]
psth_50 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5))]]
psth_20 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2))]]
psth_80 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8))]]

average_values_50 = psth_50.mean(axis=1) 
average_values_20 = psth_20.mean(axis=1) 
average_values_80 = psth_80.mean(axis=1) 

plt.plot(average_values_50, color='#cdb4db', linewidth=3) 
plt.plot(average_values_20, color='#ffafcc', linewidth=3) 
plt.plot(average_values_80, color='#a2d2ff', linewidth=3) 

plt.title("NM response at feedback outcome = correct, block = 0.5", fontsize=FONTSIZE_1)
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
plt.legend(("0.5","0.2","0.8")) 
plt.grid(False)
plt.show() 
#####################################################################################


#%% 
#################################### REACTION TIMES #################################################
plt.rcParams["figure.figsize"] = (14,10)
plt.figure(dpi=300)
width_ratios = [1, 1]
FONTSIZE_1 = 25
FONTSIZE_2 = 21
FONTSIZE_3 = 18 
event_time = 30 

# psth_50 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == -1))]]
# psth_20 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == -1))]]
# psth_80 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == -1))]]
psth_fast = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime <= 0.85) & (df_trials.feedbackType == 1))]]
psth_slow = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 1.75) & (df_trials.feedbackType == 1))]]
psth_medium = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime > 0.85) & (df_trials.reactionTime < 1.75) & (df_trials.feedbackType == 1))]]

average_values_fast = psth_fast.mean(axis=1) 
average_values_slow = psth_slow.mean(axis=1) 
average_values_medium = psth_medium.mean(axis=1) 

sem_fast = psth_fast.std(axis=1) / np.sqrt(psth_fast.shape[1])
sem_slow = psth_slow.std(axis=1) / np.sqrt(psth_slow.shape[1]) 
sem_medium = psth_medium.std(axis=1) / np.sqrt(psth_medium.shape[1])

# plt.plot(psth_fast, color='#2a9d8f', linewidth=0.5, alpha=0.2) 
# plt.plot(psth_slow, color='#fb8500', linewidth=0.5, alpha=0.2) 

plt.plot(average_values_fast, color='#d90429', linewidth=3, alpha=0.5, label='fast RT '+str(psth_fast.shape[1])) 
plt.fill_between(range(len(average_values_fast)), average_values_fast - sem_fast, average_values_fast + sem_fast, color='#d90429', alpha=0.15)
plt.plot(average_values_slow, color='#2a9d8f', linewidth=3, alpha=0.5, label='slow RT '+str(psth_slow.shape[1]))
plt.fill_between(range(len(average_values_slow)), average_values_slow - sem_slow, average_values_slow + sem_slow, color='#2a9d8f', alpha=0.15)
plt.plot(average_values_medium, color='#DBD9DC', linewidth=3, alpha=0.5, label='medium RT '+str(psth_medium.shape[1]))
plt.fill_between(range(len(average_values_medium)), average_values_medium - sem_medium, average_values_medium + sem_medium, color='#DBD9DC', alpha=0.15)
plt.suptitle("NM response at "+EVENT_NAME+" for different reaction times", fontsize=FONTSIZE_1)
plt.title("feedback outcome = 1, reactionTime 0.85 1.75", fontsize=FONTSIZE_2, pad=20)
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
#############################################################
##### -0.5 to 0.5s of 4 divisions of the reaction times #####
plt.rcParams["figure.figsize"] = (14,10)
plt.figure(dpi=300)
width_ratios = [1, 1]
FONTSIZE_1 = 25
FONTSIZE_2 = 21
FONTSIZE_3 = 18 

event_time = 30 

psth_fastest0 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime < 0.25) & (df_trials.feedbackType == 1))]]
psth_fastest = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.25) & (df_trials.reactionTime < 0.3) & (df_trials.feedbackType == 1))]]
psth_fast_m = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1))]]
psth_fast_s = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.65) & (df_trials.reactionTime < 1.2) & (df_trials.feedbackType == 1))]]

average_values_fast0 = psth_fastest0.mean(axis=1) 
average_values_fast = psth_fastest.mean(axis=1) 
average_values_slow = psth_fast_m.mean(axis=1) 
average_values_medium = psth_fast_s.mean(axis=1) 

sem_fast0 = psth_fastest0.std(axis=1) / np.sqrt(psth_fastest0.shape[1])
sem_fast = psth_fastest.std(axis=1) / np.sqrt(psth_fastest.shape[1])
sem_slow = psth_fast_m.std(axis=1) / np.sqrt(psth_fast_m.shape[1]) 
sem_medium = psth_fast_s.std(axis=1) / np.sqrt(psth_fast_s.shape[1])

# plt.plot(psth_fast, color='#2a9d8f', linewidth=0.5, alpha=0.2) 
# plt.plot(psth_slow, color='#fb8500', linewidth=0.5, alpha=0.2) 

plt.plot(average_values_fast0, color='#230c33', linewidth=5, alpha=0.95, label='<0.25 RT '+str(psth_fastest0.shape[1])) 

plt.plot(average_values_fast, color='#592e83', linewidth=5, alpha=0.8, label='0.25-0.3 RT '+str(psth_fastest.shape[1])) 
# plt.fill_between(range(len(average_values_fast)), average_values_fast - sem_fast, average_values_fast + sem_fast, color='#e76f51', alpha=0.15)
plt.plot(average_values_slow, color='#9984d4', linewidth=5, alpha=0.7, label='0.3-0.5 RT '+str(psth_fast_m.shape[1]))
# plt.fill_between(range(len(average_values_slow)), average_values_slow - sem_slow, average_values_slow + sem_slow, color='#e9c46a', alpha=0.15)
plt.plot(average_values_medium, color='#caa8f5', linewidth=5, alpha=0.6, label='0.65-1.2 RT '+str(psth_fast_s.shape[1]))
# plt.fill_between(range(len(average_values_medium)), average_values_medium - sem_medium, average_values_medium + sem_medium, color='#2a9d8f', alpha=0.15)
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
plt.xlim(20, 45)
plt.show()

#%%
####################################################
##### Barplot of the density of reaction times #####
plt.grid(False)
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]) 

for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)

ax.spines['bottom'].set_color("black")
ax.spines['left'].set_color("black") 
plt.suptitle("Reaction time density", fontsize=FONTSIZE_1)
plt.title("feedback outcome = 1", fontsize=FONTSIZE_2, pad=20)
plt.xlabel('reaction times', fontsize=FONTSIZE_2)
plt.xticks(fontsize=FONTSIZE_3)
plt.yticks(fontsize=FONTSIZE_3) 
plt.legend(fontsize=FONTSIZE_3, frameon=False) 
plt.hist(df_trials.reactionTime[df_trials.reactionTime <=1.2], bins=50, color="#9984d4") 
plt.axvline(x=0.3, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
plt.axvline(x=0.5, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
plt.show()



#%% 
##### reaction times based on contrasts #####
plt.rcParams["figure.figsize"] = (14,10)
plt.figure(dpi=300)
width_ratios = [1, 1]
FONTSIZE_1 = 25
FONTSIZE_2 = 21
FONTSIZE_3 = 18 

event_time = 30 

psth_fastest_100 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.25) & (df_trials.reactionTime < 0.3) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 1))]]
psth_fastest_25 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.25) & (df_trials.reactionTime < 0.3) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.25))]]
psth_fastest_12 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.25) & (df_trials.reactionTime < 0.3) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.125))]]
psth_fastest_6 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.25) & (df_trials.reactionTime < 0.3) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.0625))]]
psth_fastest_0 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.25) & (df_trials.reactionTime < 0.3) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0))]]
# psth_fastest_100 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 1))]]
# psth_fastest_25 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.25))]]
# psth_fastest_12 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.125))]]
# psth_fastest_6 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.0625))]]
# psth_fastest_0 = df_nph.calcium.values[psth_idx[:,((df_trials.reactionTime >= 0.3) & (df_trials.reactionTime < 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0))]]


average_values_fast_100 = psth_fastest_100.mean(axis=1) 
average_values_fast_25 = psth_fastest_25.mean(axis=1) 
average_values_fast_12 = psth_fastest_12.mean(axis=1) 
average_values_fast_6 = psth_fastest_6.mean(axis=1) 
average_values_fast_0 = psth_fastest_0.mean(axis=1) 

sem_fast_100 = psth_fastest_100.std(axis=1) / np.sqrt(psth_fastest_100.shape[1])
sem_fast_25 = psth_fastest_25.std(axis=1) / np.sqrt(psth_fastest_25.shape[1]) 
sem_fast_12 = psth_fastest_12.std(axis=1) / np.sqrt(psth_fastest_12.shape[1])
sem_fast_6 = psth_fastest_6.std(axis=1) / np.sqrt(psth_fastest_6.shape[1]) 
sem_fast_0 = psth_fastest_0.std(axis=1) / np.sqrt(psth_fastest_0.shape[1])

# plt.plot(psth_fast, color='#2a9d8f', linewidth=0.5, alpha=0.2) 
# plt.plot(psth_slow, color='#fb8500', linewidth=0.5, alpha=0.2) 


plt.plot(average_values_fast_100, color='#4273b3', linewidth=5, alpha=0.8, label='0.25-0.3 100 '+str(psth_fastest_100.shape[1])) 
plt.plot(average_values_fast_25, color='#89d0a4', linewidth=5, alpha=0.8, label='0.25-0.3 25 '+str(psth_fastest_25.shape[1]))
plt.plot(average_values_fast_12, color='#feeb9d', linewidth=5, alpha=0.8, label='0.25-0.3 12 '+str(psth_fastest_12.shape[1]))
plt.plot(average_values_fast_6, color='#ef6645', linewidth=5, alpha=0.8, label='0.25-0.3 6 '+str(psth_fastest_6.shape[1]))
plt.plot(average_values_fast_0, color='#be254a', linewidth=5, alpha=0.8, label='0.25-0.3 0 '+str(psth_fastest_0.shape[1]))

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
plt.ylim(-0.005, 0.0125) 
# plt.xlim(20, 45)
plt.show() 












#%%
########################
""" many subplots """

#%%
# Concatenate all average values and filter out NaNs and Infs
all_avg_values = np.concatenate(list(avg_vars.values()))
all_avg_values = all_avg_values[np.isfinite(all_avg_values)]

# Determine the common ylim based on the global minimum and maximum values
ylim_min = all_avg_values.min()
ylim_max = all_avg_values.max()

# Plotting
plt.rcParams["figure.figsize"] = (20, 15)
FONTSIZE_1 = 16
FONTSIZE_2 = 15
FONTSIZE_3 = 15
event_time = 30

fig, axes = plt.subplots(6, 5, figsize=(20, 15))

# Loop through each subplot and plot the corresponding average values
for i, (key, avg) in enumerate(avg_vars.items()):
    ax = axes.flatten()[i]
    ax.plot(avg, linewidth=2.5, label=key)
    ax.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
    ax.legend(fontsize=FONTSIZE_3, frameon=False)
    ax.set_title(f"{key}", fontsize=FONTSIZE_1)
    ax.set_xlabel('time since event (s)', fontsize=FONTSIZE_2)
    ax.set_ylabel('zdFF', fontsize=FONTSIZE_2)
    ax.set_ylim(ylim_min, ylim_max)
    ax.grid(False)
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color("black")
    ax.spines['left'].set_color("black")

# Turn off unused subplots
for j in range(i + 1, len(axes.flatten())):
    fig.delaxes(axes.flatten()[j])

plt.tight_layout()
plt.show() 




#%%
plt.rcParams["figure.figsize"] = (8,6)
width_ratios = [1, 1]
FONTSIZE_1 = 16
FONTSIZE_2 = 15
FONTSIZE_3 = 15 

event_time = 30 


psth_50_1_100 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 1))]] 
psth_50_1_25 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.25))]]
psth_50_1_12 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.1250))]]
psth_50_1_06 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.0625))]]
psth_50_1_00 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0))]] 
psth_50_0_100 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 1))]] 
psth_50_0_25 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.25))]]
psth_50_0_12 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.1250))]]
psth_50_0_06 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.0625))]]
psth_50_0_00 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 0))]] 

psth_20_1_100 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 1))]] 
psth_20_1_25 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.25))]]
psth_20_1_12 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.1250))]]
psth_20_1_06 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.0625))]]
psth_20_1_00 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0))]] 
psth_20_0_100 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 1))]] 
psth_20_0_25 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.25))]]
psth_20_0_12 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.1250))]]
psth_20_0_06 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.0625))]]
psth_20_0_00 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 0))]] 

psth_80_1_100 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 1))]] 
psth_80_1_25 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.25))]]
psth_80_1_12 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.1250))]]
psth_80_1_06 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.0625))]]
psth_80_1_00 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == 1) & (df_trials.allContrasts == 0))]] 
psth_80_0_100 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 1))]] 
psth_80_0_25 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.25))]]
psth_80_0_12 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.1250))]]
psth_80_0_06 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.0625))]]
psth_80_0_00 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == -1) & (df_trials.allContrasts == 0))]] 


conditions = [
    ('50', 0.5, 1, 1), ('50', 0.5, 1, 0.25), ('50', 0.5, 1, 0.1250), ('50', 0.5, 1, 0.0625), ('50', 0.5, 1, 0),
    ('50', 0.5, -1, 1), ('50', 0.5, -1, 0.25), ('50', 0.5, -1, 0.1250), ('50', 0.5, -1, 0.0625), ('50', 0.5, -1, 0),
    ('20', 0.2, 1, 1), ('20', 0.2, 1, 0.25), ('20', 0.2, 1, 0.1250), ('20', 0.2, 1, 0.0625), ('20', 0.2, 1, 0),
    ('20', 0.2, -1, 1), ('20', 0.2, -1, 0.25), ('20', 0.2, -1, 0.1250), ('20', 0.2, -1, 0.0625), ('20', 0.2, -1, 0),
    ('80', 0.8, 1, 1), ('80', 0.8, 1, 0.25), ('80', 0.8, 1, 0.1250), ('80', 0.8, 1, 0.0625), ('80', 0.8, 1, 0),
    ('80', 0.8, -1, 1), ('80', 0.8, -1, 0.25), ('80', 0.8, -1, 0.1250), ('80', 0.8, -1, 0.0625), ('80', 0.8, -1, 0)
]

# Initialize dictionaries to store PSTH variables and their averages
psth_vars = {}
avg_vars = {}

# Populate the dictionaries
for prefix, prob, feedback, contrast in conditions:
    var_name = f'psth_{prefix}_{feedback}_{str(contrast).replace(".", "")}'
    psth_vars[var_name] = df_nph.calcium.values[psth_idx[:, ((df_trials.probabilityLeft == prob) & (df_trials.feedbackType == feedback) & (df_trials.allContrasts == contrast))]]
    avg_vars[f'avg_{prefix}_{feedback}_{str(contrast).replace(".", "")}'] = psth_vars[var_name].mean(axis=1)# Define the conditions






# Plotting
plt.rcParams["figure.figsize"] = (20, 15)
FONTSIZE_1 = 16
FONTSIZE_2 = 15
FONTSIZE_3 = 15
event_time = 30

fig, axes = plt.subplots(3, 5, figsize=(20, 15))

# Loop through each subplot and plot the corresponding average values
for i, (key, avg) in enumerate(avg_vars.items()):
    ax = axes.flatten()[i]
    ax.plot(avg, linewidth=2.5, label=key)
    ax.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
    ax.legend(fontsize=FONTSIZE_3, frameon=False)
    ax.set_title(f"{key}", fontsize=FONTSIZE_1)
    ax.set_xlabel('time since event (s)', fontsize=FONTSIZE_2)
    ax.set_ylabel('zdFF', fontsize=FONTSIZE_2)
    ax.grid(False)
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color("black")
    ax.spines['left'].set_color("black")

# Turn off unused subplots
for j in range(i + 1, len(axes.flatten())):
    fig.delaxes(axes.flatten()[j])

plt.tight_layout()
plt.show()
# %% 

# Your conditions and data setup
conditions = [
    ('50', 0.5, 1, 1), ('50', 0.5, 1, 0.25), ('50', 0.5, 1, 0.1250), ('50', 0.5, 1, 0.0625), ('50', 0.5, 1, 0),
    ('50', 0.5, -1, 1), ('50', 0.5, -1, 0.25), ('50', 0.5, -1, 0.1250), ('50', 0.5, -1, 0.0625), ('50', 0.5, -1, 0),
    ('20', 0.2, 1, 1), ('20', 0.2, 1, 0.25), ('20', 0.2, 1, 0.1250), ('20', 0.2, 1, 0.0625), ('20', 0.2, 1, 0),
    ('20', 0.2, -1, 1), ('20', 0.2, -1, 0.25), ('20', 0.2, -1, 0.1250), ('20', 0.2, -1, 0.0625), ('20', 0.2, -1, 0),
    ('80', 0.8, 1, 1), ('80', 0.8, 1, 0.25), ('80', 0.8, 1, 0.1250), ('80', 0.8, 1, 0.0625), ('80', 0.8, 1, 0),
    ('80', 0.8, -1, 1), ('80', 0.8, -1, 0.25), ('80', 0.8, -1, 0.1250), ('80', 0.8, -1, 0.0625), ('80', 0.8, -1, 0)
]

# Initialize dictionaries to store PSTH variables and their averages
psth_vars = {}
avg_vars = {}

# Populate the dictionaries
for prefix, prob, feedback, contrast in conditions:
    var_name = f'psth_{prefix}_{feedback}_{str(contrast).replace(".", "")}'
    psth_vars[var_name] = df_nph.calcium.values[psth_idx[:, ((df_trials.probabilityLeft == prob) & (df_trials.feedbackType == feedback) & (df_trials.allContrasts == contrast))]]
    avg_vars[f'avg_{prefix}_{feedback}_{str(contrast).replace(".", "")}'] = psth_vars[var_name].mean(axis=1)

# Plotting
plt.rcParams["figure.figsize"] = (20, 15)
FONTSIZE_1 = 16
FONTSIZE_2 = 15
FONTSIZE_3 = 15
event_time = 30

fig, axes = plt.subplots(3, 5, figsize=(20, 15))

# Define prefixes and contrast values
prefixes = ['50', '20', '80']
contrasts = [1, 0.25, 0.1250, 0.0625, 0]

# Plot the data
for i, prefix in enumerate(prefixes):
    for j, contrast in enumerate(contrasts[:-1]):
        ax = axes[i, j]
        for key, avg in avg_vars.items():
            if key.startswith(f'avg_{prefix}_1_{str(contrast).replace(".", "")}'):
                ax.plot(avg, 'b', linewidth=2.5, label='feedback 1' if j == 0 else "")
            elif key.startswith(f'avg_{prefix}_-1_{str(contrast).replace(".", "")}'):
                ax.plot(avg, 'r', linewidth=2.5, label='feedback -1' if j == 0 else "")
        
        ax.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
        if i == 0:
            ax.set_title(f"Contrast {contrast}", fontsize=FONTSIZE_1)
        if j == 0:
            ax.set_ylabel(f"{prefix}", fontsize=FONTSIZE_2)
        if i == len(prefixes) - 1:
            ax.set_xlabel('time since event (s)', fontsize=FONTSIZE_2)
        
        ax.grid(False)
        for spine in ['right', 'top']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color("black")
        ax.spines['left'].set_color("black")

# Plot average for the condition (XX_X_0) in the last column
for i, prefix in enumerate(prefixes):
    ax = axes[i, 4]
    key_1 = f'avg_{prefix}_1_0'
    key_neg1 = f'avg_{prefix}_-1_0'
    if key_1 in avg_vars:
        ax.plot(avg_vars[key_1], 'b', linewidth=2.5, label='feedback 1' if i == 0 else "")
    if key_neg1 in avg_vars:
        ax.plot(avg_vars[key_neg1], 'r', linewidth=2.5, label='feedback -1' if i == 0 else "")
    
    ax.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
    if i == 0:
        ax.set_title("Contrast 0", fontsize=FONTSIZE_1)
    if j == 0:
        ax.set_ylabel(f"{prefix}", fontsize=FONTSIZE_2)
    if i == len(prefixes) - 1:
        ax.set_xlabel('time since event (s)', fontsize=FONTSIZE_2)
    
    ax.grid(False)
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color("black")
    ax.spines['left'].set_color("black")

# Add a single legend for the entire figure
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize=FONTSIZE_3, frameon=False)

plt.tight_layout() 
plt.show()

#%%
########################################################################################
######################################################################################## 
########################################################################################
""" all subplots with sem """ 
# Your conditions and data setup
conditions = [
    ('50', 0.5, 1, 1), ('50', 0.5, 1, 0.25), ('50', 0.5, 1, 0.1250), ('50', 0.5, 1, 0.0625), ('50', 0.5, 1, 0),
    ('50', 0.5, -1, 1), ('50', 0.5, -1, 0.25), ('50', 0.5, -1, 0.1250), ('50', 0.5, -1, 0.0625), ('50', 0.5, -1, 0),
    ('20', 0.2, 1, 1), ('20', 0.2, 1, 0.25), ('20', 0.2, 1, 0.1250), ('20', 0.2, 1, 0.0625), ('20', 0.2, 1, 0),
    ('20', 0.2, -1, 1), ('20', 0.2, -1, 0.25), ('20', 0.2, -1, 0.1250), ('20', 0.2, -1, 0.0625), ('20', 0.2, -1, 0),
    ('80', 0.8, 1, 1), ('80', 0.8, 1, 0.25), ('80', 0.8, 1, 0.1250), ('80', 0.8, 1, 0.0625), ('80', 0.8, 1, 0),
    ('80', 0.8, -1, 1), ('80', 0.8, -1, 0.25), ('80', 0.8, -1, 0.1250), ('80', 0.8, -1, 0.0625), ('80', 0.8, -1, 0)
]

# Initialize dictionaries to store PSTH variables and their averages and standard errors
psth_vars = {}
avg_vars = {}
sem_vars = {}

# Populate the dictionaries
for prefix, prob, feedback, contrast in conditions:
    var_name = f'psth_{prefix}_{feedback}_{str(contrast).replace(".", "")}'
    psth_data = df_nph.calcium.values[psth_idx[:, ((df_trials.probabilityLeft == prob) & (df_trials.feedbackType == feedback) & (df_trials.allContrasts == contrast))]]
    psth_vars[var_name] = psth_data
    avg_vars[f'avg_{prefix}_{feedback}_{str(contrast).replace(".", "")}'] = psth_data.mean(axis=1)
    sem_vars[f'sem_{prefix}_{feedback}_{str(contrast).replace(".", "")}'] = psth_data.std(axis=1) / np.sqrt(psth_data.shape[1])

# Plotting
plt.rcParams["figure.figsize"] = (25, 15)
FONTSIZE_1 = 31
FONTSIZE_2 = 21
FONTSIZE_3 = 21
event_time = 30

fig, axes = plt.subplots(3, 5, figsize=(25, 12))

# Define prefixes and contrast values
prefixes = ['50', '20', '80']
contrasts = [1, 0.25, 0.1250, 0.0625, 0]

# Plot the data
for i, prefix in enumerate(prefixes):
    for j, contrast in enumerate(contrasts[:-1]):
        ax = axes[i, j]
        for key, avg in avg_vars.items():
            sem_key = key.replace('avg_', 'sem_')
            if key.startswith(f'avg_{prefix}_1_{str(contrast).replace(".", "")}'):
                ax.plot(avg, '#1789fc', linewidth=2.5, label='feedback 1' if j == 0 else "")
                ax.fill_between(range(len(avg)), avg - sem_vars[sem_key], avg + sem_vars[sem_key], color='#1789fc', alpha=0.15)
            elif key.startswith(f'avg_{prefix}_-1_{str(contrast).replace(".", "")}'):
                ax.plot(avg, '#d63230', linewidth=2.5, label='feedback -1' if j == 0 else "")
                ax.fill_between(range(len(avg)), avg - sem_vars[sem_key], avg + sem_vars[sem_key], color='#d63230', alpha=0.15)
        
        ax.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
        if i == 0:
            ax.set_title(f"Contrast {contrast}", fontsize=FONTSIZE_2)
        if j == 0:
            ax.set_ylabel(f"{prefix}", fontsize=FONTSIZE_2)
        if i == len(prefixes) - 1:
            ax.set_xlabel('time since event (s)', fontsize=FONTSIZE_2)
        
        ax.grid(False)
        for spine in ['right', 'top']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color("black")
        ax.spines['left'].set_color("black")

# Plot average for the condition (XX_X_0) in the last column
for i, prefix in enumerate(prefixes):
    ax = axes[i, 4]
    key_1 = f'avg_{prefix}_1_0'
    key_neg1 = f'avg_{prefix}_-1_0'
    sem_key_1 = key_1.replace('avg_', 'sem_')
    sem_key_neg1 = key_neg1.replace('avg_', 'sem_')
    if key_1 in avg_vars:
        avg = avg_vars[key_1]
        ax.plot(avg, '#1789fc', linewidth=2.5, label='feedback 1' if i == 0 else "")
        ax.fill_between(range(len(avg)), avg - sem_vars[sem_key_1], avg + sem_vars[sem_key_1], color='#1789fc', alpha=0.15)
    if key_neg1 in avg_vars:
        avg = avg_vars[key_neg1]
        ax.plot(avg, '#d63230', linewidth=2.5, label='feedback -1' if i == 0 else "")
        ax.fill_between(range(len(avg)), avg - sem_vars[sem_key_neg1], avg + sem_vars[sem_key_neg1], color='#d63230', alpha=0.15)
    
    ax.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
    if i == 0:
        ax.set_title("Contrast 0", fontsize=FONTSIZE_2)
    if j == 0:
        ax.set_ylabel(f"{prefix}", fontsize=FONTSIZE_2)
    if i == len(prefixes) - 1:
        ax.set_xlabel('time since event (s)', fontsize=FONTSIZE_2)
    
    # ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_3)  # Set tick label font size

    ax.grid(False)
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color("black")
    ax.spines['left'].set_color("black")
plt.suptitle(f"NM activity around {EVENT_NAME} for different bias blocks and contrasts\n"+f"{mouse}_{date}_{region_number}_{eid}", fontsize=FONTSIZE_1)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', fontsize=FONTSIZE_2, frameon=False)

plt.tight_layout()
fig.savefig(f'/home/kceniabougrova/Documents/figures_forlabmeeting_May2024/Fig00_{mouse}_{date}_{region_number}_{EVENT_NAME}.png')
plt.show()
########################################################################################
########################################################################################

# %%
"""USED SAVED CORRECT INCORRECT""" 
""" 
change only prefix and EVENT_NAME 
"""

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
import os
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
prefix = "demux_nph_ZFM-06268_"
################################################
file_list = list_files_in_folder(path, prefix)

for file_name in file_list: 
    mouse = file_name[10:19]
    date = file_name[20:30] 
    region_number = file_name[31:32]
    print(mouse, date, region_number)
    ################################################
    """ CHANGE HERE """
    EVENT_NAME = "feedback_times"
    ################################################

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


    eid,df_trials = get_eid(mouse,date) 
    path_nph = f'demux_nph_{mouse}_{date}_{region_number}_{eid}' 
    df_nph = pd.read_parquet(os.path.join(path, path_nph+'.pqt')) 

    idx = 2
    new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
    df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
    #create allUContrasts 
    df_trials['allUContrasts'] = df_trials['allContrasts']
    df_trials.loc[df_trials['contrastRight'].isna(), 'allUContrasts'] = df_trials['allContrasts'] * -1
    df_trials.insert(loc=3, column='allUContrasts', value=df_trials.pop('allUContrasts'))

    #create reactionTime
    # reactionTime = np.array((df_trials["firstMovement_times"])-(df_trials["stimOnTrigger_times"]))
    # df_trials["reactionTime"] = reactionTime 


    a = one.load_object(eid, 'trials')
    trials = a.to_df()

    df_trials = df_trials[0:len(df_trials)-1] #to avoid the last trial not having photometry data 

    fig, ax = iblphotometry.plots.plot_raw_data_df(df_nph) 

    array_timestamps_bpod = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
    event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
    idx_event = np.searchsorted(array_timestamps_bpod, event_test) #check idx where they would be included, in a sorted way 
    print(idx_event)

    """ create a column with the trial number in the nph df """
    df_nph["trial_number"] = 0 #create a new column for the trial_number 
    df_nph.loc[idx_event,"trial_number"]=1
    df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

    # df.to_parquet('/home/kcenia/Desktop/testserver_2023_08_31/photometry20230831.parquet') 

    PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
    SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR

    sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
    n_trials = df_trials.shape[0]

    psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
    # psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials-1))

    event_feedback = np.array(df_trials[EVENT_NAME]) #pick the feedback timestamps 
    # event_feedback = event_feedback[0:len(event_feedback)-1] #KB added 20240327 CHECK WITH OW

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
    plt.savefig(f'/home/kceniabougrova/Documents/figures_forlabmeeting_May2024/Fig01_{mouse}_{date}_{region_number}_{EVENT_NAME}.png')

    plt.show() 
# %% 
##################################################
""" FOR DIFFERENT CONTRASTS USED SAVED USE """ 

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
import os
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
prefix = "demux_nph_ZFM-06305_"
################################################
file_list = list_files_in_folder(path, prefix)
for file_name in file_list: 
    mouse = file_name[10:19]
    date = file_name[20:30] 
    region_number = file_name[31:32]
    print(mouse, date, region_number)
    ################################################
    """ CHANGE HERE """
    EVENT_NAME = "feedback_times"
    ################################################

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


    eid,df_trials = get_eid(mouse,date) 
    path_nph = f'demux_nph_{mouse}_{date}_{region_number}_{eid}' 
    df_nph = pd.read_parquet(os.path.join(path, path_nph+'.pqt')) 

    idx = 2
    new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
    df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
    #create allUContrasts 
    df_trials['allUContrasts'] = df_trials['allContrasts']
    df_trials.loc[df_trials['contrastRight'].isna(), 'allUContrasts'] = df_trials['allContrasts'] * -1
    df_trials.insert(loc=3, column='allUContrasts', value=df_trials.pop('allUContrasts'))

    #create reactionTime
    # reactionTime = np.array((df_trials["firstMovement_times"])-(df_trials["stimOnTrigger_times"]))
    # df_trials["reactionTime"] = reactionTime 


    a = one.load_object(eid, 'trials')
    trials = a.to_df()

    df_trials = df_trials[0:len(df_trials)-1] #to avoid the last trial not having photometry data 

    fig, ax = iblphotometry.plots.plot_raw_data_df(df_nph) 

    array_timestamps_bpod = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
    event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
    idx_event = np.searchsorted(array_timestamps_bpod, event_test) #check idx where they would be included, in a sorted way 
    print(idx_event)

    """ create a column with the trial number in the nph df """
    df_nph["trial_number"] = 0 #create a new column for the trial_number 
    df_nph.loc[idx_event,"trial_number"]=1
    df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

    # df.to_parquet('/home/kcenia/Desktop/testserver_2023_08_31/photometry20230831.parquet') 

    PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
    SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR

    sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
    n_trials = df_trials.shape[0]

    psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
    # psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials-1))

    event_feedback = np.array(df_trials[EVENT_NAME]) #pick the feedback timestamps 
    # event_feedback = event_feedback[0:len(event_feedback)-1] #KB added 20240327 CHECK WITH OW

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
    
    psth_1_100 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == 1) & (df_trials.allContrasts == 1))]] 
    psth_1_25 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.25))]]
    psth_1_12 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.1250))]]
    psth_1_06 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.0625))]]
    psth_1_00 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == 1) & (df_trials.allContrasts == 0))]] 
    psth_0_100 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == -1) & (df_trials.allContrasts == 1))]] 
    psth_0_25 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.25))]]
    psth_0_12 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.1250))]]
    psth_0_06 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.0625))]]
    psth_0_00 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == -1) & (df_trials.allContrasts == 0))]] 

    sem_1_100 = psth_1_100.std(axis=1) / np.sqrt(psth_1_100.shape[1])
    sem_1_25 = psth_1_25.std(axis=1) / np.sqrt(psth_1_25.shape[1])
    sem_1_12 = psth_1_12.std(axis=1) / np.sqrt(psth_1_12.shape[1])
    sem_1_06 = psth_1_06.std(axis=1) / np.sqrt(psth_1_06.shape[1])
    sem_1_00 = psth_1_00.std(axis=1) / np.sqrt(psth_1_00.shape[1])
    sem_0_100 = psth_0_100.std(axis=1) / np.sqrt(psth_0_100.shape[1]) 
    sem_0_25 = psth_0_25.std(axis=1) / np.sqrt(psth_0_25.shape[1])
    sem_0_12 = psth_0_12.std(axis=1) / np.sqrt(psth_0_12.shape[1])
    sem_0_06 = psth_0_06.std(axis=1) / np.sqrt(psth_0_06.shape[1])
    sem_0_00 = psth_0_00.std(axis=1) / np.sqrt(psth_0_00.shape[1])

    average_values_1_100 = psth_1_100.mean(axis=1)
    average_values_1_25 = psth_1_25.mean(axis=1)
    average_values_1_12 = psth_1_12.mean(axis=1)
    average_values_1_06 = psth_1_06.mean(axis=1)
    average_values_1_00 = psth_1_00.mean(axis=1)
    average_values_0_100 = psth_0_100.mean(axis=1)
    average_values_0_25 = psth_0_25.mean(axis=1)
    average_values_0_12 = psth_0_12.mean(axis=1)
    average_values_0_06 = psth_0_06.mean(axis=1)
    average_values_0_00 = psth_0_00.mean(axis=1) 

    # Create a colormap
    colors = ["#03071e","#d00000","#e85d04","#ffba08","#d3d3d3"]

    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Plot for correct ones on the left subplot
    ax = axs[0]
    ax.plot(average_values_1_100, color=colors[0], linewidth=3, label="Corr 1.0", alpha=0.8)
    ax.fill_between(range(len(average_values_1_100)), average_values_1_100 - sem_1_100, average_values_1_100 + sem_1_100, color=colors[0], alpha=0.15)

    ax.plot(average_values_1_25, color=colors[1], linewidth=3, label="Corr 0.25", alpha=0.8)
    ax.fill_between(range(len(average_values_1_25)), average_values_1_25 - sem_1_25, average_values_1_25 + sem_1_25, color=colors[1], alpha=0.15)

    ax.plot(average_values_1_12, color=colors[2], linewidth=3, label="Corr 0.125", alpha=0.8)
    ax.fill_between(range(len(average_values_1_12)), average_values_1_12 - sem_1_12, average_values_1_12 + sem_1_12, color=colors[2], alpha=0.15)

    ax.plot(average_values_1_06, color=colors[3], linewidth=3, label="Corr 0.0625", alpha=0.8)
    ax.fill_between(range(len(average_values_1_06)), average_values_1_06 - sem_1_06, average_values_1_06 + sem_1_06, color=colors[3], alpha=0.15)

    ax.plot(average_values_1_00, color=colors[4], linewidth=3, label="Corr 0.0", alpha=0.8)
    ax.fill_between(range(len(average_values_1_00)), average_values_1_00 - sem_1_00, average_values_1_00 + sem_1_00, color=colors[4], alpha=0.15) 

    ax.set_title(f"Correct NM response at {EVENT_NAME}\n" + f"{mouse}_{date}_{region_number}", fontsize=FONTSIZE_1)
    ax.set_xlabel('time since event (s)', fontsize=FONTSIZE_2)
    ax.set_ylabel('zdFF', fontsize=FONTSIZE_2)
    ax.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
    ax.legend(fontsize=FONTSIZE_3, frameon=False) 
    ax.grid(False) 

    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)

    ax.spines['bottom'].set_color("black")
    ax.spines['left'].set_color("black") 

    # Plot for incorrect ones on the right subplot
    ax = axs[1]
    ax.plot(average_values_0_100, color=colors[0], linewidth=3, label="Inc 1.0", alpha=0.8)
    ax.fill_between(range(len(average_values_0_100)), average_values_0_100 - sem_0_100, average_values_0_100 + sem_0_100, color=colors[0], alpha=0.15)

    ax.plot(average_values_0_25, color=colors[1], linewidth=3, label="Inc 0.25", alpha=0.8)
    ax.fill_between(range(len(average_values_0_25)), average_values_0_25 - sem_0_25, average_values_0_25 + sem_0_25, color=colors[1], alpha=0.15)

    ax.plot(average_values_0_12, color=colors[2], linewidth=3, label="Inc 0.125", alpha=0.8)
    ax.fill_between(range(len(average_values_0_12)), average_values_0_12 - sem_0_12, average_values_0_12 + sem_0_12, color=colors[2], alpha=0.15)

    ax.plot(average_values_0_06, color=colors[3], linewidth=3, label="Inc 0.0625", alpha=0.8)
    ax.fill_between(range(len(average_values_0_06)), average_values_0_06 - sem_0_06, average_values_0_06 + sem_0_06, color=colors[3], alpha=0.15)

    ax.plot(average_values_0_00, color=colors[4], linewidth=3, label="Inc 0.0", alpha=0.8)
    ax.fill_between(range(len(average_values_0_00)), average_values_0_00 - sem_0_00, average_values_0_00 + sem_0_00, color=colors[4], alpha=0.15)

    ax.set_title(f"Incorrect NM response at {EVENT_NAME}\n" + f"{mouse}_{date}_{region_number}", fontsize=FONTSIZE_1)
    ax.set_xlabel('time since event (s)', fontsize=FONTSIZE_2)
    ax.set_ylabel('zdFF', fontsize=FONTSIZE_2)
    ax.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
    ax.legend(fontsize=FONTSIZE_3, frameon=False) 
    ax.grid(False) 

    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)

    ax.spines['bottom'].set_color("black")
    ax.spines['left'].set_color("black") 

    plt.tight_layout()
    plt.savefig(f'/home/kceniabougrova/Documents/figures_forlabmeeting_May2024/Fig02_corrinc_{mouse}_{date}_{region_number}_{EVENT_NAME}.png')
    plt.show()

#%%
#####################################
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
import os
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
prefix = "demux_nph_ZFM-06305_"
################################################
file_list = list_files_in_folder(path, prefix)
for file_name in file_list: 
    mouse = file_name[10:19]
    date = file_name[20:30] 
    region_number = file_name[31:32]
    print(mouse, date, region_number)
    ################################################
    """ CHANGE HERE """
    EVENT_NAME = "feedback_times"
    ################################################

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


    eid,df_trials = get_eid(mouse,date) 
    path_nph = f'demux_nph_{mouse}_{date}_{region_number}_{eid}' 
    df_nph = pd.read_parquet(os.path.join(path, path_nph+'.pqt')) 

    idx = 2
    new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
    df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
    #create allUContrasts 
    df_trials['allUContrasts'] = df_trials['allContrasts']
    df_trials.loc[df_trials['contrastRight'].isna(), 'allUContrasts'] = df_trials['allContrasts'] * -1
    df_trials.insert(loc=3, column='allUContrasts', value=df_trials.pop('allUContrasts'))

    #create reactionTime
    # reactionTime = np.array((df_trials["firstMovement_times"])-(df_trials["stimOnTrigger_times"]))
    # df_trials["reactionTime"] = reactionTime 


    a = one.load_object(eid, 'trials')
    trials = a.to_df()

    df_trials = df_trials[0:len(df_trials)-1] #to avoid the last trial not having photometry data 

    fig, ax = iblphotometry.plots.plot_raw_data_df(df_nph) 

    array_timestamps_bpod = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
    event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
    idx_event = np.searchsorted(array_timestamps_bpod, event_test) #check idx where they would be included, in a sorted way 
    print(idx_event)

    """ create a column with the trial number in the nph df """
    df_nph["trial_number"] = 0 #create a new column for the trial_number 
    df_nph.loc[idx_event,"trial_number"]=1
    df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

    # df.to_parquet('/home/kcenia/Desktop/testserver_2023_08_31/photometry20230831.parquet') 

    PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
    SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR

    sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
    n_trials = df_trials.shape[0]

    psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
    # psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials-1))

    event_feedback = np.array(df_trials[EVENT_NAME]) #pick the feedback timestamps 
    # event_feedback = event_feedback[0:len(event_feedback)-1] #KB added 20240327 CHECK WITH OW

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
    
    psth_1_100 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == 1) & (df_trials.allContrasts == 1))]] 
    psth_1_25 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.25))]]
    psth_1_12 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.1250))]]
    psth_1_06 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == 1) & (df_trials.allContrasts == 0.0625))]]
    psth_1_00 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == 1) & (df_trials.allContrasts == 0))]] 
    psth_0_100 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == -1) & (df_trials.allContrasts == 1))]] 
    psth_0_25 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.25))]]
    psth_0_12 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.1250))]]
    psth_0_06 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == -1) & (df_trials.allContrasts == 0.0625))]]
    psth_0_00 = df_nph.calcium.values[psth_idx[:,((df_trials.feedbackType == -1) & (df_trials.allContrasts == 0))]] 

    sem_1_100 = psth_1_100.std(axis=1) / np.sqrt(psth_1_100.shape[1])
    sem_1_25 = psth_1_25.std(axis=1) / np.sqrt(psth_1_25.shape[1])
    sem_1_12 = psth_1_12.std(axis=1) / np.sqrt(psth_1_12.shape[1])
    sem_1_06 = psth_1_06.std(axis=1) / np.sqrt(psth_1_06.shape[1])
    sem_1_00 = psth_1_00.std(axis=1) / np.sqrt(psth_1_00.shape[1])
    sem_0_100 = psth_0_100.std(axis=1) / np.sqrt(psth_0_100.shape[1]) 
    sem_0_25 = psth_0_25.std(axis=1) / np.sqrt(psth_0_25.shape[1])
    sem_0_12 = psth_0_12.std(axis=1) / np.sqrt(psth_0_12.shape[1])
    sem_0_06 = psth_0_06.std(axis=1) / np.sqrt(psth_0_06.shape[1])
    sem_0_00 = psth_0_00.std(axis=1) / np.sqrt(psth_0_00.shape[1])

    average_values_1_100 = psth_1_100.mean(axis=1)
    average_values_1_25 = psth_1_25.mean(axis=1)
    average_values_1_12 = psth_1_12.mean(axis=1)
    average_values_1_06 = psth_1_06.mean(axis=1)
    average_values_1_00 = psth_1_00.mean(axis=1)
    average_values_0_100 = psth_0_100.mean(axis=1)
    average_values_0_25 = psth_0_25.mean(axis=1)
    average_values_0_12 = psth_0_12.mean(axis=1)
    average_values_0_06 = psth_0_06.mean(axis=1)
    average_values_0_00 = psth_0_00.mean(axis=1) 

    # Create a colormap
    colors = ["#03071e","#d00000","#e85d04","#ffba08","#d3d3d3"]

    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Plot for correct ones on the left subplot
    ax = axs[0]
    ax.plot(average_values_1_100, color=colors[0], linewidth=3, label="Corr 1.0", alpha=0.8)
    ax.fill_between(range(len(average_values_1_100)), average_values_1_100 - sem_1_100, average_values_1_100 + sem_1_100, color=colors[0], alpha=0.15)

    ax.plot(average_values_1_25, color=colors[1], linewidth=3, label="Corr 0.25", alpha=0.8)
    ax.fill_between(range(len(average_values_1_25)), average_values_1_25 - sem_1_25, average_values_1_25 + sem_1_25, color=colors[1], alpha=0.15)

    ax.plot(average_values_1_12, color=colors[2], linewidth=3, label="Corr 0.125", alpha=0.8)
    ax.fill_between(range(len(average_values_1_12)), average_values_1_12 - sem_1_12, average_values_1_12 + sem_1_12, color=colors[2], alpha=0.15)

    ax.plot(average_values_1_06, color=colors[3], linewidth=3, label="Corr 0.0625", alpha=0.8)
    ax.fill_between(range(len(average_values_1_06)), average_values_1_06 - sem_1_06, average_values_1_06 + sem_1_06, color=colors[3], alpha=0.15)

    ax.plot(average_values_1_00, color=colors[4], linewidth=3, label="Corr 0.0", alpha=0.8)
    ax.fill_between(range(len(average_values_1_00)), average_values_1_00 - sem_1_00, average_values_1_00 + sem_1_00, color=colors[4], alpha=0.15) 

    ax.set_title(f"Correct NM response at {EVENT_NAME}\n" + f"{mouse}_{date}_{region_number}", fontsize=FONTSIZE_1)
    ax.set_xlabel('time since event (s)', fontsize=FONTSIZE_2)
    ax.set_ylabel('zdFF', fontsize=FONTSIZE_2)
    ax.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
    ax.legend(fontsize=FONTSIZE_3, frameon=False) 
    ax.grid(False) 

    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)

    ax.spines['bottom'].set_color("black")
    ax.spines['left'].set_color("black") 

    # Plot for incorrect ones on the right subplot
    ax = axs[1]
    ax.plot(average_values_0_100, color=colors[0], linewidth=3, label="Inc 1.0", alpha=0.8)
    ax.fill_between(range(len(average_values_0_100)), average_values_0_100 - sem_0_100, average_values_0_100 + sem_0_100, color=colors[0], alpha=0.15)

    ax.plot(average_values_0_25, color=colors[1], linewidth=3, label="Inc 0.25", alpha=0.8)
    ax.fill_between(range(len(average_values_0_25)), average_values_0_25 - sem_0_25, average_values_0_25 + sem_0_25, color=colors[1], alpha=0.15)

    ax.plot(average_values_0_12, color=colors[2], linewidth=3, label="Inc 0.125", alpha=0.8)
    ax.fill_between(range(len(average_values_0_12)), average_values_0_12 - sem_0_12, average_values_0_12 + sem_0_12, color=colors[2], alpha=0.15)

    ax.plot(average_values_0_06, color=colors[3], linewidth=3, label="Inc 0.0625", alpha=0.8)
    ax.fill_between(range(len(average_values_0_06)), average_values_0_06 - sem_0_06, average_values_0_06 + sem_0_06, color=colors[3], alpha=0.15)

    ax.plot(average_values_0_00, color=colors[4], linewidth=3, label="Inc 0.0", alpha=0.8)
    ax.fill_between(range(len(average_values_0_00)), average_values_0_00 - sem_0_00, average_values_0_00 + sem_0_00, color=colors[4], alpha=0.15)

    ax.set_title(f"Incorrect NM response at {EVENT_NAME}\n" + f"{mouse}_{date}_{region_number}", fontsize=FONTSIZE_1)
    ax.set_xlabel('time since event (s)', fontsize=FONTSIZE_2)
    ax.set_ylabel('zdFF', fontsize=FONTSIZE_2)
    ax.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
    ax.legend(fontsize=FONTSIZE_3, frameon=False) 
    ax.grid(False) 

    ax.spines[['right', 'top']].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)

    ax.spines['bottom'].set_color("black")
    ax.spines['left'].set_color("black") 

    plt.tight_layout()
    plt.savefig(f'/home/kceniabougrova/Documents/figures_forlabmeeting_May2024/Fig02_corrinc_{mouse}_{date}_{region_number}_{EVENT_NAME}.png')
    plt.show()
###############################################################

