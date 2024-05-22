
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
"""
Load wheel data 
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

df_trials = df_trials[0:len(df_trials)-1]



"""
now I should have df_trials and df_nph
"""
# df = df_nph
# df_alldata = df_trials
# %% 
fig, ax = iblphotometry.plots.plot_raw_data_df(df_nph) 

array_timestamps_bpod = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(array_timestamps_bpod, event_test) #check idx where they would be included, in a sorted way 
print(idx_event)

""" create a column with the trial number in the nph df """
df_nph["trial_number"] = 0 #create a new column for the trial_number 
# for i in idx_event: 
#     df["trial_number"][i] = 1 #add an 1 whenever that event occurred (intervals_0) 
df_nph.loc[idx_event,"trial_number"]=1

df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

# df.to_parquet('/home/kcenia/Desktop/testserver_2023_08_31/photometry20230831.parquet') 

PERIEVENT_WINDOW = [-1,2] #never to be changed!!! "constant" 
SAMPLING_RATE = 30 #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR
sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
n_trials = df_trials.shape[0]

psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 
# psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials-1))

event_feedback = np.array(df_trials.feedback_times) #pick the feedback timestamps 
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
split correct vs incorrect 
"""
array_timestamps_bpod = np.array(df_nph.times) #pick the nph timestamps transformed to bpod clock 
event_test = np.array(df_trials.intervals_0) #pick the intervals_0 timestamps 
idx_event = np.searchsorted(array_timestamps_bpod, event_test) #check idx where they would be included, in a sorted way 
print(idx_event)

""" create a column with the trial number in the nph df """
df_nph["trial_number"] = 0 #create a new column for the trial_number 
# for i in idx_event: 
#     df["trial_number"][i] = 1 #add an 1 whenever that event occurred (intervals_0) 
df_nph.loc[idx_event,"trial_number"]=1

df_nph["trial_number"] = df_nph.trial_number.cumsum() #sum the [i-1] to i in order to get the trial number 

# df.to_parquet('/home/kcenia/Desktop/testserver_2023_08_31/photometry20230831.parquet') 

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
plt.axvline(x=30, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed") 
plt.show()

sns.heatmap(psth_error.T, cbar=True)
plt.axvline(x=30, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed") 
plt.show()




#%%
#%%
plt.rcParams["figure.figsize"] = (8,6)
width_ratios = [1, 1]
FONTSIZE_1 = 16
FONTSIZE_2 = 15
FONTSIZE_3 = 15 

event_time = 30 

psth_50 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.5) & (df_trials.feedbackType == -1))]]
psth_20 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.2) & (df_trials.feedbackType == -1))]]
psth_80 = df_nph.calcium.values[psth_idx[:,((df_trials.probabilityLeft == 0.8) & (df_trials.feedbackType == -1))]]

plt.plot(psth_50, color='#cdb4db', linewidth=1, alpha=0.2) 
plt.plot(psth_20, color='#ffafcc', linewidth=1, alpha=0.2) 
plt.plot(psth_80, color='#a2d2ff', linewidth=1, alpha=0.2) 
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
plt.legend(("0.5","0.2","0.8")) 
plt.ylim(-0.005,0.004)
plt.grid(False)
plt.show()












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
plt.plot(average_values_50, color='#220901', linewidth=2.5) 
plt.plot(average_values_20, color='#f6aa1c', linewidth=2.5) 
plt.plot(average_values_80, color='#941b0c', linewidth=2.5) 
plt.plot(average_values_10, color='#bc3908', linewidth=2.5) 
plt.title("NM response at feedback outcome = correct, cR = 1, for the 3 blocks", fontsize=FONTSIZE_1)
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
plt.legend(("1","0","0.1250","0.0625")) 
plt.grid(False)
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

event_time = 90 

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
plt.show()
#####################################################################################

# %%
#####################################################################################
plt.rcParams["figure.figsize"] = (14,10)
plt.figure(dpi=300)
width_ratios = [1, 1]
FONTSIZE_1 = 16
FONTSIZE_2 = 15
FONTSIZE_3 = 15 

event_time = 90 

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
plt.show() 
#####################################################################################





# %% 
#####################################################################################
plt.rcParams["figure.figsize"] = (14,10)
plt.figure(dpi=300)
width_ratios = [1, 1]
FONTSIZE_1 = 16
FONTSIZE_2 = 15
FONTSIZE_3 = 15 

event_time = 90 

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
FONTSIZE_3 = 21 

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
plt.legend(fontsize="20")

plt.title("NM response at feedback outcome = correct, reactionTime 0.85 1.75", fontsize=FONTSIZE_1)
plt.xlabel('time since event (s)', fontsize=FONTSIZE_2)
plt.ylabel('zdFF', fontsize=FONTSIZE_2)
plt.axvline(x=event_time, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
plt.xticks(fontsize=FONTSIZE_2)
plt.yticks(fontsize=FONTSIZE_2)
plt.grid(False)
ax = plt.subplot()
ax.spines[['right', 'top']].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
# plt.ylim(-0.005, 0.0075)
plt.show()


# %%
