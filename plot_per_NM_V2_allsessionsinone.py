"""
18July204August2024
KB 
from: plot_per_NM.py 
""" 
#%%
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

# Get the list of good sessions and their info 
# df_goodsessions = pd.read_csv('/home/ibladmin/Downloads/Mice_GOOD_sorted.csv') 
# df_goodsessions['Date'] = pd.to_datetime(df_goodsessions['Date'], format='%m/%d/%Y')
# df_gs = df_goodsessions[['Mouse', 'Date', 'NM', 'region']] 

dtype = {'nph_file': int, 'nph_bnc': int, 'region': int}
# df1 = pd.read_excel('/home/kceniabougrova/Downloads/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1 = pd.read_excel('/mnt/h0/kb/Mice performance tables 100.xlsx' , 'A4_2024',dtype=dtype) 
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d')
df1.date = df1.date.astype(str) 

# Edit the event! 
EVENT = 'feedback_times'

# Initialize empty containers
psth_combined = None 
df_trials_combined = pd.DataFrame()

EXCLUDES = []  
IMIN = 0

# Choose the NM
NM="DA" #"DA", "5HT", "NE", "ACh"
df_goodsessions = df1[df1["NM"]==NM].reset_index(drop=True)

####################################
#test_04 = test_04.drop(32).reset_index(drop=True)
#for DA: 
EXCLUDES = [5,6,8,12]  
#for 5HT: 
# EXCLUDES = [] 
#for NE: 
# EXCLUDES = []  
#for ACh: 
# EXCLUDES = []  
IMIN = 0 
df_goodsessions["Mouse"] = df_goodsessions.mouse
df_goodsessions["Date"] = df_goodsessions.date

#%%
for i in range(len(df_goodsessions)): 
    try: 
        print(i,df_goodsessions['Mouse'][i])
        if i < IMIN:
            continue
        if i in EXCLUDES:
            continue
        mouse = df_goodsessions.Mouse[i] 
        date = df_goodsessions.Date[i]
        if isinstance(date, pd.Timestamp):
            date = date.strftime('%Y-%m-%d')
        region = df_goodsessions.region[i]
        eid, df_trials = get_eid(mouse,date)
        print(f"{mouse} | {date} | {region} | {eid}")
        print(f"i | {mouse} | {date} | {region} | {eid}")
        print(f"i | {mouse} | {date} | {region} | {eid}")
        print(f"i | {mouse} | {date} | {region} | {eid}")
        print(f"i | {mouse} | {date} | {region} | {eid}")
        print(f"i | {mouse} | {date} | {region} | {eid}")
        print(f"i | {mouse} | {date} | {region} | {eid}")
        print(f"i | {mouse} | {date} | {region} | {eid}")
        print(f"i | {mouse} | {date} | {region} | {eid}")


        df_trials['trialNumber'] = range(1, len(df_trials) + 1) 
        df_trials["mouse"] = mouse
        df_trials["date"] = date
        df_trials["region"] = region
        df_trials["eid"] = eid 

        path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/' 
        # path_initial = f'/mnt/h0/kb/data/psth_npy/30082024/' 
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
    except: 
        print("ERROR: ",i)
        EXCLUDES.append(i)
        continue


    #%%
        ##################################################################################################
        # PLOT heatmap and correct vs incorrect 
        psth_good = psth_combined[:,(df_trials_combined.feedbackType == 1)]
        # psth_error = psth_combined[:,(df_trials_combined.feedbackType == -1)]
        # Calculate averages and SEM
        psth_good_avg = psth_good.mean(axis=1)
        # sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
        # psth_error_avg = psth_error.mean(axis=1)
        # sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

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

        fig.suptitle(f'jove2019_{EVENT}_{mouse}_{date}_{region}_{NM}_{eid}', y=1, fontsize=14)
        plt.tight_layout()
        plt.show()
    ##################################################################################################








#%%
""" PLOT PER MOUSE, DIVIDED BY SESSIONS, ONLY CORRECT TRIALS """
def avg_sem(data):
    avg = data.mean(axis=1)
    sem = data.std(axis=1) / np.sqrt(data.shape[1])
    return avg, sem

# Create the figure and gridspec
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 3])

def plot_neuromodulator(psth_combined, df_trials, title, mouse):
    # Filter to include only correct trials 
    # psth_mouse = psth_combined[:, (df_trials.mouse == mouse) & (df_trials.feedbackType == 1)]

    correct_trials_mask = (df_trials.mouse == mouse) & (df_trials.feedbackType == 1)
    df_trials_correct = df_trials[correct_trials_mask]
    unique_dates = df_trials_correct.date.unique()
    
    unique_dates.sort()

    cmap = plt.get_cmap('brg')
    colors = [cmap(i / len(unique_dates)) for i in range(len(unique_dates))]

    # Normalize transparency based on the number of sessions
    num_dates = len(unique_dates)
    alpha_increment = 1 / num_dates

    for i, date in enumerate(unique_dates):
        date_mask = df_trials.date == date
        combined_mask = correct_trials_mask & date_mask
        psth_combined_on_date = psth_combined[:, combined_mask.values]
        
        if psth_combined_on_date.shape[1] > 0:
            avg, sem = avg_sem(psth_combined_on_date)
            alpha = alpha_increment * (i + 1)
            color = colors[i]
            plt.plot(avg, color=color, linewidth=1, label=f'{date}')
            plt.fill_between(range(len(avg)), avg - sem, avg + sem, color=color, alpha=alpha * 0.4)
    
    plt.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    plt.ylabel('Average Value')
    plt.xlabel('Time')
    plt.title(title+ ' mouse '+mouse)

mouse_names = df_trials_combined_ACh.mouse.unique() #change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
for mouse_name in mouse_names: 
    # Plot for DA
    fig = plt.figure(figsize=(12, 12))
    plot_neuromodulator(psth_combined_ACh, df_trials_combined_ACh, 'ACh', mouse=mouse_name) #change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Adding legend outside the plots
    plt.legend()
    plt.legend()
    fig.suptitle('Neuromodulator activity for correct trials across different sessions in 1 mouse', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()










#%%





import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assuming avg_sem is a defined function to calculate average and SEM

def avg_sem(data):
    avg = np.mean(data, axis=1)
    sem = np.std(data, axis=1) / np.sqrt(data.shape[1])
    return avg, sem

# Create the figure and gridspec
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 3])

def plot_neuromodulator(psth_combined, df_trials, title, mouse):
    # Filter to include only correct trials 
    correct_trials_mask = (df_trials.mouse == mouse) & (df_trials.feedbackType == 1)
    df_trials_correct = df_trials[correct_trials_mask]
    
    # Get unique combinations of date and region
    unique_combinations = df_trials_correct[['date', 'region']].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(by=['date', 'region'])

    cmap = plt.get_cmap('brg')
    colors = [cmap(i / len(unique_combinations)) for i in range(len(unique_combinations))]

    # Normalize transparency based on the number of sessions
    num_combinations = len(unique_combinations)
    alpha_increment = 1 / num_combinations

    for i, (date, region) in enumerate(unique_combinations.values):
        combined_mask = correct_trials_mask & (df_trials.date == date) & (df_trials.region == region)
        psth_combined_on_combination = psth_combined[:, combined_mask.values]
        
        if psth_combined_on_combination.shape[1] > 0:
            avg, sem = avg_sem(psth_combined_on_combination)
            alpha = alpha_increment * (i + 1)
            color = colors[i]
            plt.plot(avg, color=color, linewidth=1, label=f'{date} - {region}')
            plt.fill_between(range(len(avg)), avg - sem, avg + sem, color=color, alpha=alpha * 0.4)
    
    plt.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    plt.ylabel('Average Value')
    plt.xlabel('Time')
    plt.title(f'{title} mouse {mouse}')

mouse_names = df_trials_combined_5HT.mouse.unique()
for mouse_name in mouse_names: 
    # Plot for DA
    fig = plt.figure(figsize=(12, 12))
    plot_neuromodulator(psth_combined_5HT, df_trials_combined_5HT, '5HT', mouse=mouse_name)
    # Adding legend outside the plots
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.suptitle('Neuromodulator activity for correct trials across different sessions in 1 mouse', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()
