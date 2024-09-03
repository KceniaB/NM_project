"""
KB 
28082024
SAVE THE MULTIPLE PLOTS IN 1 IMAGE 
BY MOUSE, ALL SESSIONS 
"""
########################################################################################### 
###########################################################################################
###########################################################################################
""" 
1. SAVE THE MULTIPLE PLOTS IN 1 IMAGE - BY MOUSE, ALL SESSIONS 

"""
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

mice = {"ZFM-04392","ZFM-04019","ZFM-04022","ZFM-04026","ZFM-03447","ZFM-03448", "ZFM-03450", "ZFM-03059","ZFM-03062","ZFM-03065","ZFM-03061", "ZFM-05235","ZFM-05236","ZFM-05245","ZFM-05248", "ZFM-06305","ZFM-06948","ZFM-04533","ZFM-04534", "ZFM-06171", "ZFM-06271","ZFM-06272","ZFM-06262","ZFM-06275"}

for mouse in mice: 
    try: 
        # Directory containing the images
        directory = '/mnt/h0/kb/data/psth_npy/'

        # Prefix to match the filenames
        # prefix = f'Fig03_psth_feedback_times_{mouse}_'
        prefix = f'Fig03_psth_{mouse}_' 
        # prefix = f'Fig02_feedback_times_{mouse}_'
        prefix = f'Fig02_{mouse}_'

        # Regular expression to extract date from filenames
        date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')

        # Get all filenames that match the prefix
        files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.png')]

        # Sort files by date extracted from filename
        files.sort(key=lambda x: re.search(date_pattern, x).group())

        # Number of images
        n_images = len(files)

        # Set up the figure: determine the grid size (e.g., square or close to square)
        n_cols = int(n_images**0.5)
        n_rows = n_images // n_cols + (n_images % n_cols > 0)

        # Create the subplots with a larger figure size
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18))

        # Flatten axes for easy iteration if more than one subplot
        axes = axes.flatten() if n_images > 1 else [axes]

        # Mouse name and event details
        mouse_name = f"{mouse}"
        event = "feedback_times"
        plot = "allsessions_lineplot"

        # Loop through files and plot each image in the appropriate subplot
        for ax, file in zip(axes, files):
            # Load the image
            img = mpimg.imread(os.path.join(directory, file))
            ax.imshow(img)
            ax.axis('off')  # Hide axes

            # Extract date from filename and set as title
            date_str = re.search(date_pattern, file).group()
            ax.set_title(f"{mouse_name} - {date_str}", fontsize=12)

        # Hide any remaining empty subplots if they exist
        for ax in axes[len(files):]:
            ax.axis('off')

        # Adjust layout and display
        plt.tight_layout()

        # Save the figure with high DPI (300 is a common choice for high-quality output)
        # plt.savefig(os.path.join(directory, f"{mouse_name}_{event}_{plot}.png"), dpi=300)

        plt.show()
    except: 
        print(f"no data for {mouse}")import os

###########################################################################################
###########################################################################################
###########################################################################################
# #%%
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

mice = {"ZFM-04392","ZFM-04019","ZFM-04022","ZFM-04026","ZFM-03447","ZFM-03448", "ZFM-03450", "ZFM-03059","ZFM-03062","ZFM-03065","ZFM-03061", "ZFM-05235","ZFM-05236","ZFM-05245","ZFM-05248", "ZFM-06305","ZFM-06948","ZFM-04533","ZFM-04534", "ZFM-06171", "ZFM-06271","ZFM-06272","ZFM-06262","ZFM-06275"}

for mouse in mice: 
    try: 
        # Directory containing the images
        directory = '/mnt/h0/kb/data/psth_npy/'

        # Prefix to match the filenames
        # prefix = f'Fig03_psth_feedback_times_{mouse}_'
        prefix = f'Fig03_psth_{mouse}_'

        # Regular expression to extract date from filenames
        date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')

        # Get all filenames that match the prefix
        files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.png')]

        # Sort files by date extracted from filename
        files.sort(key=lambda x: re.search(date_pattern, x).group())

        # Number of images
        n_images = len(files)

        # Set up the figure: determine the grid size (e.g., square or close to square)
        n_cols = int(n_images**0.5)
        n_rows = n_images // n_cols + (n_images % n_cols > 0)

        # Create the subplots with a larger figure size
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18))

        # Flatten axes for easy iteration if more than one subplot
        axes = axes.flatten() if n_images > 1 else [axes]

        # Mouse name and event details
        mouse_name = f"{mouse}"
        event = "feedback_times"
        plot = "allsessions_lineplot"

        # Loop through files and plot each image in the appropriate subplot
        for ax, file in zip(axes, files):
            # Load the image
            img = mpimg.imread(os.path.join(directory, file))
            ax.imshow(img)
            ax.axis('off')  # Hide axes

            # Extract date from filename and set as title
            date_str = re.search(date_pattern, file).group()
            ax.set_title(f"{mouse_name} - {date_str}", fontsize=12)

        # Hide any remaining empty subplots if they exist
        for ax in axes[len(files):]:
            ax.axis('off')

        # Adjust layout and display
        plt.tight_layout()

        # Save the figure with high DPI (300 is a common choice for high-quality output)
        plt.savefig(os.path.join(directory, f"{mouse_name}_{event}_{plot}.png"), dpi=300)

        plt.show()
    except: 
        print(f"no data for {mouse}")
        continue

        continue

###########################################################################################
###########################################################################################
###########################################################################################

#%%
"""
KB
29082024
2. LIST THE FILES IN A FOLDER AND SORT THEM 
    SAVE IN EXCEL

""" 
import os
import re
import pandas as pd

# Directory where the files are located
directory = '/mnt/h0/kb/data/psth_npy/'

# Prefix for filtering the files
prefix = f'Fig02_'

# Regular expression to extract the date from filenames
date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')

# Get all filenames that match the prefix and end with '.png'
files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.png')]

# Extract dates and sort files by date
sorted_files = sorted(files, key=lambda x: date_pattern.search(x).group())

# Create a DataFrame for exporting to Excel
df = pd.DataFrame(sorted_files, columns=['Filename'])

# Save the sorted filenames to an Excel file
output_file = os.path.join(directory, 'sorted_files.xlsx')
df.to_excel(output_file, index=False)

print(f'Sorted filenames have been saved to {output_file}')

###########################################################################################
###########################################################################################
###########################################################################################

#%%
"""
3. CHECK THE GENERATED IMAGES IN 1. AND LABEL THE FILES IN 2. ACCORDING TO IF IT IS A GOOD SESSION OR NOT 

"""

###########################################################################################
###########################################################################################
###########################################################################################

#%%
"""
4. LOAD THE EXCEL
    FILTER BY "G" FOR GOOD SESSIONS 
    ADD Mouse date region EXTRACTED FROM Filename 

""" 
import pandas as pd 
import re

# Define the file path
file_path = '/home/ibladmin/Downloads/3. Results - sessions summarized .xlsx'
df = pd.read_excel(file_path)

# Filter the DataFrame to include only rows where the column 'Good/Recheck/Bad' has the value 'G'
df_G = df[df['Good/Recheck/Bad'] == 'G']
df_G = df_G.reset_index(drop=True) 

# Regular expression to capture the mouse, date, and region
pattern = r'_(ZFM-\d+)_([\d\-]+)_(Region\d+G)'

# Extract the mouse, date, and region using str.extract
df_G[['mouse', 'date', 'region']] = df_G['Filename'].str.extract(pattern) 

###########################################################################################
###########################################################################################
###########################################################################################

#%%
"""
5. LOAD THE RAW PHOTOMETRY AND BEHAVIOR FOR THOSE SESSIONS AND REDO THE PRE-PROCESSING! 

""" 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from iblphotometry.preprocessing import jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 
from one.api import ONE #always after the imports 
one = ONE(cache_dir="/mnt/h0/kb/data/one") 

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

EXCLUDES=[]
IMIN=0

for i,rec in df_G_2.iterrows(): 
    try: 
        if i < IMIN:
            continue
        if i in EXCLUDES:
            continue
        #get data info
        region = rec.region
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
        SAMPLING_RATE = int(fs) #not a constant: print(1/np.mean(np.diff(array_timestamps_bpod))) #sampling rate #acq_FR 
        EVENT = "feedback_times"

        sample_window = np.arange(PERIEVENT_WINDOW[0] * SAMPLING_RATE, PERIEVENT_WINDOW[1] * SAMPLING_RATE + 1)
        n_trials = df_trials.shape[0]

        psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

        event_feedback = np.array(df_trials[EVENT]) #pick the feedback timestamps 

        feedback_idx = np.searchsorted(array_timestamps_bpod, event_feedback) #check idx where they would be included, in a sorted way 

        psth_idx += feedback_idx

        # try: 
        photometry_s_1 = df_nph.calcium_photobleach.values[psth_idx] 
        np.save(f'/mnt/h0/kb/data/psth_npy/30082024/preprocess_calcium_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_1)
        photometry_s_2 = df_nph.isosbestic_photobleach.values[psth_idx] 
        np.save(f'/mnt/h0/kb/data/psth_npy/30082024/preprocess_isosbestic_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_2)
        photometry_s_3 = df_nph.calcium_jove2019.values[psth_idx] 
        np.save(f'/mnt/h0/kb/data/psth_npy/30082024/preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_3)
        photometry_s_4 = df_nph.isosbestic_jove2019.values[psth_idx] 
        np.save(f'/mnt/h0/kb/data/psth_npy/30082024/preprocess_isosbestic_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_4)
        photometry_s_5 = df_nph.calcium_mad.values[psth_idx] 
        np.save(f'/mnt/h0/kb/data/psth_npy/30082024/preprocess_calcium_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_5)
        photometry_s_6 = df_nph.isosbestic_mad.values[psth_idx] 
        np.save(f'/mnt/h0/kb/data/psth_npy/30082024/preprocess_isosbestic_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_6) 

        # except: 
        #     print("#####################################CLIPPED PSTH#################################")
        #     # Clip the indices to be within the valid range, preserving the original shape
        #     psth_idx_clipped = np.clip(psth_idx, 0, len(df_nph.calcium_photobleach.values) - 1) 
        #     psth_idx = psth_idx_clipped 
        #     photometry_s_1 = df_nph.calcium_photobleach.values[psth_idx] 
        #     np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_1)
        #     photometry_s_2 = df_nph.isosbestic_photobleach.values[psth_idx] 
        #     np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_photobleach_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_2)
        #     photometry_s_3 = df_nph.calcium_jove2019.values[psth_idx] 
        #     np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_3)
        #     photometry_s_4 = df_nph.isosbestic_jove2019.values[psth_idx] 
        #     np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_4)
        #     photometry_s_5 = df_nph.calcium_mad.values[psth_idx] 
        #     np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_5)
        #     photometry_s_6 = df_nph.isosbestic_mad.values[psth_idx] 
        #     np.save(f'/mnt/h0/kb/data/psth_npy/preprocess_isosbestic_mad_{EVENT}_{mouse}_{date}_{region}_{eid}.npy', photometry_s_6) 


        psth_good = df_nph.calcium_mad.values[psth_idx[:,(df_trials.feedbackType == 1)]]
        psth_error = df_nph.calcium_mad.values[psth_idx[:,(df_trials.feedbackType == -1)]]
        # Calculate averages and SEM
        psth_good_avg = psth_good.mean(axis=1)
        sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
        psth_error_avg = psth_error.mean(axis=1)
        sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

        time_vector = np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], len(psth_good_avg))

        # Create the figure and gridspec
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

        # Plot the heatmap and line plot for correct trials
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(psth_good.T, cbar=False, ax=ax1) #, center = 0.0)
        ax1.invert_yaxis()
        ax1.axvline(x=SAMPLING_RATE, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
        ax1.set_title('Correct Trials')
        # Set x-axis tick labels to show time in seconds for the heatmaps
        ticks = np.linspace(0, len(time_vector)-1, num=5)
        tick_labels = np.round(np.linspace(PERIEVENT_WINDOW[0], PERIEVENT_WINDOW[1], num=5), 2)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(tick_labels)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(time_vector, psth_good_avg, color='#2f9c95', linewidth=3) 
        # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
        # ax2.fill_between(time_vector, psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
        ax2.fill_between(time_vector, psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
        ax2.axvline(x=0, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax2.set_ylabel('Average Value')
        ax2.set_xlabel('Time (s)')

        # Plot the heatmap and line plot for incorrect trials
        ax3 = fig.add_subplot(gs[0, 1])
        sns.heatmap(psth_error.T, cbar=False, ax=ax3) #, center = 0.0)
        ax3.invert_yaxis()
        ax3.axvline(x=SAMPLING_RATE, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
        ax3.set_title('Incorrect Trials')

        ax3.set_xticks(ticks)
        ax3.set_xticklabels(tick_labels)

        ax4 = fig.add_subplot(gs[1, 1], sharey=ax2)
        ax4.plot(time_vector, psth_error_avg, color='#d62828', linewidth=3)
        ax4.fill_between(time_vector, psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
        ax4.axvline(x=0, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax4.set_ylabel('Average Value')
        ax4.set_xlabel('Time (s)')

        fig.suptitle(f'calcium_mad_{EVENT}_{mouse}_{date}_{region}_{eid}', y=1, fontsize=14)
        plt.tight_layout()
        plt.savefig(f'/mnt/h0/kb/data/psth_npy/30082024/Fig02_{EVENT}_{mouse}_{date}_{region}_{eid}.png')
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
        ax1.plot(time_vector, psth_good_avg_1, color='#0892a5', linewidth=3) 
        # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
        ax1.fill_between(time_vector, psth_good_avg_1 - sem_good_1, psth_good_avg_1 + sem_good_1, color='#0892a5', alpha=0.15) 
        ax1.plot(time_vector, psth_error_avg_1, color='#d62828', linewidth=3)
        ax1.fill_between(time_vector, psth_error_avg_1 - sem_error_1, psth_error_avg_1 + sem_error_1, color='#d62828', alpha=0.15)
        ax1.axvline(x=0, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax1.set_ylabel('Average Value')
        ax1.set_xlabel('Time (s)') 

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_vector, psth_good_avg_2, color='#0892a5', linewidth=3) 
        # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
        ax2.fill_between(time_vector, psth_good_avg_2 - sem_good_2, psth_good_avg_2 + sem_good_2, color='#0892a5', alpha=0.15) 
        ax2.plot(time_vector, psth_error_avg_2, color='#d62828', linewidth=3)
        ax2.fill_between(time_vector, psth_error_avg_2 - sem_error_2, psth_error_avg_2 + sem_error_2, color='#d62828', alpha=0.15)
        ax2.axvline(x=0, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax2.set_ylabel('Average Value')
        ax2.set_xlabel('Time (s)') 

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(time_vector, psth_good_avg_3, color='#0892a5', linewidth=3) 
        # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
        ax3.fill_between(time_vector, psth_good_avg_3 - sem_good_3, psth_good_avg_3 + sem_good_3, color='#0892a5', alpha=0.15) 
        ax3.plot(time_vector, psth_error_avg_3, color='#d62828', linewidth=3)
        ax3.fill_between(time_vector, psth_error_avg_3 - sem_error_3, psth_error_avg_3 + sem_error_3, color='#d62828', alpha=0.15)
        ax3.axvline(x=0, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        ax3.set_ylabel('Average Value')
        ax3.set_xlabel('Time (s)') 

        fig.suptitle(f'{EVENT}_{mouse}_{date}_{region}_{eid}', y=1, fontsize=14)
        plt.tight_layout()
        plt.savefig(f'/mnt/h0/kb/data/psth_npy/30082024/Fig03_psth_{EVENT}_{mouse}_{date}_{region}_{eid}.png')
        plt.show() 
        
        print(f"DONE {mouse} | {date} | {region} | {eid}")
    

        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Assuming `time_vector`, `df_trials`, `psth_idx`, and `df_nph` are already defined
        unique_contrasts = np.sort(df_trials.allContrasts.unique())

        # Define the subsets of data
        all_data = df_trials
        correct_data = df_trials[df_trials.feedbackType == 1]
        incorrect_data = df_trials[df_trials.feedbackType == -1]

        # Function to plot psth for different contrast levels
        def plot_contrasts(ax, data, psth_idx, color_map, title):
            for contrast in unique_contrasts:
                trials_for_contrast = data[data.allContrasts == contrast]
                idx_for_contrast = psth_idx[:, trials_for_contrast.index]
                
                psth_values = df_nph.calcium_jove2019.values[idx_for_contrast]
                psth_avg = psth_values.mean(axis=1)
                sem = psth_values.std(axis=1) / np.sqrt(psth_values.shape[1])
                
                ax.plot(time_vector, psth_avg, color=color_map[contrast], linewidth=1.75, label=f'{contrast}')
                ax.fill_between(time_vector, psth_avg - sem, psth_avg + sem, color=color_map[contrast], alpha=0.15)
            
            ax.axvline(x=0, color="black", alpha=0.9, linewidth=2, linestyle="dashed")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Average Value')
            ax.set_title(title)
            ax.legend(title='Contrast')

        # Define color map for the contrasts
        # Create a color gradient from dark blue to light blue for contrasts between -1 and 1
        n_colors = len(unique_contrasts) - 2  # Number of contrasts excluding -1.0, 1.0, and 0
        blue_palette = sns.color_palette("Blues", n_colors=n_colors)

        # Create a dictionary that maps each contrast to a color
        color_map = {}
        for i, contrast in enumerate(unique_contrasts):
            if contrast == 1.0 or contrast == -1.0:
                color_map[contrast] = 'black'
            elif contrast == 0:
                color_map[contrast] = 'yellow'
            else:
                # Adjust index to fit within the blue palette for non-extreme contrasts
                color_map[contrast] = blue_palette[i-1] if contrast != 0 else 'yellow'

        # Create the figure and gridspec
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 3)

        # Plot for all contrasts
        ax1 = fig.add_subplot(gs[0, 0])
        plot_contrasts(ax1, all_data, psth_idx, color_map, "All Contrasts")

        # Plot for correct trials
        ax2 = fig.add_subplot(gs[0, 1])
        plot_contrasts(ax2, correct_data, psth_idx, color_map, "Correct Trials (Feedback Type = 1)")

        # Plot for incorrect trials
        ax3 = fig.add_subplot(gs[0, 2])
        plot_contrasts(ax3, incorrect_data, psth_idx, color_map, "Incorrect Trials (Feedback Type = -1)")

        # Adjust layout and show
        fig.suptitle(f'{EVENT}_{mouse}_{date}_{region}_{eid}', y=0.95, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'/mnt/h0/kb/data/psth_npy/30082024/Fig04_psth_{EVENT}_{mouse}_{date}_{region}_{eid}.png')
        plt.show()




    except: 
        EXCLUDES.append(i)
        IMIN=i
        print(i,"#####################################################################")


































###########################################################################################
###########################################################################################
###########################################################################################

#%%
"""
6. COMPRESS OR EXPAND THE DATA AROUND 2 BEHAVIORAL EVENTS 

""" 
import numpy as np
import matplotlib.pyplot as plt

# Convert timestamps to numpy arrays
array_timestamps_bpod = np.array(df_nph.times)
stimOnTrigger_times = np.array(df_trials.stimOnTrigger_times)
feedback_times = np.array(df_trials.feedback_times)

# Calculate the time difference between feedback and stimOnTrigger
time_diff = feedback_times - stimOnTrigger_times

# Filter trials where the time difference is less than 3 seconds
valid_trials_mask = time_diff < 2
stimOnTrigger_times_filtered = stimOnTrigger_times[valid_trials_mask]
feedback_times_filtered = feedback_times[valid_trials_mask]

# Find indices corresponding to stimOnTrigger_times and feedback_times in df_nph
stim_idx_filtered = np.searchsorted(array_timestamps_bpod, stimOnTrigger_times_filtered)
feedback_idx_filtered = np.searchsorted(array_timestamps_bpod, feedback_times_filtered)

# Extract photometry data between stimOnTrigger_times and feedback_times
photometry_segments_filtered = []
for i in range(len(stim_idx_filtered)):
    photometry_segment = df_nph.calcium_mad.values[stim_idx_filtered[i]:feedback_idx_filtered[i]]
    photometry_segments_filtered.append(photometry_segment)

# Normalize Photometry Data to Fit a 1.5-Second Window
PERIEVENT_WINDOW = 1  # Desired time window in seconds
SAMPLING_RATE = int(fs)  # Sampling rate in Hz
target_duration = int(PERIEVENT_WINDOW * SAMPLING_RATE)  # Convert to samples

def normalize_photometry_segment(segment, target_duration):
    original_duration = len(segment)
    original_time = np.linspace(0, original_duration - 1, original_duration)
    target_time = np.linspace(0, original_duration - 1, target_duration)
    normalized_segment = np.interp(target_time, original_time, segment)
    return normalized_segment

# Normalize each segment by compressing or expanding to 1.5 seconds
normalized_segments_filtered = [normalize_photometry_segment(segment, target_duration) for segment in photometry_segments_filtered]

# Plot Normalized Data
plt.figure(figsize=(10, 8))
for segment in normalized_segments_filtered:
    plt.plot(np.linspace(0, PERIEVENT_WINDOW, len(segment)), segment, linewidth=0.5, color='gray', alpha=0.15)

# Plot the average of all trials
average_photometry_filtered = np.mean(normalized_segments_filtered, axis=0)
plt.plot(np.linspace(0, PERIEVENT_WINDOW, len(average_photometry_filtered)), average_photometry_filtered, color='red', linewidth=3)

plt.axvline(x=0, color="black", linestyle="--", label="stimOnTrigger_times")
plt.axvline(x=PERIEVENT_WINDOW, color="blue", linestyle="--", label="feedback_times")

plt.xlabel("Time (s)")
plt.ylabel("Normalized Calcium Signal")
plt.title(f'Normalized Photometry Data (Filtered for Feedback-StimOnTrigger < 3s)')
plt.legend()
plt.show()
# %%
""" 6.2 plot sent to Zach, divided by contrast=0 and =1 and by feedbackType=1 and =-1""" 
import numpy as np
import matplotlib.pyplot as plt

# Function to normalize photometry data
def normalize_photometry_segment(segment, target_duration):
    original_duration = len(segment)
    original_time = np.linspace(0, original_duration - 1, original_duration)
    target_time = np.linspace(0, original_duration - 1, target_duration)
    normalized_segment = np.interp(target_time, original_time, segment)
    return normalized_segment

# General parameters
PERIEVENT_WINDOW = 0.5  # Desired time window in seconds
SAMPLING_RATE = int(fs)  # Sampling rate in Hz
target_duration = int(PERIEVENT_WINDOW * SAMPLING_RATE)  # Convert to samples

# Convert timestamps and other relevant columns to numpy arrays
array_timestamps_bpod = np.array(df_nph.times)
stimOnTrigger_times = np.array(df_trials.stimOnTrigger_times)
feedback_times = np.array(df_trials.feedback_times)
allContrasts = np.array(df_trials.allContrasts)
feedbackTypes = np.array(df_trials.feedbackType)

# Set up the subplot figure
fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

# Function to process and plot data
def process_and_plot(contrast_value, ax, title_suffix):
    # Filter trials based on the contrast and feedback type
    for feedback_type, color, label in [ 
        (1, 'steelblue', 'Correct'), 
        (-1, 'red', 'Incorrect')
    ]:
        # Apply filters for contrast and feedback type
        valid_mask = (allContrasts == contrast_value) & (feedbackTypes == feedback_type)
        stimOnTrigger_times_filtered = stimOnTrigger_times[valid_mask]
        feedback_times_filtered = feedback_times[valid_mask]

        # Calculate the time difference between feedback and stimOnTrigger
        time_diff = feedback_times_filtered - stimOnTrigger_times_filtered

        # Filter trials where the time difference is less than 5 seconds
        valid_trials_mask = time_diff < 5
        stimOnTrigger_times_filtered = stimOnTrigger_times_filtered[valid_trials_mask]
        feedback_times_filtered = feedback_times_filtered[valid_trials_mask]

        # Find indices corresponding to stimOnTrigger_times and feedback_times in df_nph
        stim_idx_filtered = np.searchsorted(array_timestamps_bpod, stimOnTrigger_times_filtered)
        feedback_idx_filtered = np.searchsorted(array_timestamps_bpod, feedback_times_filtered)

        # Extract photometry data between stimOnTrigger_times and feedback_times
        photometry_segments_filtered = []
        for i in range(len(stim_idx_filtered)):
            photometry_segment = df_nph.calcium_mad.values[stim_idx_filtered[i]:feedback_idx_filtered[i]]
            photometry_segments_filtered.append(photometry_segment)

        # Normalize each segment by compressing or expanding to 1.5 seconds
        normalized_segments_filtered = [normalize_photometry_segment(segment, target_duration) for segment in photometry_segments_filtered]

        # Plot the average of all trials
        if normalized_segments_filtered:  # Check if there are valid segments to plot
            average_photometry_filtered = np.mean(normalized_segments_filtered, axis=0)
            ax.plot(np.linspace(0, PERIEVENT_WINDOW, len(average_photometry_filtered)), average_photometry_filtered, color=color, linewidth=3, label=f'{label}')

    # Add vertical lines for events
    ax.axvline(x=0, color="black", linestyle="--", label="stimOnTrigger_times")
    ax.axvline(x=PERIEVENT_WINDOW, color="blue", linestyle="--", label="feedback_times")

    # Add labels and title
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Calcium Signal")
    ax.set_title(f'Normalized Photometry Data for 0.5s\n(Filtered for Feedback-StimOnTrigger < 5s)\nContrast = {title_suffix}')
    ax.legend()

# Process and plot data for contrast = 0
process_and_plot(0, axes[0], "0")

# Process and plot data for contrast = 1
process_and_plot(1, axes[1], "1")

# Adjust layout and display the plot
plt.ylim(-0.0075, 0.0075)
plt.tight_layout()
plt.show() 
#%% 
