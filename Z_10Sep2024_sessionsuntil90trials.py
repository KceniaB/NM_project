#%% 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib.colors as mcolors
from matplotlib.dates import date2num
from brainbox.io.one import SessionLoader
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
        session_path_behav = f'/home/kceniabougrova/Documents/nph/Behav_2024Mar20/{mouse}/{date}/001/'
        df_alldata = extract_all(session_path_behav)
        table_data = df_alldata[0]['table']
        trials = pd.DataFrame(table_data) 
    return eid, trials 

#%%





df_goodsessions = pd.read_csv('/home/ibladmin/Downloads/Mice_GOOD_sorted.csv') 
# df_goodsessions = pd.read_csv('/mnt/h0/kb/Mice performance tables 100 2.csv')
df_goodsessions['Date'] = pd.to_datetime(df_goodsessions['Date'], format='%m/%d/%Y')
df_gs = df_goodsessions[['Mouse', 'Date', 'NM', 'region']] 

# Edit the event! 
EVENT = 'feedback_times'

# Initialize empty containers
psth_combined = None
df_trials_combined = pd.DataFrame()

# # Choose the NM
# NM="DA" #"DA", "5HT", "NE", "ACh"
# df_goodsessions = df_gs[df_gs["NM"]==NM].reset_index(drop=True)
df_goodsessions = df_gs
mice_sessions = df_goodsessions.groupby('Mouse')
cmap = plt.get_cmap('viridis')  


EXCLUDES = [] 
IMIN = 0  # To start from here when rerunning; from scratch: write 0

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
        df_trials = df_trials[0:90]

        path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/' 
        path = path_initial + f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy'

        # Load psth_idx from file
        psth_idx = np.load(path)
        psth_idx = psth_idx[:, :90] 

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
        mean_psth = np.mean(psth_idx, axis=1)
        stderr_psth = np.std(psth_idx, axis=1) / np.sqrt(psth_idx.shape[1])
        plt.figure()
        plt.suptitle("Mouse = "+mouse)
        plt.axvline(x=29, color='black', linestyle='dashed')
        # Plot the mean line
        plt.plot(np.arange(len(mean_psth)), mean_psth, color='blue', label=date)

        # Shaded area for the error (mean ± stderr)
        plt.fill_between(np.arange(len(mean_psth)), mean_psth - stderr_psth, mean_psth + stderr_psth, 
            color='blue', alpha=0.1)
        # Get the current limits of the x-axis
        x_min, x_max = plt.xlim()
        # Define the number of ticks
        num_ticks = 10
        # Generate x-tick positions based on the actual data range
        xtick_positions = np.linspace(x_min, x_max, num_ticks)
        # Map the custom labels from -1 to 2
        xtick_labels = np.linspace(-1, 2, num_ticks)
        # Set the custom x-ticks with positions and mapped labels
        plt.xticks(xtick_positions, labels=np.round(xtick_labels, 2))
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        plt.xlabel(f"time since {EVENT} (s)")
        plt.ylabel("Processed calcium signal") 
        plt.legend()
        plt.tight_layout()

        # Print shapes to verify
        print("Shape of psth_combined:", psth_combined.shape)
        print("Shape of df_trials_combined:", df_trials_combined.shape)
    except: 
        print("ERROR: ", i)
        EXCLUDES.append(i)





#%%
#########################################################################################################################################################
""" 
this one works and saves the plots 
it was the v0
11Sep2024 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.dates import date2num

# Group by mouse and sort by date
mice_sessions = df_goodsessions.groupby('Mouse')

# Define colormap for gradient effect
cmap = plt.get_cmap('viridis')  

# Iterate over each mouse
for mouse, sessions in mice_sessions:
    sessions = sessions.sort_values('Date').reset_index(drop=True)  # Sort sessions by date
    num_sessions = len(sessions)
    num_plots = (num_sessions // 10) + (num_sessions % 10 != 0)  # Calculate number of plots needed
    
    for plot_index in range(num_plots):
        fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes, fixed size for consistency
        fig.suptitle(f"Mouse = {mouse} (Session Group {plot_index + 1})", fontsize=14)
        
        # Define session indices for the current plot
        start_index = plot_index * 10
        end_index = min(start_index + 10, num_sessions)
        current_sessions = sessions.iloc[start_index:end_index]
        
        # Normalize dates for colormap
        norm = mcolors.Normalize(vmin=date2num(sessions['Date'].min()), vmax=date2num(sessions['Date'].max()))
        
        # Loop over the sessions for this mouse and plot each session
        legend_lines = []
        legend_labels = []
        for i, row in current_sessions.iterrows():
            try:
                date = row['Date'].strftime('%Y-%m-%d')
                region = row['region']
                eid, df_trials = get_eid(mouse, date)
                
                path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/'
                path = path_initial + f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy'
                
                # Load psth_idx from file
                psth_idx = np.load(path)
                psth_idx = psth_idx[:, :90]
                
                # Concatenate psth_idx arrays
                if psth_combined is None:
                    psth_combined = psth_idx
                else:
                    psth_combined = np.hstack((psth_combined, psth_idx))
                
                # Create allContrasts and allSContrasts
                idx = 2
                new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight'])
                df_trials.insert(loc=idx, column='allContrasts', value=new_col)
                df_trials['allSContrasts'] = df_trials['allContrasts']
                df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
                df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))
                
                # Create reactionTime
                reactionTime = np.array((df_trials["firstMovement_times"]) - (df_trials["stimOnTrigger_times"]))
                df_trials["reactionTime"] = reactionTime
                
                # Concatenate df_trials DataFrames
                df_trials_combined = pd.concat([df_trials_combined, df_trials], axis=0)
                df_trials_combined.reset_index(drop=True, inplace=True)
                
                # Calculate mean and standard error across psth_idx
                mean_psth = np.mean(psth_idx, axis=1)
                stderr_psth = np.std(psth_idx, axis=1) / np.sqrt(psth_idx.shape[1])
                
                # Determine color for this session
                color = cmap(norm(date2num(row['Date'])))
                
                # Plot the mean line for this session
                line, = ax.plot(np.arange(len(mean_psth)), mean_psth, color=color)
                legend_lines.append(line)
                
                # Shaded area for the error (mean ± stderr)
                ax.fill_between(np.arange(len(mean_psth)), mean_psth - stderr_psth, mean_psth + stderr_psth,
                                color=color, alpha=0.1)
                
                # Add vertical line (no label)
                ax.axvline(x=29, color='black', linestyle='dashed')
                
                # Calculate additional statistics from df_trials (first 90 trials)
                first_90_trials = df_trials.head(90)
                num_unique_contrasts = len(first_90_trials['allContrasts'].unique())
                
                # Filter for correct choices where contrastLeft and contrastRight are 1
                correct_left = first_90_trials[(first_90_trials['contrastLeft'] == 1) & (first_90_trials['feedbackType'] == 1)]
                correct_right = first_90_trials[(first_90_trials['contrastRight'] == 1) & (first_90_trials['feedbackType'] == 1)]
                
                # Percentage of correct choices
                total_left_right_trials = len(first_90_trials[(first_90_trials['contrastLeft'] == 1) | 
                                                              (first_90_trials['contrastRight'] == 1)])
                percent_correct = ((len(correct_left) + len(correct_right)) / total_left_right_trials) * 100 if total_left_right_trials > 0 else 0

                # Append the compact legend label with date, contrasts, and correct percentage
                legend_labels.append(f"{date} | {num_unique_contrasts} | {percent_correct:.0f}%")

            except Exception as e:
                print(f"ERROR with session {i}: {e}")
                EXCLUDES.append(i)
        
        # Display the legend for the line plots outside the plot area
        ax.legend(legend_lines, legend_labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        # Format the x-axis
        ax.set_xlabel(f"time since {EVENT} (s)")
        ax.set_ylabel("Processed calcium signal")
        x_min, x_max = ax.get_xlim()
        num_ticks = 10
        xtick_positions = np.linspace(x_min, x_max, num_ticks)
        xtick_labels = np.linspace(-1, 2, num_ticks)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(np.round(xtick_labels, 2))

        # Remove the right and top spines for a cleaner look
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Adjust layout to make space for legends outside the plot
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the layout to make room for the legend and stats
        filename = f"mouse_{mouse}_session_group_{plot_index + 1}.png"
        plt.savefig(filename, bbox_inches='tight')  # Save with tight bounding box to avoid clipping
        plt.show()
#########################################################################################################################################################
#%% 
"""
also works well
4 subplots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.dates import date2num
from matplotlib import cm

# Group by mouse and sort by date
mice_sessions = df_goodsessions.groupby('Mouse')

# Change this to control how many sessions you want to display per plot (now 8 sessions per plot)
sessions_per_plot = 8

# Iterate over each mouse
for mouse, sessions in mice_sessions:
    sessions = sessions.sort_values('Date').reset_index(drop=True)  # Sort sessions by date
    num_sessions = len(sessions)
    num_plots = (num_sessions // sessions_per_plot) + (num_sessions % sessions_per_plot != 0)  # Calculate number of plots needed
    
    for plot_index in range(num_plots):
        # Create 4 subplots (1 row, 4 columns) for each session group
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # Increased figure size for 4 subplots
        
        # Set the main title
        fig.suptitle(f"Mouse = {mouse} (Session Group {plot_index + 1})", fontsize=16)
        
        # Define session indices for the current plot
        start_index = plot_index * sessions_per_plot
        end_index = min(start_index + sessions_per_plot, num_sessions)
        current_sessions = sessions.iloc[start_index:end_index]
        
        # Discretize the colormap
        cmap = plt.get_cmap('viridis', sessions_per_plot)  # Get the desired number of colors
        
        # Loop over the sessions for this mouse and plot each session
        legend_lines = []
        legend_labels = []
        for i, row in current_sessions.iterrows():
            try:
                date = row['Date'].strftime('%Y-%m-%d')
                region = row['region']
                eid, df_trials = get_eid(mouse, date)

                # Create allContrasts and allSContrasts
                idx = 2
                new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight'])
                df_trials.insert(loc=idx, column='allContrasts', value=new_col)
                df_trials['allSContrasts'] = df_trials['allContrasts']
                df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
                df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))
                
                # Create reactionTime
                reactionTime = np.array((df_trials["firstMovement_times"]) - (df_trials["stimOnTrigger_times"]))
                df_trials["reactionTime"] = reactionTime
                
                path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/'
                path = path_initial + f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy'
                
                # Load psth_idx from file
                psth_idx = np.load(path)
                
                # Split the psth_idx array into four chunks (0-90, 90-180, 180-270, 270-360)
                psth_chunks = [psth_idx[:, i:i+90] for i in range(0, 360, 90)]
                
                # Determine color for this session using session index
                color = cmap(i % sessions_per_plot)

                # Loop through each of the 4 trial ranges and plot on respective axes
                for ax_idx, ax in enumerate(axes):
                    mean_psth = np.mean(psth_chunks[ax_idx], axis=1)
                    stderr_psth = np.std(psth_chunks[ax_idx], axis=1) / np.sqrt(psth_chunks[ax_idx].shape[1])
                    
                    # Plot the mean line for this session on the respective subplot
                    line, = ax.plot(np.arange(len(mean_psth)), mean_psth, color=color)
                    legend_lines.append(line)
                    
                    # Shaded area for the error (mean ± stderr)
                    ax.fill_between(np.arange(len(mean_psth)), mean_psth - stderr_psth, mean_psth + stderr_psth,
                                    color=color, alpha=0.1)
                    
                    # Add vertical line (no label)
                    ax.axvline(x=29, color='black', linestyle='dashed')
                    
                    # Label the x-axis for only the bottom row
                    ax.set_xlabel(f"time since {EVENT} (s)")
                    ax.set_ylabel("Processed calcium signal")
                    
                    # Format the x-axis
                    x_min, x_max = ax.get_xlim()
                    num_ticks = 10
                    xtick_positions = np.linspace(x_min, x_max, num_ticks)
                    xtick_labels = np.linspace(-1, 2, num_ticks)
                    ax.set_xticks(xtick_positions)
                    ax.set_xticklabels(np.round(xtick_labels, 2))
                    
                    # Remove the right and top spines for a cleaner look
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                
                # Calculate additional statistics from df_trials (first 360 trials)
                first_360_trials = df_trials.head(360)
                num_unique_contrasts = len(first_360_trials['allContrasts'].unique())
                
                # Filter for correct choices where contrastLeft and contrastRight are 1
                correct_left = first_360_trials[(first_360_trials['contrastLeft'] == 1) & (first_360_trials['feedbackType'] == 1)]
                correct_right = first_360_trials[(first_360_trials['contrastRight'] == 1) & (first_360_trials['feedbackType'] == 1)]
                
                # Percentage of correct choices
                total_left_right_trials = len(first_360_trials[(first_360_trials['contrastLeft'] == 1) | 
                                                               (first_360_trials['contrastRight'] == 1)])
                percent_correct = ((len(correct_left) + len(correct_right)) / total_left_right_trials) * 100 if total_left_right_trials > 0 else 0

                # Append the compact legend label with date, contrasts, and correct percentage
                legend_labels.append(f"{date} | {num_unique_contrasts} | {percent_correct:.0f}%")

            except Exception as e:
                print(f"ERROR with session {i}: {e}")
                EXCLUDES.append(i)
        
        # Display the legend outside the plot area below the subplots
        fig.legend(legend_lines, legend_labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=10)
        
        # Adjust layout to make space for the legend
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make room for the legend
        filename = f"mouse_{mouse}_session_group_{plot_index + 1}.png"
        plt.savefig(filename, bbox_inches='tight')  # Save with tight bounding box to avoid clipping
        plt.show()



########################################################################################
#%%
"""
works well
4 subplots correct
"""
# Iterate over each mouse
for mouse, sessions in mice_sessions:
    sessions = sessions.sort_values('Date').reset_index(drop=True)  # Sort sessions by date
    num_sessions = len(sessions)
    num_plots = (num_sessions // sessions_per_plot) + (num_sessions % sessions_per_plot != 0)  # Calculate number of plots needed

    # Initialize variables to track the global y-limits across all sessions for this mouse
    global_ymin, global_ymax = np.inf, -np.inf

    # Step 1: Calculate the global y-limits (min/max) for all sessions of this mouse
    for plot_index in range(num_plots):
        start_index = plot_index * sessions_per_plot
        end_index = min(start_index + sessions_per_plot, num_sessions)
        current_sessions = sessions.iloc[start_index:end_index]

        for i, row in current_sessions.iterrows():
            try:
                date = row['Date'].strftime('%Y-%m-%d')
                region = row['region']
                eid, df_trials = get_eid(mouse, date)

                path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/'
                path = path_initial + f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy'

                # Load psth_idx from file
                psth_idx = np.load(path)

                # Split the psth_idx array into four chunks (0-90, 90-180, 180-270, 270-360)
                psth_chunks = [psth_idx[:, i:i + 90] for i in range(0, 360, 90)]

                # Loop through each of the 4 trial ranges
                for psth_chunk in psth_chunks:
                    mean_psth = np.mean(psth_chunk, axis=1)
                    stderr_psth = np.std(psth_chunk, axis=1) / np.sqrt(psth_chunk.shape[1])

                    # Update global min/max y-values
                    global_ymin = min(global_ymin, np.min(mean_psth - stderr_psth))
                    global_ymax = max(global_ymax, np.max(mean_psth + stderr_psth))

            except Exception as e:
                print(f"ERROR with session {i}: {e}")
                EXCLUDES.append(i)

    # Step 2: Plot the sessions with the same y-limits across all subplots for this mouse
    for plot_index in range(num_plots):
        # Create 4 subplots (1 row, 4 columns) for each session group
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # Increased figure size for 4 subplots

        # Set the main title
        fig.suptitle(f"Mouse = {mouse} (Session Group {plot_index + 1})", fontsize=16)

        # Define session indices for the current plot
        start_index = plot_index * sessions_per_plot
        end_index = min(start_index + sessions_per_plot, num_sessions)
        current_sessions = sessions.iloc[start_index:end_index]

        # Discretize the colormap
        cmap = plt.get_cmap('viridis', sessions_per_plot)  # Get the desired number of colors

        # Dictionary to map color to label
        color_to_label = {}

        # Loop over the sessions for this mouse and plot each session
        for session_idx, (i, row) in enumerate(current_sessions.iterrows()):
            try:
                date = row['Date'].strftime('%Y-%m-%d')
                region = row['region']
                eid, df_trials = get_eid(mouse, date)

                # Create allContrasts and allSContrasts
                idx = 2
                new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight'])
                df_trials.insert(loc=idx, column='allContrasts', value=new_col)
                df_trials['allSContrasts'] = df_trials['allContrasts']
                df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
                df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))
                
                # Create reactionTime
                reactionTime = np.array((df_trials["firstMovement_times"]) - (df_trials["stimOnTrigger_times"]))
                df_trials["reactionTime"] = reactionTime

                path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/'
                path = path_initial + f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy'

                # Load psth_idx from file
                psth_idx = np.load(path)

                # Split the psth_idx array into four chunks (0-90, 90-180, 180-270, 270-360)
                psth_chunks = [psth_idx[:, i:i + 90] for i in range(0, 360, 90)]

                # Determine color for this session using session index
                color = cmap(session_idx % sessions_per_plot)  # Use session_idx for color

                # Loop through each of the 4 trial ranges and plot on respective axes
                for ax_idx, ax in enumerate(axes):
                    mean_psth = np.mean(psth_chunks[ax_idx], axis=1)
                    stderr_psth = np.std(psth_chunks[ax_idx], axis=1) / np.sqrt(psth_chunks[ax_idx].shape[1])

                    # Plot the mean line for this session on the respective subplot
                    line, = ax.plot(np.arange(len(mean_psth)), mean_psth, color=color)

                    # Shaded area for the error (mean ± stderr)
                    ax.fill_between(np.arange(len(mean_psth)), mean_psth - stderr_psth, mean_psth + stderr_psth,
                                    color=color, alpha=0.1)

                    # Add vertical line (no label)
                    ax.axvline(x=29, color='black', linestyle='dashed')

                    # Label the x-axis for only the bottom row
                    ax.set_xlabel(f"time since {EVENT} (s)")
                    ax.set_ylabel("Processed calcium signal")

                    # Format the x-axis
                    x_min, x_max = ax.get_xlim()
                    num_ticks = 10
                    xtick_positions = np.linspace(x_min, x_max, num_ticks)
                    xtick_labels = np.linspace(-1, 2, num_ticks)
                    ax.set_xticks(xtick_positions)
                    ax.set_xticklabels(np.round(xtick_labels, 2))

                    # Remove the right and top spines for a cleaner look
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)

                    # Set the y-limits to be the same across all subplots
                    ax.set_ylim(global_ymin, global_ymax)

                    # Set the title for each subplot to indicate the trial range
                    trial_ranges = ['0-90', '91-180', '181-270', '271-360']
                    ax.set_title(f"Trials {trial_ranges[ax_idx]}")

                # Calculate additional statistics from df_trials (first 360 trials)
                first_360_trials = df_trials.head(360)
                num_unique_contrasts = len(first_360_trials['allContrasts'].unique())

                # Filter for correct choices where contrastLeft and contrastRight are 1
                correct_left = first_360_trials[(first_360_trials['contrastLeft'] == 1) & (first_360_trials['feedbackType'] == 1)]
                correct_right = first_360_trials[(first_360_trials['contrastRight'] == 1) & (first_360_trials['feedbackType'] == 1)]

                # Percentage of correct choices
                total_left_right_trials = len(first_360_trials[(first_360_trials['contrastLeft'] == 1) |
                                                               (first_360_trials['contrastRight'] == 1)])
                percent_correct = ((len(correct_left) + len(correct_right)) / total_left_right_trials) * 100 if total_left_right_trials > 0 else 0

                # Append the compact legend label with date, contrasts, and correct percentage
                label = f"{date} | {num_unique_contrasts} | {percent_correct:.0f}%"
                color_to_label[color] = label  # Map color to label

            except Exception as e:
                print(f"ERROR with session {i}: {e}")
                EXCLUDES.append(i)

        # Create the legend
        handles = [plt.Line2D([0], [0], color=color, lw=2) for color in color_to_label.keys()]
        labels = [color_to_label[color] for color in color_to_label.keys()]
        fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, -0.1), fontsize=10)

        # Adjust layout to make space for the legend
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make room for the legend
        filename = f"mouse_{mouse}_session_group_{plot_index + 1}_2.png"
        plt.savefig(filename, bbox_inches='tight')  # Save with tight bounding box to avoid clipping
        plt.show()

################################################################################################

#%%
"""
ADDED TO DRIVE
WORKS 
"""

"""
12Sep2024
works
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df_goodsessions = pd.read_csv('/home/ibladmin/Downloads/Mice_GOOD_sorted.csv')
df_goodsessions = pd.read_csv('/mnt/h0/kb/Mice performance tables 100 2.csv')
df_goodsessions['Date'] = df_goodsessions.date

df_goodsessions['Mouse'] = df_goodsessions.mouse

df_goodsessions['Date'] = pd.to_datetime(df_goodsessions['Date'], format='%m/%d/%Y')
df_gs = df_goodsessions[['Mouse', 'Date', 'NM', 'region']]
df_gs['Date'] = pd.to_datetime(df_gs['Date'], errors='coerce')

# Initialize empty containers
psth_combined = None
df_trials_combined = pd.DataFrame()

# Choose the NM
# NM="DA" #"DA", "5HT", "NE", "ACh"
# df_goodsessions = df_gs[df_gs["NM"]==NM].reset_index(drop=True)
df_goodsessions = df_gs
mice_sessions = df_goodsessions.groupby('Mouse')
cmap = plt.get_cmap('viridis')
sessions_per_plot = 8

EXCLUDES = []
IMIN = 0  # To start from here when rerunning; from scratch: write 0

# Iterate over each mouse
for mouse, sessions in mice_sessions:
    sessions = sessions.sort_values('Date').reset_index(drop=True)  # Sort sessions by date
    num_sessions = len(sessions)
    num_plots = (num_sessions // sessions_per_plot) + (num_sessions % sessions_per_plot != 0)  # Calculate number of plots needed

    # Initialize variables to track the global y-limits across all sessions for this mouse
    global_ymin, global_ymax = np.inf, -np.inf
    # Step 1: Calculate the global y-limits (min/max) for all sessions of this mouse
    for plot_index in range(num_plots):
        start_index = plot_index * sessions_per_plot
        end_index = min(start_index + sessions_per_plot, num_sessions)
        current_sessions = sessions.iloc[start_index:end_index]
        current_sessions['date'] = pd.to_datetime(current_sessions['date'], errors='coerce')   

        for i, row in current_sessions.iterrows():
            try:
                if pd.notna(row['date']):  # Check if the date is valid
                    date = row['date'].strftime('%Y-%m-%d')
                    print(date)
                region = row['region']
                eid, df_trials = get_eid(mouse, date)
                nm = row['NM']

                # Create allContrasts and allSContrasts
                idx = 2
                new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight'])
                df_trials.insert(loc=idx, column='allContrasts', value=new_col)
                df_trials['allSContrasts'] = df_trials['allContrasts']
                df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
                df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))

                # Create reactionTime
                reactionTime = np.array((df_trials["firstMovement_times"]) - (df_trials["stimOnTrigger_times"]))
                df_trials["reactionTime"] = reactionTime

                # Filter df_trials for feedback types and contrast conditions
                feedback1_trials = df_trials[df_trials['feedbackType'] == 1]
                feedback_minus1_trials = df_trials[df_trials['feedbackType'] == -1]
                contrast_high = df_trials[df_trials['allContrasts'].isin([1, 0.5, 0.25])]
                contrast_low = df_trials[df_trials['allContrasts'].isin([0, 0.0625])]

                # Determine the EVENT value based on the plot index
                if plot_index < 2:
                    EVENT = 'feedback_times'
                else:
                    EVENT = 'stimOnTrigger_times'

                path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/'
                path = path_initial + f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy'

                # Load psth_idx from file
                psth_idx = np.load(path)

                # Ensure psth_idx has enough trials before splitting into chunks
                total_trials = psth_idx.shape[1]  # Number of columns (trials)
                chunk_size = 90

                # Split the psth_idx array into available full chunks (0-90, 90-180, ...)
                psth_chunks = [psth_idx[:, i:i + chunk_size] for i in range(0, total_trials, chunk_size)]

                # Loop through each of the trial chunks
                for psth_chunk in psth_chunks:
                    # Filter psth_idx based on feedbackType=1 indices within this chunk
                    feedback1_indices = feedback1_trials.index.tolist()
                    chunk_start = psth_chunks.index(psth_chunk) * chunk_size
                    chunk_end = chunk_start + chunk_size
                    chunk_indices_1 = [i for i in feedback1_indices if chunk_start <= i < chunk_end]

                    if chunk_indices_1:
                        psth_idx_filtered = psth_chunk[:, [i - chunk_start for i in chunk_indices_1]]
                        mean_psth = np.mean(psth_idx_filtered, axis=1)
                        stderr_psth = np.std(psth_idx_filtered, axis=1) / np.sqrt(psth_idx_filtered.shape[1])
                        global_ymin = min(global_ymin, np.min(mean_psth - stderr_psth))
                        global_ymax = max(global_ymax, np.max(mean_psth + stderr_psth))

                    # Filter psth_idx based on feedbackType=-1 indices within this chunk
                    feedback_minus1_indices = feedback_minus1_trials.index.tolist()
                    chunk_indices_minus1 = [i for i in feedback_minus1_indices if chunk_start <= i < chunk_end]

                    if chunk_indices_minus1:
                        psth_idx_filtered = psth_chunk[:, [i - chunk_start for i in chunk_indices_minus1]]
                        mean_psth = np.mean(psth_idx_filtered, axis=1)
                        stderr_psth = np.std(psth_idx_filtered, axis=1) / np.sqrt(psth_idx_filtered.shape[1])
                        global_ymin = min(global_ymin, np.min(mean_psth - stderr_psth))
                        global_ymax = max(global_ymax, np.max(mean_psth + stderr_psth))

                    # Filter psth_idx based on contrast conditions within this chunk
                    contrast_high_indices = contrast_high.index.tolist()
                    contrast_low_indices = contrast_low.index.tolist()

                    chunk_indices_high = [i for i in contrast_high_indices if chunk_start <= i < chunk_end]
                    chunk_indices_low = [i for i in contrast_low_indices if chunk_start <= i < chunk_end]

                    if chunk_indices_high:
                        psth_idx_filtered = psth_chunk[:, [i - chunk_start for i in chunk_indices_high]]
                        mean_psth = np.mean(psth_idx_filtered, axis=1)
                        stderr_psth = np.std(psth_idx_filtered, axis=1) / np.sqrt(psth_idx_filtered.shape[1])
                        global_ymin = min(global_ymin, np.min(mean_psth - stderr_psth))
                        global_ymax = max(global_ymax, np.max(mean_psth + stderr_psth))

                    if chunk_indices_low:
                        psth_idx_filtered = psth_chunk[:, [i - chunk_start for i in chunk_indices_low]]
                        mean_psth = np.mean(psth_idx_filtered, axis=1)
                        stderr_psth = np.std(psth_idx_filtered, axis=1) / np.sqrt(psth_idx_filtered.shape[1])
                        global_ymin = min(global_ymin, np.min(mean_psth - stderr_psth))
                        global_ymax = max(global_ymax, np.max(mean_psth + stderr_psth))

            except Exception as e:
                print(f"ERROR with session {i}: {e}")
                EXCLUDES.append(i)



    # Step 2: Plot the sessions with the same y-limits across all subplots for this mouse
    for plot_index in range(num_plots):
        # Create 4 rows x 4 columns of subplots for each session group
        fig, axes = plt.subplots(4, 4, figsize=(18, 16))  # Adjust figure size for 4 rows

        # Set the main title
        fig.suptitle(f"Mouse = {mouse} (Session Group {plot_index + 1}) All Conditions", fontsize=16)

        # Define session indices for the current plot
        start_index = plot_index * sessions_per_plot
        end_index = min(start_index + sessions_per_plot, num_sessions)
        current_sessions = sessions.iloc[start_index:end_index]
        current_sessions['date'] = pd.to_datetime(current_sessions['date'], errors='coerce')   

        # Discretize the colormap
        cmap = plt.get_cmap('viridis', sessions_per_plot)  # Get the desired number of colors

        # Dictionary to map color to label
        color_to_label = {}

        # Loop over the sessions for this mouse and plot each session
        for session_idx, (i, row) in enumerate(current_sessions.iterrows()):
            try:
                if pd.notna(row['date']):  # Check if the date is valid
                    date = row['date'].strftime('%Y-%m-%d')
                    print(date)
                region = row['region']
                eid, df_trials = get_eid(mouse, date)

                # Create allContrasts and allSContrasts
                idx = 2
                new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight'])
                df_trials.insert(loc=idx, column='allContrasts', value=new_col)
                df_trials['allSContrasts'] = df_trials['allContrasts']
                df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
                df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))

                # Create reactionTime
                reactionTime = np.array((df_trials["firstMovement_times"]) - (df_trials["stimOnTrigger_times"]))
                df_trials["reactionTime"] = reactionTime

                # Filter df_trials for feedback types and contrast conditions
                feedback1_trials = df_trials[df_trials['feedbackType'] == 1]
                feedback_minus1_trials = df_trials[df_trials['feedbackType'] == -1]
                contrast_high = df_trials[df_trials['allContrasts'].isin([1, 0.5, 0.25])]
                contrast_low = df_trials[df_trials['allContrasts'].isin([0, 0.0625])]

                # Determine the EVENT value based on the row index
                for row_idx in range(4):
                    if row_idx < 2:
                        EVENT = 'feedback_times'
                    else:
                        EVENT = 'stimOnTrigger_times'

                    path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/'
                    path = path_initial + f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy'

                    # Load psth_idx from file
                    psth_idx = np.load(path)

                    # Ensure psth_idx has enough trials before splitting into chunks
                    total_trials = psth_idx.shape[1]  # Number of columns (trials)
                    chunk_size = 90

                    # Split the psth_idx array into available full chunks (0-90, 90-180, ...)
                    psth_chunks = [psth_idx[:, i:i + chunk_size] for i in range(0, total_trials, chunk_size)]

                    # Loop through each of the 4 trial ranges and plot on respective axes
                    for ax_idx in range(4):
                        if ax_idx >= len(psth_chunks):
                            continue  # If fewer than 4 chunks, skip extra axes

                        psth_chunk = psth_chunks[ax_idx]

                        # Select the subplot axis based on the row index and column index
                        ax = axes[row_idx, ax_idx]

                        chunk_start = ax_idx * chunk_size
                        chunk_end = chunk_start + chunk_size

                        # Plot feedbackType = 1 on the top rows (axes[0, ax_idx] and axes[1, ax_idx])
                        if row_idx == 0:
                            trials_indices = feedback1_trials.index
                            title = f"Feedback 1: Trials {chunk_start}-{chunk_end}"
                        elif row_idx == 1:
                            trials_indices = feedback_minus1_trials.index
                            title = f"Feedback -1: Trials {chunk_start}-{chunk_end}"
                        elif row_idx == 2:
                            trials_indices = contrast_high.index
                            title = f"High Contrasts: Trials {chunk_start}-{chunk_end}"
                        elif row_idx == 3:
                            trials_indices = contrast_low.index
                            title = f"Low Contrasts: Trials {chunk_start}-{chunk_end}"

                        filtered_indices = [i for i in trials_indices if chunk_start <= i < chunk_end]
                        if filtered_indices:
                            psth_idx_filtered = psth_chunk[:, [i - chunk_start for i in filtered_indices]]
                            mean_psth = np.mean(psth_idx_filtered, axis=1)
                            stderr_psth = np.std(psth_idx_filtered, axis=1) / np.sqrt(psth_idx_filtered.shape[1])

                            line, = ax.plot(np.arange(len(mean_psth)), mean_psth, color=cmap(session_idx % sessions_per_plot))
                            ax.fill_between(np.arange(len(mean_psth)), mean_psth - stderr_psth, mean_psth + stderr_psth,
                                            color=cmap(session_idx % sessions_per_plot), alpha=0.1)
                            ax.axvline(x=29, color='black', linestyle='dashed')
                            ax.set_title(title)
                            ax.set_ylim(global_ymin, global_ymax)
                            ax.set_xlabel(f"time since {EVENT} (s)")
                            ax.set_ylabel("Processed calcium signal")

                first_360_trials = df_trials.head(360)
                num_unique_contrasts = len(first_360_trials['allContrasts'].unique())

                color_to_label[cmap(session_idx % sessions_per_plot)] = f"Session {i} ({date})\nRegion: {region}\nContrasts: {num_unique_contrasts}"

            except Exception as e:
                print(f"ERROR with session {i}: {e}")
                EXCLUDES.append(i)

        # Add a shared legend at the bottom of the figure
        if color_to_label:
            handles = [plt.Line2D([0], [0], color=color, lw=2) for color in color_to_label.keys()]
            labels = [color_to_label[color] for color in color_to_label.keys()]
            fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, -0.1), fontsize=10)
        else:
            print("No valid session data to create a legend.")

        # Adjust layout to make space for the legend
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make room for the legend

        filename = f"mouse_{mouse}_session_group_{plot_index + 1}_all_conditions.png"
        plt.savefig(filename, bbox_inches='tight')  # Save with tight bounding box to avoid clipping

        plt.show()
        # Close the figure to release memory
        plt.close(fig)
print(EXCLUDES)


################################################################################################ 






































#%%
#%%
#%%
#%%
#%%
#%%

#%%
for i in range(len(df_gs)):
    if i < IMIN:
        continue
    if i in EXCLUDES:
        continue

    mouse = df_gs.Mouse[i]
    date = df_gs.Date[i]
    if isinstance(date, pd.Timestamp):
        date = date.strftime('%Y-%m-%d')
    region = df_gs.region[i]
    region = f"Region{region}G"
    
    eid, df_trials = get_eid(mouse, date) 
    df_trials["mouse"] = mouse
    df_trials["date"] = date 
    df_trials["region"] = region
    
    print(f"{mouse} | {date} | {region} | {eid}")
    print(f"i | {mouse} | {date} | {region} | {eid}")
    
    sl = SessionLoader(one=one, eid=eid) 
    file_photometry = sl.session_path.joinpath("alf", region, "raw_photometry.pqt") #KB commented 11Sep2024
    # file_photometry = session_path_nph+"alf/"+region+"/raw_photometry.pqt" #KB added 11Sep2024
    df_ph = pd.read_parquet(file_photometry) 
    print(df_ph)
    new_file_name = f"nph_{mouse}_{date}_{region}.pqt"
    new_file_path = os.path.join("/mnt/h0/kb/data/nph_pqt", new_file_name)
    new_file_name2 = f"nph_session_cut_{mouse}_{date}_{region}.pqt"
    new_file_path2 = os.path.join("/mnt/h0/kb/data/nph_pqt/session_cut", new_file_name2)
    
    df_ph_entire_signal = df_ph
    test=df_ph_entire_signal
    fs = (1 / np.median(np.diff(test.times.values))) #KB added 11Sep2024
    test['calcium_photobleach'] = photobleaching_lowpass(df_ph_entire_signal["raw_calcium"].values, fs=fs) #KB
    test['isosbestic_photobleach'] = photobleaching_lowpass(df_ph_entire_signal["raw_isosbestic"], fs=fs)
    test['calcium_jove2019'] = jove2019(df_ph_entire_signal["raw_calcium"], df_ph_entire_signal["raw_isosbestic"], fs=fs) 
    test['isosbestic_jove2019'] = jove2019(df_ph_entire_signal["raw_isosbestic"], df_ph_entire_signal["raw_calcium"], fs=fs)
    test['calcium_mad'] = preprocess_sliding_mad(df_ph_entire_signal["raw_calcium"].values, df_ph_entire_signal["times"].values, fs=fs)
    test['isosbestic_mad'] = preprocess_sliding_mad(df_ph_entire_signal["raw_isosbestic"].values, df_ph_entire_signal["times"].values, fs=fs) 
    df_ph_entire_signal=test
    df_ph_entire_signal.to_parquet(new_file_path)

    df_ph_crop = df_ph
    trial_start = df_trials["intervals_0"].iloc[0] - 10
    trial_end = df_trials["intervals_1"].iloc[-1] + 10
    selected_df_nph = df_ph_crop[(df_ph_crop["times"] >= trial_start) & (df_ph_crop["times"] <= trial_end)]
    selected_df_nph = selected_df_nph.reset_index(drop=True)
    selected_df_nph.to_parquet(new_file_path2)


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

    df_trials_400 = df_trials[0:401] 
    n_trials_400 = df_trials_400.shape[0] 

    df_trials_90 = df_trials[0:91] 
    n_trials_90 = df_trials_90.shape[0] 

    psth_idx = np.tile(sample_window[:,np.newaxis], (1, n_trials)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

    event_feedback = np.array(df_trials[EVENT_NAME]) #pick the feedback timestamps 

    feedback_idx = np.searchsorted(nph_times, event_feedback) #check idx where they would be included, in a sorted way 

    psth_idx += feedback_idx

    photometry_feedback = selected_df_nph.calcium_jove2019.values[psth_idx] 

    photometry_feedback_avg = np.mean(photometry_feedback, axis=1)
    # plt.plot(photometry_feedback_avg) 

    
    psth_idx_400 = np.tile(sample_window[:,np.newaxis], (1, n_trials_400)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

    event_feedback_400 = np.array(df_trials_400[EVENT_NAME]) #pick the feedback timestamps 

    feedback_idx_400 = np.searchsorted(nph_times, event_feedback_400) #check idx where they would be included, in a sorted way 

    psth_idx_400 += feedback_idx_400

    photometry_feedback_400 = selected_df_nph.calcium_jove2019.values[psth_idx_400] 

    photometry_feedback_avg_400 = np.mean(photometry_feedback_400, axis=1)
    # plt.plot(photometry_feedback_avg) 


    psth_idx_90 = np.tile(sample_window[:,np.newaxis], (1, n_trials_90)) #KB commented 20240327 BUT USE THIS ONE; CHECK WITH OW 

    event_feedback_90 = np.array(df_trials_90[EVENT_NAME]) #pick the feedback timestamps 

    feedback_idx_90 = np.searchsorted(nph_times, event_feedback_90) #check idx where they would be included, in a sorted way 

    psth_idx_90 += feedback_idx_90

    photometry_feedback_90 = selected_df_nph.calcium_jove2019.values[psth_idx_90] 

    photometry_feedback_avg_90 = np.mean(photometry_feedback_90, axis=1)
    # plt.plot(photometry_feedback_avg) 


    # import os

    # fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    # color_1 = '#008080'  # Brownish/greyish for first plot
    # color_2 = '#aa3e98' #add a purple greyish color 
    # color_3 = '#708090'  # Greyish for third plot
    # color_4 = '#008080'  # Teal for fourth plot

    # axes[0].plot(df_ph_entire_signal.times, df_ph_entire_signal.raw_calcium, color=color_1, linewidth=0.1)
    # axes[0].set_ylabel('Raw Calcium')
    # axes[0].set_title('Entire Signal, Raw Calcium')

    # axes[1].plot(selected_df_nph.times, selected_df_nph.raw_isosbestic, color=color_2, linewidth=0.1)
    # axes[1].set_ylabel('Raw isosbestic')
    # axes[1].set_title('Signal Cut, Raw Isosbestic')

    # axes[2].plot(selected_df_nph.times, selected_df_nph.calcium_jove2019, color=color_4, linewidth=0.1)
    # axes[2].set_ylabel('Processed Calcium')
    # axes[2].set_title('Signal Cut, Calcium Preprocessed')
    # axes[2].set_xlabel('Time (s)') 

    # # Check if the 399th trial exists and draw a vertical line
    # if len(df_trials) > 399:
    #     trial_400 = df_trials["intervals_1"].iloc[399]
    #     axes[2].axvline(x=trial_400, linestyle="dashed", color="gray")

    # fig.suptitle(f"{mouse}_{date}_{region}", fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show() 


    def avg_sem(data):
        avg = data.mean(axis=1)
        sem = data.std(axis=1) / np.sqrt(data.shape[1])
        return avg, sem

    # Create the figure and gridspec
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 3])

    def plot_neuromodulator(psth_combined, df_trials, psth_combined_90, df_trials_90, title, mouse):
        psth_correct = psth_combined[:, (df_trials.feedbackType == 1)] 
        psth_correct_90 = psth_combined_90[:, (df_trials_90.feedbackType == 1)] 

        avg2, sem2 = avg_sem(psth_correct_90)
        color = "#aa3e98"
        plt.plot(avg2, color=color, linewidth=2, label="first 90 trials")
        plt.fill_between(range(len(avg2)), avg2 - sem2, avg2 + sem2, color=color, alpha=0.18)
        
        plt.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
        plt.ylabel('Average Value')
        plt.xlabel('Time')
        plt.title(title+ ' mouse '+mouse, fontsize=16)

    # Plot for DA
    fig = plt.figure(figsize=(12, 12))
    plot_neuromodulator(photometry_feedback, df_trials, photometry_feedback_90, df_trials_90, title = "psth aligned to feedback ", mouse=df_trials.mouse[0]) #change here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Adding legend outside the plots
    plt.legend(fontsize=14)
    fig.suptitle('mouse = '+mouse+" | trial# =  "+str(len(df_trials_90))+' | contrast # = ', y=1.02, fontsize=18)
    # fig.suptitle('mouse = '+mouse+"trial# =  "+len(psth_correct_90)+'contrast # = '+len(df_trials_90.allContrasts) , y=1.02, fontsize=18)
    plt.tight_layout()
    plt.show()


    # Print the path of the saved file for verification
    print(f"Saved photometry file to: {new_file_path2}")

    # Print the path of the saved file for verification
    print(f"Saved photometry file to: {new_file_path}")
# %%
