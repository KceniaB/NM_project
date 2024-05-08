#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from brainbox.behavior.dlc import likelihood_threshold
from brainbox.behavior.dlc import get_speed
from scipy.signal import convolve


from one.api import ONE
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
