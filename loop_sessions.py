"""
2024-April-11
KceniaB 

""" 

import numpy as np
import pandas as pd 
from brainbox.io.one import SessionLoader
from one.api import ONE
one = ONE()
import matplotlib.pyplot as plt 
import seaborn as sns
# from functions_nm import load_trials 
from functions_nm2 import * 
import neurodsp.utils 
from pathlib import Path
import iblphotometry.plots
import iblphotometry.dsp 

df1 = pd.read_excel('/home/kceniabougrova/Downloads/Mice performance tables 100.xlsx' , 'A4_2024') 






""" PHOTOMETRY """ 
df_test = df1[(df1.date == "2024-03-22") & (df1.Mouse == "ZFM-06275")]
def extract_data_info(df = df1): 
    for i in range(len(df["Mouse"])): 
        date = df.date #"2024-03-22"
        mouse = df.Mouse #"ZFM-06948" 
        nphfile_number = df.nph_file #"0"
        bncfile_number = df.nph_bnc #"0"
        # region = "Region"+(df["region"])+"G" 
    return mouse, date, nphfile_number, bncfile_number 
#, region 

mouse, date, nphfile_number, bncfile_number = extract_data_info(df = df_test)

if mouse == "ZFM-06948" or mouse == "ZFM-06305":
    nm = "ACh"

source_folder = ("/home/kceniabougrova/Documents/nph/"+date+"/")
df_nph = pd.read_csv(source_folder+"raw_photometry"+nphfile_number+".csv") 
df_nphttl = pd.read_csv(source_folder+"bonsai_DI"+bncfile_number+nphfile_number+".csv")



