
def extract_data_info(df): 
    """ 
    extracts in string format the mouse name, date of the session, nph file number, bnc file number and regions
    """
    for i in range(len(df["Mouse"])): 
        mouse = df.Mouse.values[0] #"ZFM-06948" 
        date = df['date'].dt.strftime('%Y-%m-%d').values[0] #"2024-03-22"
        nphfile_number = str(df.nph_file.values[0]) #"0"
        bncfile_number = str(df.nph_bnc.values[0]) #"0"
        region = "Region"+str(df.region.values[0])+"G"
        region2 = "Region"+str(df.region2.values[0])+"G" 
        if mouse == "ZFM-06948" or mouse == "ZFM-06305": 
            nm = "ACh" 
        elif mouse == "ZFM-06275": 
            nm = "NE" 
    return mouse, date, nphfile_number, bncfile_number, region, region2, nm

def get_nph(date, nphfile_number, bncfile_number): 
    source_folder = ("/home/kceniabougrova/Documents/nph/"+date+"/")
    df_nph = pd.read_csv(source_folder+"raw_photometry"+nphfile_number+".csv") 
    df_nphttl = pd.read_csv(source_folder+"bonsai_DI"+bncfile_number+nphfile_number+".csv") 
    return df_nph, df_nphttl 

def get_eid(mouse,date): 
    eids = one.search(subject=mouse, date=date) 
    eid = eids[0]
    ref = one.eid2ref(eid)
    print(eid)
    print(ref) 
    return eid 
        