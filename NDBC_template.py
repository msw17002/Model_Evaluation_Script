#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#https://www.ndbc.noaa.gov/measdes.shtml
#https://www.ndbc.noaa.gov/hull.shtml
#https://www.ndbc.noaa.gov/cmanht.shtml -> CMAN
#https://www.ndbc.noaa.gov/activestations.xml -> station metadata
#-----------------------------------------------------------------------------------------------
#Standard Meteorological Data
#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft
#2014 09 11 16 50 120  5.0  6.0   0.6     6   4.2 134 1016.5  29.3  30.5  24.4   MM +0.3    MM
#-----------------------------------------------------------------------------------------------
#Description of Fields
#WDIR:  Wind direction (the direction the wind is coming from in degrees clockwise from true N) during the ...
#       same period used for WSPD. See Wind Averaging Methods
#WSPD:  Wind speed (m/s) averaged over an eight-minute period for buoys and a two-minute period for land stations... 
#       Reported Hourly. See Wind Averaging Methods.
#GST:   Peak 5 or 8 second gust speed (m/s) measured during the eight-minute or two-minute period...
#       The 5 or 8 second period can be determined by payload, See the Sensor Reporting, Sampling, and Accuracy section.
#WVHT:  Significant wave height (meters) is calculated as the average of the highest one-third of all...
#       of the wave heights during the 20-minute sampling period. See the Wave Measurements section.
#DPD:   Dominant wave period (seconds) is the period with the maximum wave energy. See the Wave Measurements section.
#APD:   Average wave period (seconds) of all waves during the 20-minute period. See the Wave Measurements section.
#MWD:   The direction from which the waves at the dominant period (DPD) are coming. The units are degrees from true North,...
#       increasing clockwise, with North as 0 (zero) degrees and East as 90 degrees. See the Wave Measurements section.
#PRES:  Sea level pressure (hPa). For C-MAN sites and Great Lakes buoys, the recorded pressure is reduced to sea level...
#       using the method described in NWS Technical Procedures Bulletin 291 (11/14/80). ( labeled BAR in Historical files)
#ATMP:  Air temperature (Celsius). For sensor heights on buoys, see Hull Descriptions. For sensor heights at C-MAN stations,...
#       see C-MAN Sensor Locations
#WTMP:  Sea surface temperature (Celsius). For buoys the depth is referenced to the hull's waterline....
#       For fixed platforms it varies with tide, but is referenced to, for near Mean Lower Low Water (MLLW).
#DEWP:  Dewpoint temperature taken at the same height as the air temperature measurement.
#VIS:   Station visibility (nautical miles). Note that buoy stations are limited to reports from 0 to 1.6 nmi.
#PTDY:  Pressure Tendency is the direction (plus or minus) and the amount of pressure change (hPa)for a three... 
#       hour period ending at the time of observation. (not in Historical files)
#TIDE:  The water level in feet above or below Mean Lower Low Water (MLLW).
#-----------------------------------------------------------------------------------------------
import urllib.request
from math import atan
from astropy.io import ascii
import os
from datetime import datetime
import glob
from datetime import timedelta
import numpy as np
import numpy.matlib
import shutil
import gzip
import pandas as pd
from os import listdir
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

##############################################################################################################
def find_files( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]
##########################################################################################
# Round the observation's time to determine hourly observations
##########################################################################################
#def hour_rounder(t):
#    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
#    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
#               +timedelta(hours=t.minute//30))

def roundTime(dt=None, roundTo=0): #0 is an arb. number
    if dt == None : dt = datetime.datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + timedelta(0,rounding-seconds,-dt.microsecond)
#roundTime(Date_times[i,0],roundTo=minutes_round*60) #minutes_round is important. 60==houry, 10==10 minutely, 30==30 minutely, etc...
##########################################################################################
# Determine string of date as YYYYMMDDHH
##########################################################################################
def YYYYMMDDHH_string(Datetime_Value):
    if Datetime_Value.month>9: Month_str = str(Datetime_Value.month)
    else: Month_str = "0"+str(Datetime_Value.month)   
    if Datetime_Value.day>9: Day_str = str(Datetime_Value.day)
    else: Day_str = "0"+str(Datetime_Value.day)
    if Datetime_Value.hour>9: Hour_str = str(Datetime_Value.hour)
    else: Hour_str = "0"+str(Datetime_Value.hour)
    if Datetime_Value.minute>9: Minute_str = str(Datetime_Value.minute)
    else: Minute_str = "0"+str(Datetime_Value.minute)
    return (str(Datetime_Value.year)+Month_str+Day_str+Hour_str+Minute_str)

##########################################################################################
# CALCULATIONS
##########################################################################################
def Relative_Humidity(Temperature_C,Dew_Temperature_C):
    e  = 6.11*10**((7.5*Dew_Temperature_C)/(237.3+Dew_Temperature_C))
    es = 6.11*10**((7.5*Temperature_C)/(237.3+Temperature_C))
    RH = (e/es) * 100
    return (RH)

def Wet_Bulb_Stull(Temperature_C,RH_perc):
    C1 = 0.151977 #Stull Tw
    C2 = 8.313659 #Stull Tw 
    C3 = 1.676331 #Stull Tw
    C4 = 0.00391838 #Stull Tw
    C5 = 0.023101 #Stull Tw
    C6 = 4.686035 #Stull Tw
    tw = Temperature_C*atan(C1*(RH_perc+C2)**0.5)+atan(Temperature_C+RH_perc)-atan(RH_perc-C3)+C4*((RH_perc)**1.5)*atan(C5*RH_perc)-C6
    return (tw)

PYTHON_ARRAY=SED_REPLACE
Buoy_path       = './Buoy_Files/'
ev_yyyy         = PYTHON_ARRAY[0]
ev_mm           = PYTHON_ARRAY[1]
ev_dd           = PYTHON_ARRAY[2]
ev_hh           = PYTHON_ARRAY[3]
ev_dur          = PYTHON_ARRAY[4]

Buoy_url             = 'http://ndbc.noaa.gov/data/historical/stdmet/'
Buoy_data_path       = Buoy_path
Buoy_data_list       = find_files(Buoy_data_path,"")
Buoy_station         = pd.read_csv(Buoy_path+"Stations/"+"Buoy_Staions.csv")
Time_diff_limit_mins = 15  #should be less than minutes_round
minutes_round        = 60 #set to hourly observations

for event in range(1):#len(Events_List)):
    #Start time of evaluation
    Initialization_Year      = ev_yyyy #int(Events_List['Start_Year'][event])
    Initialization_Month     = ev_mm   #int(Events_List['Start_Month'][event])
    Initialization_Day       = ev_dd   #int(Events_List['Start_Day'][event])
    Initialization_Hour      = ev_hh   #int(Events_List['Start_Hour'][event])
    Event_Duration           = ev_dur  #int(Events_List['Durration'][event])
    # time-window for analysis (shrink dataframe)
    start_time = datetime(Initialization_Year,Initialization_Month,Initialization_Day,Initialization_Hour,
                          0,0)#+timedelta(hours=12)
    start_end  = start_time+timedelta(hours=int(Event_Duration))-timedelta(hours=12)
    print(start_time)
    print(start_end)
    str_begin  = YYYYMMDDHH_string(start_time)
    str_end    = YYYYMMDDHH_string(start_end)
    now        = datetime.now()
    now        = now.year
    for year in range(1):
        #Iterate by year
        download_log = 1
        for ndbc_station in range(len(Buoy_station)):#len(ndbc)):
            print(Buoy_station['STATION_ID'][ndbc_station])
            ########################################################################
            ###Download .txt.gz file and move it to archive directory
            ########################################################################    
            #Iterate by station
            station_file_targz = Buoy_station['STATION_ID'][ndbc_station]+'h'+str(Initialization_Year)+'.txt.gz'
            url                = Buoy_url+station_file_targz
            download_log       = 0
            #If the file doesn't exist, download it and process it--------------------------------------------------
            if (station_file_targz not in find_files( Buoy_data_path, suffix=".txt.gz" )):
                print("file doesnt exist")
                try: 
                    download_log = 1
                    print("downloading")
                    urllib.request.urlretrieve(url, station_file_targz)
                    shutil.move(station_file_targz,Buoy_data_path)
                    #print("uncompressing")
                    with gzip.open(Buoy_data_path+station_file_targz, 'rb') as f_in:
                        with open(Buoy_data_path+station_file_targz.replace(".gz",""), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                except:
                    download_log = 0
                    print("file not available")
            #--------------------------------------------------------------------------------------------------------
            #If the file does exist, but the year mached the current year, the file is removed then replaced with
            #an update .tar.gz file (in case the event was recent)
            elif (station_file_targz in find_files( Buoy_data_path, suffix=".txt.gz" )) & (int(Initialization_Year)==int(now)):
                print("file exists, but will be updated")
                os.remove(Buoy_data_path+station_file_targz)
                try: 
                    download_log = 1
                    print("downloading")
                    urllib.request.urlretrieve(url, station_file_targz)
                    shutil.move(station_file_targz,Buoy_data_path)
                    #print("uncompressing")
                    with gzip.open(Buoy_data_path+station_file_targz, 'rb') as f_in:
                        with open(Buoy_data_path+station_file_targz.replace(".gz",""), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                except: 
                    download_log = 0
                    print("file not available")
            #--------------------------------------------------------------------------------------------------------
            #The file already exists and it is from last year. Just read its processed text file.
            elif (station_file_targz in find_files( Buoy_data_path, suffix=".txt.gz" )) & (int(Initialization_Year)<int(now)):
                download_log = 1
                print("file has already been downloaded and is from a previous year")
            ########################################################################
            ###Read uncompressed data 
            ########################################################################  
            ASCII_header_try = ["YYYY","YY","MM","DD","hh","mm","WDIR","WD","WSPD","GST","WVHT","DPD",
                                     "APD","MWD","BAR","PRES","ATMP","WTMP","DEWP","VIS","TIDE"]
            if download_log == 1:
                print("data-processing")
                Buoy_yearly_ascii = ascii.read(Buoy_data_path+station_file_targz.replace(".gz","")) 
                Buoy_yearly_np    = np.zeros((len(Buoy_yearly_ascii),18))

                #Determine headers of ascii file
                ASCII_header_try = ["YYYY","YY","MM","DD","hh","mm","WDIR","WD","WSPD","GST","GSP","WVHT","DPD",
                    "APD","MWD","BAR","PRES","ATMP","WTMP","DEWP","VIS","TIDE"]
                Loop_names     = []
                for i in range(len(ASCII_header_try)):
                    try:
                        Buoy_yearly_ascii[ASCII_header_try[i]][0]
                        Loop_names.append(ASCII_header_try[i])
                    except: pass
                    
                #Remove missing cells 'MM'
                for i in range(len(Buoy_yearly_ascii)):
                    for j in range(18):
                        if Buoy_yearly_ascii[Loop_names[j]][i]=='MM': np.nan
                        else: Buoy_yearly_np[i,j] = Buoy_yearly_ascii[Loop_names[j]][i]

                ###Converting to pd DataFrame
                Buoy_yearly_pd = pd.DataFrame(Buoy_yearly_np)
                for col in range(18):
                    Buoy_yearly_pd.rename(columns={col: Loop_names[col]},inplace = True)
                station_repmat            = np.matlib.repmat(Buoy_station['STATION_ID'][ndbc_station],
                                                                  len(Buoy_yearly_pd), 1)
                Buoy_yearly_pd['Station'] = pd.DataFrame(station_repmat)
                ################################################################################################
                ### Add derived/old header file names where/when applicable
                ################################################################################################
                if 'WSPD10' not in Buoy_yearly_pd: Buoy_yearly_pd['WSPD10'] = '' 
                if 'WSPD20' not in Buoy_yearly_pd: Buoy_yearly_pd['WSPD20'] = '' 
                if 'YYYY' in Buoy_yearly_pd: Buoy_yearly_pd = Buoy_yearly_pd.rename(columns={'YYYY': 'YY'})
                if 'WD' in Buoy_yearly_pd:   Buoy_yearly_pd = Buoy_yearly_pd.rename(columns={'WD': 'WDIR'})
                if 'BAR' in Buoy_yearly_pd:  Buoy_yearly_pd =Buoy_yearly_pd.rename(columns={'BAR': 'PRES'})
                if 'WD' in Buoy_yearly_pd:  Buoy_yearly_pd =Buoy_yearly_pd.rename(columns={'WD': 'WDIR'})    
                if 'DIR' in Buoy_yearly_pd:  Buoy_yearly_pd =Buoy_yearly_pd.rename(columns={'DIR': 'WDIR'})       
                if 'SPD' in Buoy_yearly_pd:  Buoy_yearly_pd =Buoy_yearly_pd.rename(columns={'SPD': 'WSPD'})       
                if 'GSP' in Buoy_yearly_pd:  Buoy_yearly_pd =Buoy_yearly_pd.rename(columns={'GSP': 'GST'})       
                if 'BARO' in Buoy_yearly_pd:  Buoy_yearly_pd =Buoy_yearly_pd.rename(columns={'BARO': 'PRES'})       
                
                # determine if the observation falls within the time-window for analysis
                row_logical      = np.zeros((len(Buoy_yearly_pd), 1))
                # is the datetime within the analysis window
                for i in range(len(Buoy_yearly_pd)): #Is the date within the time of analysis?
                    dt_check = datetime(int(Buoy_yearly_pd['YY'][i]),int(Buoy_yearly_pd['MM'][i]),
                                                    int(Buoy_yearly_pd['DD'][i]),int(Buoy_yearly_pd['hh'][i]),
                                                    int(Buoy_yearly_pd['mm'][i]))
                    if start_time <= dt_check <= start_end:
                        row_logical[i,0] = True   
                # add boolean results back into the original dataframe [did the observation fall w/in the analysis time-window]
                Buoy_yearly_pd['Date_Boolean'] = pd.DataFrame(row_logical)
                # remove 'false' boolean results
                Buoy_yearly_pd                 = Buoy_yearly_pd[Buoy_yearly_pd.Date_Boolean != 0].reset_index()
                #########################################################################################################
                ### ***DETERMINE TIMES & T-DIFF*** ###
                #########################################################################################################
                # preallocate arrays to determine hourly observations (comparing vs. NWP hourly output)
                d = len(Buoy_yearly_pd)
                Time_Matrix = np.empty([d,2], dtype=object)
                time_valid  = np.empty([d,1], dtype=object)
                Date_times  = np.empty([d,1], dtype=object)
                # determine the time difference (minutes) b/n the nearest hour vs. the observation's time,
                # return a matrix of the observation's rounded YYYY, MM, DD, HH, and TDIFF 
                for i in range(d):
                    Date_times[i,0]     = datetime(int(Buoy_yearly_pd['YY'][i]),int(Buoy_yearly_pd['MM'][i]),
                                                    int(Buoy_yearly_pd['DD'][i]),int(Buoy_yearly_pd['hh'][i]),
                                                    int(Buoy_yearly_pd['mm'][i]))
                    time_round          = roundTime(Date_times[i,0],roundTo=minutes_round*60) #hour_rounder(Date_times[i,0])
                    time_valid[i,0]     = datetime.strptime(str(Date_times[i,0]), '%Y-%m-%d %H:%M:%S')
                    tdiff               = abs(time_valid[i,0]-time_round)
                    tdiff               = int(tdiff.seconds/60)
                    Time_Matrix[i,:]    = YYYYMMDDHH_string(time_round),str(tdiff)
                Dates       = np.unique(Time_Matrix[:,0])
                row_logical = np.zeros((len(Time_Matrix[:,0]), 1))
                for z in range(len(Dates)):
                    Log_Key = np.expand_dims(Time_Matrix[:,0]==Dates[z],1)
                    if len(Log_Key>0):
                        row             = [i for i, j in enumerate(Log_Key) if j]
                        Key_Time_Matrix = Time_Matrix[row]
                        row_check       = np.argmin(np.array(Key_Time_Matrix[:,1], dtype=np.float), axis=None, out=None)
                        row             = row[row_check]
                        row_logical[row,0] = 1
                Buoy_yearly_pd = pd.concat([Buoy_yearly_pd.reset_index(), pd.DataFrame(Time_Matrix).reset_index()], axis=1)
                if len(Buoy_yearly_pd)>0:
                    Buoy_yearly_pd['Hourly_Boolean'] = pd.DataFrame(row_logical)
                    Names_compressed = ['Date_Valid_Hour','Time_Diff_mins']
                    for col in range(2):
                        Buoy_yearly_pd.rename(columns={col: Names_compressed[col]},inplace = True)
                    ################################################################################################
                    ### Cleaning missing data/un-needed columns
                    ################################################################################################                
                    ################################################################################################
                    ### Cleaning missing data/un-needed columns
                    ################################################################################################                
                    Buoy_yearly_pd = Buoy_yearly_pd.drop(['level_0', 'index'], axis=1)
                    Buoy_yearly_pd = Buoy_yearly_pd[Buoy_yearly_pd.Hourly_Boolean != 0].reset_index()
                    Buoy_yearly_pd = Buoy_yearly_pd.drop('index', axis=1)
                    #ATMP	WTMP 999 remove
                    Loop_names        = ["WDIR","WSPD","GST","WVHT","DPD","APD","MWD","PRES","ATMP","WTMP","DEWP","VIS","TIDE"]
                    Loop_thresh       = [999.0 ,99.0  ,99.0 ,99.0  ,99.0 ,99.0 ,999.0,9999.0,999.0 ,999.0  ,999.0 ,99.0 ,99.0  ]
                    for i in range(len(Loop_names)):
                        Buoy_yearly_pd[Loop_names[i]] = Buoy_yearly_pd[Loop_names[i]].replace(Loop_thresh[i], np.nan)
                    Buoy_yearly_pd = Buoy_yearly_pd.replace('', np.nan)
                    Buoy_yearly_pd = Buoy_yearly_pd[Buoy_yearly_pd["Time_Diff_mins"].astype(float)<Time_diff_limit_mins]
                    Buoy_yearly_pd = Buoy_yearly_pd.reset_index()
                    Buoy_yearly_pd = Buoy_yearly_pd.drop('index', axis=1)
                    if len(Buoy_yearly_pd)>0:
                        Buoy_yearly_pd = Buoy_yearly_pd[["WDIR","WSPD","GST","WVHT","DPD","APD","MWD","PRES",
                                                                               "ATMP","WTMP","DEWP","VIS","TIDE","WSPD10","WSPD20",
                                                                             "Date_Valid_Hour","Station","Time_Diff_mins"]]
                        Buoy_yearly_pd["RELH"] = ''
                        Buoy_yearly_pd["WBTM"] = ''

                        Calc_Variables = np.empty((len(Buoy_yearly_pd), 2), dtype=object)
                        for i in range(len(Buoy_yearly_pd)):
                            Calc_Variables[i,0] = Relative_Humidity(Buoy_yearly_pd['ATMP'][i],Buoy_yearly_pd['DEWP'][i])
                            Calc_Variables[i,1] = Wet_Bulb_Stull(Buoy_yearly_pd['ATMP'][i],Calc_Variables[i,0])

                        Buoy_yearly_pd["RELH"] = pd.DataFrame(Calc_Variables[:,0])
                        Buoy_yearly_pd["WBTM"] = pd.DataFrame(Calc_Variables[:,1])
                        if len(Buoy_yearly_pd)>84: print("Houston, we have a problem")
                        print("Saving processed data from station "+Buoy_station['STATION_ID'][ndbc_station]+' for '+str(Initialization_Year))
                        Buoy_yearly_pd.to_csv(str(Initialization_Year)+"_"+Buoy_station['STATION_ID'][ndbc_station]+".txt", sep=',',index=False)
                else: "" #print("No Data for Time Window")
    print("Concating event .txt files")
    outfilename = "NDBC_Obs.csv"
    with open(outfilename, 'wb') as outfile:
        for filename in glob.glob('*.txt'):
            if filename == outfilename:
                        # don't want to copy the output into the output
                continue
            with open(filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, outfile)
