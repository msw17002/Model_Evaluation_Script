#!/usr/bin/env python
# coding: utf-8

# Assuming you have your Analysis stations
from itertools import compress
from datetime import timedelta
from datetime import datetime
import numpy.matlib
import dateutil.parser
import numpy as np
import pandas as pd
from math import atan
#import urllib.request

# delete these
import glob
import shutil
import os
from os import listdir
import sys

#out_csv_name = "2020010112_2020010312.csv"
######################################################################################
###SIMULATION CONSTANTS
######################################################################################
Time_diff_limit_mins     = 5
ISD_dir                  = r'https://www.ncei.noaa.gov/data/global-hourly/access/'
#Events_List              = pd.read_csv(r'./Evaluation_Time_Info.csv')
ISD_inventory            = './ISD_CSV_Files/'
Spinup                   = int(0)                                        # ADD A SPINUP TIME... SET TO 0 IF NO SPINUP TIME
minutes_round            = 10
now                      = datetime.now()
now                      = now.year
#print("Obtaining ISD Observations from: "+Station_Path+"WRF381_Stations.csv")
Station_Path             = '/shared/stormcenter/AirMG/AirMG_Group/ISD/Templates/'
ISD_station_csv          = pd.read_csv(Station_Path+"WRF381_Stations.csv")
#ISD_station_csv.rename(columns={0:'File_Name',1:'Lat',2:'Lon'}, inplace=True)
#File_Name,Lat,Lon
Spinup                   = int(0)                                        # ADD A SPINUP TIME... SET TO 0 IF NO SPINUP TIME

######################################################################################
###START TIME AND END TIME FOR OBSERVATION WEBSCRAPE
####Start_YYYY  Start_MM        Start_DD        Start_HH        End_YYYY        End_MM  End_DD  End_HH
######################################################################################
PYTHON_SED_STRING=TEN_FIELDS
out_name      = str(PYTHON_SED_STRING[0])+"_"+str(PYTHON_SED_STRING[1])
out_csv_name  = out_name+".csv"
ISD_directory = out_name
#Start time of evaluation
Initialization_Year  = int(PYTHON_SED_STRING[2])
Initialization_Month = int(PYTHON_SED_STRING[3])
Initialization_Day   = int(PYTHON_SED_STRING[4])
Initialization_Hour  = int(PYTHON_SED_STRING[5])
#End time of evaluation
Termination_Year  = int(PYTHON_SED_STRING[6])
Termination_Month = int(PYTHON_SED_STRING[7])
Termination_Day   = int(PYTHON_SED_STRING[8])
Termination_Hour  = int(PYTHON_SED_STRING[9])
### Obtain the Start and End Times of the Simulation
Datetime_Initialization  = datetime(Initialization_Year, Initialization_Month, Initialization_Day,Initialization_Hour,0,0)
Datetime_Termination     = datetime(Termination_Year   ,    Termination_Month,    Termination_Day,   Termination_Hour,0,0)

# "In addition, the sensor’s starting threshold for response to wind direction and wind speed is 2 knots. 
# Winds measured at 2-knots or less are reported as calm."
WS_TH = 2*0.514444
# The minimum gust speed reported by ASOS is 14 knots.
WG_TH = 14*0.514444
##########################################################################################
# Find all .txt files in working directory
##########################################################################################
def find_files(path_to_dir,suffix):
        filenames=listdir(path_to_dir)
        return [filename for filename in filenames if filename.endswith(suffix)]

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
# Round the observation's time to determine hourly observations
##########################################################################################
def roundTime(dt=None, roundTo=0):
    if dt == None : dt = datetime.datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + timedelta(0,rounding-seconds,-dt.microsecond)
##########################################################################################
# Data conversions
##########################################################################################
def Temperature_Delim_Calc_C(Temperature_Cell): #TEMPERATURE
    T2_C = np.nan
    if len(Temperature_Cell["TMP"])>0: # SFC TEMPERATURE
        Temp_Obs = Temperature_Cell["TMP"].split(',')
        if Temp_Obs[1] == '0' or Temp_Obs[1] == '1' or Temp_Obs[1] == '4' or Temp_Obs[1] == '5' or Temp_Obs[1] == '9' or Temp_Obs[1].isalpha():
            T2_C = float(Temp_Obs[0])/10 #celsius
            if T2_C > 100:
                T2_C = np.nan
    return (T2_C)

def DewPoint_Delim_Calc_C(DewPoint_Cell): #DEW POINT TEMPERATURE
    DP2_C = np.nan
    if len(DewPoint_Cell["DEW"])>0:
        DP_Obs = DewPoint_Cell["DEW"].split(',')
        if DP_Obs[1] == '0' or DP_Obs[1] == '1' or DP_Obs[1] == '4' or DP_Obs[1] == '5' or DP_Obs[1] == '9' or DP_Obs[1].isalpha():
            DP2_C = float(DP_Obs[0])/10
            if DP2_C > 100:
                DP2_C = np.nan
    return (DP2_C)

def WindGust_Delim_Calc_ms(WindGust_Cell): #WIND GUST
    WG_ms = np.nan
    if len(WindGust_Cell["OC1"])>0: 
        WG_Obs = WindGust_Cell["OC1"].split(',')
        if WG_Obs[1] == '0' or WG_Obs[1] == '1' or WG_Obs[1] == '4' or WG_Obs[1] == '5' or WG_Obs[1] == '9' or WG_Obs[1].isalpha():
            WG_ms = float(WG_Obs[0])/10 #celsius
            if WG_ms > 100:
                WG_ms = np.nan
            elif WG_ms<WG_TH:
                WG_ms = np.nan
    return (WG_ms)

def WindSp_WindDir_Calc_ms_deg(Wind_Cell): #WIND SPEED | WIND DIRECTION
    wsms_wddeg = np.nan,np.nan
    if len(Wind_Cell["WND"])>0:
        Wind_Obs = Wind_Cell["WND"].split(',')
        if Wind_Obs[1] == '0' or Wind_Obs[1] == '1' or Wind_Obs[1] == '4' or Wind_Obs[1] == '5' or Wind_Obs[1] == '9' or Wind_Obs[1].isalpha():
            if Wind_Obs[2] == "N":
                wsms_wddeg = float(Wind_Obs[3])/10,float(Wind_Obs[0]) #degrees 
            elif Wind_Obs[2] == "C" or Wind_Obs[2] == '9':
                wsms_wddeg = np.nan,np.nan
        if wsms_wddeg[0]<WS_TH:
            wsms_wddeg = np.nan,np.nan
        elif wsms_wddeg[0]>100:
            wsms_wddeg = np.nan,np.nan
        elif wsms_wddeg[1]>361:
            wsms_wddeg = np.nan,np.nan
    return (wsms_wddeg) 

def StationPres_Alt_Calc_hpa(Pressure_Cell): #STATION PRESSURE
    SP_hpa = np.nan
    if len(Pressure_Cell["MA1"])>0:
        Pressure = Pressure_Cell["MA1"].split(',')
        if Pressure[3] == '0' or Pressure[3] == '1' or Pressure[3] == '4' or Pressure[3] == '5' or Pressure[3] == '9' or Pressure[3].isalpha():
            SP_hpa = (float(Pressure[2])/10)*100 
            if SP_hpa > 200000:
                SP_hpa = np.nan
    return (SP_hpa)    

def Visibility_meters(Visibility_Cell):
    VIS_m = np.nan
    if len(Visibility_Cell["VIS"])>0:
        Visibility = Visibility_Cell["VIS"].split(',')
        if Visibility[1] == '0' or Visibility[1] == '1' or Visibility[1] == '4' or Visibility[1] == '5' or Visibility[1] == '9' or Visibility[1].isalpha():
            VIS_m = float(Visibility[0])
            if VIS_m == 999999:
                VIS_m = np.nan
    return (VIS_m)    

##########################################################################################
# CALCULATIONS
##########################################################################################

def Specific_Humidity(Dew_Temperature_C,Station_Pressure_pat):
    e = 6.11*10**((7.5*Dew_Temperature_C)/(237.3+Dew_Temperature_C))
    rv = (0.622*e)/(Station_Pressure_pat+e)
    sh = rv/(1+rv)*100
    return (sh)

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

####################################################################################################################################################
###MAIN SCRIPT
####################################################################################################################################################
### End times
if Datetime_Termination.month>9:Month_str = str(Datetime_Termination.month)
else:Month_str = "0"+str(Datetime_Termination.month)   
if Datetime_Termination.day>9:Day_str = str(Datetime_Termination.day)
else:Day_str = "0"+str(Datetime_Termination.day)
str_end = str(Datetime_Termination.year)+Month_str+Day_str
### Start times
if Datetime_Initialization.month>9:Month_str = str(Datetime_Initialization.month)
else:Month_str = "0"+str(Datetime_Initialization.month)   
if Datetime_Initialization.day>9:Day_str = str(Datetime_Initialization.day)
else:Day_str = "0"+str(Datetime_Initialization.day)
str_begin = str(Datetime_Initialization.year)+Month_str+Day_str
print("Extracting ISD observations from "+str_begin+" to "+str_end)
### Year iteration
Year_it_list = [str(Datetime_Termination.year),str(Datetime_Initialization.year)]
Year_it      = len(np.unique(Year_it_list))
print("Iterating through: "+str(Year_it)+" years")
######################################################################################
###STATION WEBSCRAPING BASED ON Year
######################################################################################
for x in range(Year_it):
    print("Year: "+Year_it_list[x])
    print("n Stations: "+str(len(ISD_station_csv)))
    for y in range(len(ISD_station_csv)): #Station loop
        #Do I already have the file?
        ########################################################################
        ###Download .txt.gz file and move it to archive directory
        ########################################################################    
        #Iterate by station
        station_file_csv        = str(ISD_station_csv["Station"][y])+".csv"
        station_file_csv_save   = str(Initialization_Year)+"_"+station_file_csv
        url                     = ISD_dir+str(Initialization_Year)+"/"+str(ISD_station_csv["Station"][y])+".csv"
#        #print(url)
#        download_log       = 0
#        #If the file doesn't exist, download it and process it--------------------------------------------------
#        if (station_file_csv_save not in find_files( ISD_inventory, suffix=".csv" )):
#            print(str(ISD_station_csv["Station"][y])+": file doesnt exist")
#            try: 
#                download_log = 1
#                print(str(ISD_station_csv["Station"][y])+": downloading")
#                urllib.request.urlretrieve(url, station_file_csv)
#                shutil.move(station_file_csv,ISD_inventory+"/"+station_file_csv_save)
#            except:
#                download_log = 0
#                print(str(ISD_station_csv["Station"][y])+": file not available")
#        #--------------------------------------------------------------------------------------------------------
#        #If the file does exist, but the year mached the current year, the file is removed then replaced with
#        #an update .tar.gz file (in case the event was recent)
#        elif (station_file_csv_save in find_files( ISD_inventory, suffix=".csv" )) & (int(Initialization_Year)==int(now)):
#            print(str(ISD_station_csv["Station"][y])+": file exists, but will be updated")
#            os.remove(ISD_inventory+station_file_csv)
#            try: 
#                download_log = 1
#                print(str(ISD_station_csv["Station"][y])+": downloading")
#                urllib.request.urlretrieve(url, station_file_csv)
#                shutil.move(station_file_csv,ISD_inventory+"/"+station_file_csv_save)
#            except: 
#                download_log = 0
#                print(str(ISD_station_csv["Station"][y])+": file not available")
#        #--------------------------------------------------------------------------------------------------------
#        #The file already exists and it is from last year. Just read its processed text file.
#        elif (station_file_csv_save in find_files( ISD_inventory, suffix=".csv" )) & (int(Initialization_Year)<int(now)):
#            download_log = 1
#            print(str(ISD_station_csv["Station"][y])+": file has already been downloaded and is from a previous year")
	download_log = 1
	try: Compressed_ISD = pd.read_csv(url,dtype=object)
        except: download_log = 0
        ######################################################################################
        ###PRE-PROCESSING DATA
        ######################################################################################   
        if download_log==1: #Is data for 'y' station available on said date?
            print("Station "+str(ISD_station_csv["Station"][y])+" is available")
            Data_Server   = str_begin+"_"+str_end+"_"+str(ISD_station_csv["Station"][y])
            ISD_Station = str(ISD_station_csv["Station"][y])
            #print("Downloading Data for Station: "+ISD_Station)
            #########################################################################################################
            ### ***SHRINK DATA*** ###
            #########################################################################################################
            # variables for analysis (shrink dataframe)
            #Compressed_ISD = pd.read_csv(ISD_inventory+station_file_csv_save,dtype=object)
            # time-window for analysis (shrink dataframe)
            start_time       = Datetime_Initialization
            start_end        = Datetime_Termination
            # datetime of observation
            datetime_station = np.expand_dims(Compressed_ISD["DATE"].to_numpy(dtype=str),1)
            # determine if the observation falls within the time-window for analysis
            row_logical      = np.zeros((len(datetime_station), 1))
            for i in range(len(row_logical)): #Is the date within the time of analysis?
                if start_time <= dateutil.parser.parse(datetime_station[i,0]) <= start_end:
                    row_logical[i,0] = True   
            # add boolean results back into the original dataframe [did the observation fall w/in the analysis time-window]
            Compressed_ISD['Date_Boolean'] = pd.DataFrame(row_logical)
            # remove 'false' boolean results
            Compressed_ISD                 = Compressed_ISD[Compressed_ISD.Date_Boolean != 0]
            # replace 'nan' values with empty strings for data-processing
            Compressed_ISD                 = Compressed_ISD.replace(np.nan, '', regex=True).reset_index(drop=True)
            # create an array filled with the station's wmo/wban identifiers
            Stations                       = np.matlib.repmat(ISD_Station, len(Compressed_ISD), 1)
            #########################################################################################################
            ### ***DETERMINE TIMES & T-DIFF*** ###
            #########################################################################################################
            # preallocate arrays to determine hourly observations (comparing vs. NWP hourly output)
            d = len(Compressed_ISD)
            Time_Matrix = np.empty([d,4], dtype=object)
            time_valid  = np.empty([d,1], dtype=object)
            Date_times  = np.empty([d,1], dtype=object)
            # determine the time difference (minutes) b/n the nearest hour vs. the observation's time,
            # return a matrix of the observation's rounded YYYY, MM, DD, HH, and TDIFF 
            for i in range(d):
                Date_times[i,0]     = dateutil.parser.parse(Compressed_ISD["DATE"][i])
                time_round          = roundTime(Date_times[i,0],roundTo=minutes_round*60) #hour_rounder(Date_times[i,0])
                time_round_precip   = Date_times[i,0].replace(microsecond=0, second=0, minute=0)
                time_valid[i,0]     = datetime.strptime(str(Date_times[i,0]), '%Y-%m-%d %H:%M:%S')
                tdiff               = abs(time_valid[i,0]-time_round)
                tdiff               = int(tdiff.seconds/60)
                Time_Matrix[i,:]    = Compressed_ISD["DATE"][i],YYYYMMDDHH_string(time_round),str(tdiff),YYYYMMDDHH_string(time_round_precip)
            Dates       = np.unique(Time_Matrix[:,1])
            row_logical = np.zeros((len(Time_Matrix[:,1]), 1))
            for z in range(len(Dates)):
                Log_Key = np.expand_dims(Time_Matrix[:,1]==Dates[z],1)
                if len(Log_Key>0):
                    row             = [i for i, j in enumerate(Log_Key) if j]
                    Key_Time_Matrix = Time_Matrix[row]
                    row_check       = np.argmin(np.array(Key_Time_Matrix[:,2], dtype=np.float), axis=None, out=None)
                    row             = row[row_check]
                    row_logical[row,0] = 1
            Compressed_ISD = pd.concat([Compressed_ISD, pd.DataFrame(Time_Matrix)], axis=1)
            if len(Compressed_ISD)>0:
                Compressed_ISD['Hourly_Boolean'] = pd.DataFrame(row_logical)
                Names_compressed = ['Date_Time_Obs','Date_Valid_Hour','Time_Diff_mins','Time_Valid_Precip']
                for col in range(4):
                    Compressed_ISD.rename(columns={col: Names_compressed[col]},inplace = True)
            if len(Compressed_ISD)>0:
                #########################################################################################################
                ### ***HOURLY OBSERVATIONS*** ###
                #########################################################################################################
                Compressed_ISD = Compressed_ISD[Compressed_ISD.Hourly_Boolean != 0].reset_index(drop=True)
                Compressed_ISD = Compressed_ISD[pd.to_numeric(Compressed_ISD['Time_Diff_mins'])<Time_diff_limit_mins]    
                ISD_DATA        = np.empty((len(Compressed_ISD), 15), dtype=object)
                #########################################################################################################
                ### ***ADD MANDATORY COLUMNS IF NOT AVAILABLE THEN SHRINK DATA*** ###
                #########################################################################################################
                # variables for analysis (shrink dataframe)
                if 'STATION' not in Compressed_ISD: Compressed_ISD['STATION'] = ''    
                if 'DATE' not in Compressed_ISD: Compressed_ISD['DATE'] = ''    
                if 'TMP' not in Compressed_ISD: Compressed_ISD['TMP'] = ''    
                if 'DEW' not in Compressed_ISD: Compressed_ISD['DEW'] = ''    
                if 'WND' not in Compressed_ISD: Compressed_ISD['WND'] = ''    
                if 'OC1' not in Compressed_ISD: Compressed_ISD['OC1'] = ''    
                if 'MA1' not in Compressed_ISD: Compressed_ISD['MA1'] = ''    
                if 'VIS' not in Compressed_ISD: Compressed_ISD['VIS'] = ''    
                if 'SLP' not in Compressed_ISD: Compressed_ISD['SLP'] = ''    
                if 'REM' not in Compressed_ISD: Compressed_ISD['REM'] = '' 
                Compressed_ISD = Compressed_ISD[["STATION","DATE","Date_Valid_Hour","Time_Diff_mins","TMP","DEW","WND","OC1",
                                                 "MA1","VIS","SLP","REM"]]
                #########################################################################################################
                ### ***CALCULATIONS*** ###
                #########################################################################################################
                for j in range(len(Compressed_ISD)):
                    # metadata for the observations that match the query time
                    ISD_tm        = Compressed_ISD.iloc[j,:]
                    Station       = ISD_tm["STATION"]
                    Date_Valid    = ISD_tm["Date_Valid_Hour"]
                    Date_Obs      = ISD_tm["DATE"]
                    Time_Diff_m   = ISD_tm["Time_Diff_mins"]
		    Join_Array    = str(Station)+"_"+str(Date_Valid)
                    T2_C          = Temperature_Delim_Calc_C(ISD_tm)                    #2-M TEMPERATURE celsius
                    Td_C          = DewPoint_Delim_Calc_C(ISD_tm)                       #2-M DEW POINT TEMPERATURE celsius
                    WSms_WDdeg    = WindSp_WindDir_Calc_ms_deg(ISD_tm)            #WIND DIRECTION degrees | WIND SPEED m/s
                    WG_ms         = WindGust_Delim_Calc_ms(ISD_tm)                     #WIND GUST m/s
                    SP_hpa        = StationPres_Alt_Calc_hpa(ISD_tm)                  #STATION PRESSURE hpa
                    Vis_m         = Visibility_meters(ISD_tm)
                    # calculate variables that are not readilly available by ISD
                    RH         = np.round(Relative_Humidity(T2_C,Td_C),2)                #RELATIVE HUMIDITY percent
                    Tw_C       = np.round(Wet_Bulb_Stull(T2_C,RH),2)                   #WET BULB TEMPERATURE celsius
                    SH         = Specific_Humidity(Td_C,SP_hpa)                          #SPECIFIC HUMIDITY kg/kg     
                    # precipitation type and sky condition
                    # PType      = Precipitation_Type(ISD_tm)
                    ISD_DATA[j,:] = np.transpose(np.expand_dims(numpy.asarray([Station,Date_Valid,Join_Array,Date_Obs,
                                                                                        Time_Diff_m,T2_C,Td_C,Tw_C,RH,SH,
                                                                                        WSms_WDdeg[1],WSms_WDdeg[0],WG_ms,SP_hpa,
                                                                                        Vis_m],dtype=object),1))            
                #Precipitation_DataFrame
                ISD_DATA_df = pd.DataFrame(ISD_DATA)
                ISD_headers = ["WBAN","Time_Valid","Join_Array","Date_Obs","Time_Diff_min","T2_C_o","Td_C_o",
                                        "Tw_C_o","RH_pct_o","SH_kgkg_o","WD_deg_o",
                                        "WS_ms_o","WG_ms_o","PSFC_pa_o","Vis_m_o"]
                for col in range(14):
                    ISD_DATA_df.rename(columns={col: ISD_headers[col]},inplace = True)
                forprint = ISD_DATA_df
                print("Saving File: "+Year_it_list[x]+"_"+Data_Server+".txt")
                np.savetxt(Year_it_list[x]+"_"+Data_Server+".txt", forprint, delimiter=",",fmt='%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s')

print("Extraction was successful")
#########################################################################################################
### CREATING ONE FILE FOR ONE EVENT, MOVING ALL FILES INTO A DIRECTORY###
######################################################################################################### 
os.mkdir(ISD_directory)
outfilename   = out_csv_name
with open(outfilename, 'wb') as outfile:
    for filename in glob.glob('*.txt'):
        if filename == outfilename:
                # don't want to copy the output into the output
            continue
        with open(filename, 'rb') as readfile:
            shutil.copyfileobj(readfile, outfile)
Text_Files_ISD = find_files('.',".txt")
for i in range(len(Text_Files_ISD)):
    shutil.move(Text_Files_ISD[i],ISD_directory)
