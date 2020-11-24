#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
from metpy.units import units
from siphon.simplewebservice.wyoming import WyomingUpperAir
import matplotlib        as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
from sklearn.metrics import mean_squared_error
import shutil
from os import listdir
import os
import time, glob
from datetime import timedelta
import pandas as pd
import numpy as np
import numpy.matlib
from math import atan
np.set_printoptions(suppress=True) 

Sounding_Stations = ['ALB','CHH','OKX'] #to add station, appen to array ''
Isobaric_Surfaces = np.expand_dims(np.arange(50, 1025, 25),1)
k_con = 0.286

###
DATE_ARRAY=SED_BASH
WorkEnv              = './'
Initialization_Year  = int(DATE_ARRAY[0])
Initialization_Month = int(DATE_ARRAY[1])
Initialization_Day   = int(DATE_ARRAY[2])
Initialization_Hour  = int(DATE_ARRAY[3])
dtime                = int(DATE_ARRAY[4]) #durration
#YYY,MM,DD,HH,DT
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
    return (str(Datetime_Value.year)+Month_str+Day_str+Hour_str)

##########################################################################################
# Create list of csv's
##########################################################################################
def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

##########################################################################################
# Calculations
##########################################################################################
def Relative_Humidity(Temperature,Dew_Temperature):
    RH_perc = 100*(np.exp((17.625*Dew_Temperature)/(243.04+Dew_Temperature))/np.exp((17.625*Temperature)/(243.04+Temperature)))
    return (np.round(RH_perc,2))

def Potential_Temperature(Temperature_iso,Pressure_sfc,Pressure_iso):
    PotT = (Temperature_iso+273.15)*(Pressure_sfc/Pressure_iso)**k_con
    return (np.round(PotT,2))
#https://www.sciencedirect.com/topics/earth-and-planetary-sciences/potential-temperature

def K_to_C(Temperature_K):
    TC = Temperature_K-273.15
    return (TC)

def Wet_Bulb_Stull(Temperature_C,RH_perc):
    C1 = 0.151977 #Stull Tw
    C2 = 8.313659 #Stull Tw 
    C3 = 1.676331 #Stull Tw
    C4 = 0.00391838 #Stull Tw
    C5 = 0.023101 #Stull Tw
    C6 = 4.686035 #Stull Tw
    tw = Temperature_C*atan(C1*(RH_perc+C2)**0.5)+atan(Temperature_C+RH_perc)-atan(RH_perc-C3)+C4*((RH_perc)**1.5)*atan(C5*RH_perc)-C6
    return (np.round(tw,2))

def Specific_Humidity(Dew_Temperature_C,Station_Pressure_pat):
    e = 6.11*10**((7.5*Dew_Temperature_C)/(237.3+Dew_Temperature_C))
    rv = (0.622*e)/(Station_Pressure_pat+e)
    sh = rv/(1+rv)*100
    return (sh)

Datetime_Initialization = datetime(Initialization_Year, Initialization_Month, Initialization_Day, Initialization_Hour)+timedelta(hours=int(12))
Sim_Start               = YYYYMMDDHH_string(Datetime_Initialization)
Datetime_Termination    = Datetime_Initialization+timedelta(hours=int(dtime))
if Datetime_Termination.month>9: Month_str = str(Datetime_Termination.month)
else: Month_str = "0"+str(Datetime_Termination.month)   
if Datetime_Termination.day>9: Day_str = str(Datetime_Termination.day)
else: Day_str = "0"+str(Datetime_Termination.day)
str_end = str(Datetime_Termination.year)+Month_str+Day_str

if Datetime_Initialization.month>9: Month_str = str(Datetime_Initialization.month)
else: Month_str = "0"+str(Datetime_Initialization.month)   
if Datetime_Initialization.day>9: Day_str = str(Datetime_Initialization.day)
else: Day_str = "0"+str(Datetime_Initialization.day)
str_begin = str(Datetime_Initialization.year)+Month_str+Day_str
CSV_print = str_begin+"_"+str_end+".csv"

delta = timedelta(hours=6)
Data_Server = np.empty([12,3], dtype=object)
for y in range(12):
    if y == 0:
        Data_Server[y,0] = Datetime_Initialization
        Data_Server[y,1] = Datetime_Initialization<=Datetime_Termination
    else:
        Data_Server[y,0] = Datetime_Initialization+delta*y
        Data_Server[y,1] = (Datetime_Initialization+delta*y)<=Datetime_Termination
    Data_Server[y,2] = (Data_Server[y,0].hour) == 0 or (Data_Server[y,0].hour == 12)
print(Data_Server)
Log = Data_Server[:,1:3]
Log = np.expand_dims(np.sum(Log,axis=1)==2,1)
a = np.expand_dims(Data_Server[:,0],1)
Loop_Dates = np.expand_dims(a[Log],1)
print("z loop")
print(Loop_Dates)
for z in range(len(Sounding_Stations)):     #len(Sounding_Stations)
    station = Sounding_Stations[z]
    for y in range(len(Loop_Dates)):
        YEAR = Loop_Dates[y,0].year
        MONTH = Loop_Dates[y,0].month
        DAY = Loop_Dates[y,0].day
        HOUR = Loop_Dates[y,0].hour
        date = datetime(YEAR,MONTH,DAY,HOUR)     
        print(station,date)
        #Web-scrape data
        try:
            df = WyomingUpperAir.request_data(date, station)
            taskbarskies = True
        except:
            taskbarskies = False
            pass # doing nothing on exception  
#deriving variables
        pressure_int  = np.empty((len(df), 1), dtype=int)
        derived_float = np.empty((len(df), 4), dtype=float)
        for i in range(len(df)):
            Psfc               = df['pressure'][0]
            pressure_int[i]    = int(df['pressure'][i])
            derived_float[i,0] = Relative_Humidity(df['temperature'][i],df['dewpoint'][i])
            derived_float[i,1] = Wet_Bulb_Stull(df['temperature'][i],derived_float[i,0])
            derived_float[i,2] = Potential_Temperature(df['temperature'][i],Psfc,df['pressure'][i])
            derived_float[i,3] = Specific_Humidity(df['temperature'][i],df['pressure'][i]*100)
        derived_float[:,0:2] = np.round(derived_float[:,0:2],2)
        derived_float        = pd.DataFrame(derived_float)
        derived_headers      = ['relh_pct','wetbulb_celcius','ptltemp_kelvin','spechum_kgkg']
        for y in range(len(derived_headers)):
            derived_float.rename(columns={y: derived_headers[y]},inplace = True)
        pressure_int = pd.DataFrame(pressure_int)
        pressure_int.rename(columns={0: 'isobaric_surface_valid'},inplace = True)
        #creating join array
        Sounding_Info   = [station,YEAR,MONTH,DAY,HOUR]
        Sounding_Info   = np.matlib.repmat(Sounding_Info, len(df), 1)
        Sim_Info        = np.matlib.repmat(Sim_Start,len(df), 1)
        Details         = pd.DataFrame(np.hstack((Sim_Info,Sounding_Info)))
        Details_headers = ['sim_start','station','YYYY_valid','MM_valid','DD_valid','HH_valid']
        for y in range(len(Details_headers)):
            Details.rename(columns={y: Details_headers[y]},inplace = True)
        #concatting dataframes
        Wyoming_data = pd.concat([Details,pressure_int,df,derived_float], axis=1)    
        if taskbarskies == True:
            #print csv file
            Wyoming_data.to_csv(station+""+str(YEAR)+str(MONTH)+str(DAY)+str(HOUR)+""+str_begin+"_"+str_end+".txt",index=False)
with open("VP_"+CSV_print, 'wb') as outfile:
    for filename in glob.glob('*.txt'):
        if filename == CSV_print:
        # don't want to copy the output into the output
            continue
        with open(filename, 'rb') as readfile:
            shutil.copyfileobj(readfile, outfile)
directory= WorkEnv
os.chdir(directory)
files=glob.glob('*.txt')
for filename in files:
    os.unlink(filename)

