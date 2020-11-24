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

def Statistical_Metrics(Obs,Predicted):
    Stats_Vals    = np.zeros((5,1))
    X             = Obs
    Y             = Predicted
    var_nan = np.transpose(np.vstack((X,Y)))
    var = var_nan[~np.isnan(var_nan).any(axis=1)]
    X = var[:,0]
    Y = var[:,1] 
    N_len = len(Y)
    if (len(X)>1) and (len(Y)>1):
        #STATISTICS
        BIAS       = np.mean(Y-X) #Mean Bias
        _, _, r_value, _, _ = stats.linregress(X,Y) #Fit/slope intercept
        RSQUARED   = r_value**2
        RMSE       = mean_squared_error(X, Y)**0.5 #RMSE
        #CRMSE
        CRMSE = (RMSE**2-BIAS**2)**0.5
        # imaginary vs real
        if BIAS>RMSE:
            CRMSE = CRMSE.imag
        elif BIAS<RMSE:
            CRMSE = CRMSE.real
        elif BIAS==RMSE:
            CRMSE = 0
        Stats_Vals[0,0] = RSQUARED
        Stats_Vals[1,0] = BIAS
        Stats_Vals[2,0] = RMSE
        Stats_Vals[3,0] = CRMSE
        Stats_Vals[4,0] = N_len
    else: 
        Stats_Vals[0,0] = np.nan
        Stats_Vals[1,0] = np.nan
        Stats_Vals[2,0] = np.nan
        Stats_Vals[3,0] = np.nan
        Stats_Vals[4,0] = np.nan
    return (Stats_Vals)

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

        Wyoming_Matrix = np.asarray(df[["pressure","temperature","dewpoint","speed","direction"]]) #hpa,C,C,kntos,degrees
        Wyoming_Matrix[:,3] = Wyoming_Matrix[:,3]*0.514444 #kts->m/s
        row_idx = np.zeros((len(Isobaric_Surfaces), 3))
        for k in range(len(row_idx)):
            Isobaric_diff = Isobaric_Surfaces[k] - Wyoming_Matrix[:,0]
            if sum(Isobaric_diff == 0) == 1:
                Log = Isobaric_diff == 0
                idx_list = [i for i, j in enumerate(Log == True) if j]
                row_idx[k,0] = idx_list[0]
            else:
                if len(Isobaric_diff[Isobaric_diff<0]) > 0:
                    Min_Log = np.max(np.expand_dims(Isobaric_diff[Isobaric_diff<0],1))
                    idx_list = [i for i, j in enumerate(Isobaric_diff == Min_Log) if j]
                    row_idx[k,1] = idx_list[0]
                else: row_idx[k,1] = np.nan
            
                if len(Isobaric_diff[Isobaric_diff>0]) > 0:
                    Max_Log = np.min(np.expand_dims(Isobaric_diff[Isobaric_diff>0],1))
                    idx_list = [i for i, j in enumerate(Isobaric_diff == Max_Log) if j]
                    row_idx[k,2] = idx_list[0]
                else: row_idx[k,2] = np.nan
            
        Indexing_mat = np.hstack((Isobaric_Surfaces,row_idx))

        Obs_mat = np.zeros((len(row_idx), 9))
        for k in range(len(row_idx)):
            if (row_idx[k,0] != 0):
                Obs_mat[k,0:5] = Wyoming_Matrix[int(row_idx[k,0])]
            elif (sum(np.isnan(row_idx[k,:])) ==1):
                Obs_mat[k,0:5] = np.nan
            elif (abs(row_idx[k,1]-row_idx[k,2])>1):
                Obs_mat[k,0:5] = np.nan
            else:
                y1 = Wyoming_Matrix[int(row_idx[k,1]),0]
                y2 = Wyoming_Matrix[int(row_idx[k,2]),0]
                qry_y = Isobaric_Surfaces[k]
                delta_y = abs(y1-qry_y)+abs(y2-qry_y)
                w1 = 1-(abs(y1-qry_y)/delta_y)
                w2 = 1-(abs(y2-qry_y)/delta_y)
                b = np.transpose(np.expand_dims(w1*Wyoming_Matrix[int(row_idx[k,1])]+w2*Wyoming_Matrix[int(row_idx[k,2])],1))
                Obs_mat[k,0:5] = b[0]
            Psfc         = Wyoming_Matrix[0,0] #mb
            if np.isnan(Psfc)==1:
                print("Oh, Noes!")
            Obs_mat[k,5] = Relative_Humidity(Obs_mat[k,1],Obs_mat[k,2]) #%
            Obs_mat[k,6] = Wet_Bulb_Stull(Obs_mat[k,1],Obs_mat[k,5])
            Obs_mat[k,7] = Potential_Temperature(Obs_mat[k,1],Psfc,Obs_mat[k,0])
            Obs_mat[k,8] = Specific_Humidity(Obs_mat[k,1],Obs_mat[k,0]*100)
            Obs_mat[:,0:7] = np.round(Obs_mat[:,0:7],2)
        Sounding_Info = [station,YEAR,MONTH,DAY,HOUR]
        Sounding_Info = np.matlib.repmat(Sounding_Info, len(Obs_mat), 1)
        Sim_Info      = np.matlib.repmat(Sim_Start,len(Obs_mat), 1)
            
        Wyoming_Interpolation = np.hstack((Sim_Info,Sounding_Info,Isobaric_Surfaces,Obs_mat))
        if taskbarskies == True:
            np.savetxt(station+"_"+str(YEAR)+str(MONTH)+str(DAY)+str(HOUR)+"_"+str_begin+"_"+str_end+".txt", Wyoming_Interpolation, delimiter=",",fmt='%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s')
#Headahs = [Sim_info,"WBAN","YYYY","MM","DD","HH","P_iso","P_into","Tiso_o","Tdiso_o","WSiso_o","WDiso_o","Rhiso_o","TOiso_o","PTiso_o","SHiso_o"]
filename = find_csv_filenames(WorkEnv, suffix=".txt")

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

