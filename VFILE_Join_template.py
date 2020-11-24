#!/usr/bin/env python
# coding: utf-8

# In[174]:

import os
import matplotlib        as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
from datetime import datetime
from datetime import timedelta
import dateutil.parser
from matplotlib import cm
import numpy as np
import pandas as pd

#JOIN_ARRAY=SED_BASH_REPLACE
Wyoming_path = "./Wyoming_vertical_profile.csv" #JOIN_ARRAY[0]
WRF_path     = "./WRF_vertical_profile.csv"  #JOIN_ARRAY[1]

##############################################################################################################
def YYYYMMDDHH_string(Datetime_Value):
    if Datetime_Value.month>9:Month_str = str(Datetime_Value.month)
    else:Month_str = "0"+str(Datetime_Value.month)   
    if Datetime_Value.day>9:Day_str = str(Datetime_Value.day)
    else:Day_str = "0"+str(Datetime_Value.day)
    if Datetime_Value.hour>9:Hour_str = str(Datetime_Value.hour)
    else:Hour_str = "0"+str(Datetime_Value.hour)
    if Datetime_Value.minute>9:Minute_str = str(Datetime_Value.minute)
    else: Minute_str = "0"+str(Datetime_Value.minute)
    return (str(Datetime_Value.year)+Month_str+Day_str+Hour_str)
################################################################################
###JOINING
################################################################################
wrf_headers           = ['WBAN','MM','DD','YYYY','Init_Hour','Init_Minute','Delta_hour','P_iso','Tiso_f','Qiso_f',
                         'WSiso_f','WDiso_f']
wrfout                = pd.read_csv(WRF_path)#Wyoming_path)#,names=wrf_headers)
print(wrfout)
Concat                = np.empty((len(wrfout), 1), dtype=object)
wrfout["Join_Array"]  = ""
for i in range(len(Concat)):
    NWP_dt            = datetime(wrfout["YYYY"][i],wrfout["MM"][i],wrfout["DD"][i],wrfout["Init_Hour"][i],
                                 wrfout["Init_Minute"][i],0)+timedelta(hours=float(wrfout["Delta_hour"][i]))
    NWP_dt            = YYYYMMDDHH_string(NWP_dt)
    Concat[i,0]       = str(wrfout['WBAN'][i]) + "_"+NWP_dt + "_" + str(wrfout["P_iso"][i])
wrfout["Join_Array"]  = pd.DataFrame(Concat)
################################################################################
Wyoming_headers      = ["Sim_info","WBAN","YYYY","MM","DD","HH","P_iso","P_into","Tiso_o","Tdiso_o","WSiso_o",
                        "WDiso_o","Rhiso_o","TOiso_o","PTiso_o","Qiso_o"]
Wyoming_Ob           = pd.read_csv(Wyoming_path)#,names=Wyoming_headers)
Concat               = np.empty((len(Wyoming_Ob), 1), dtype=object)
Date_Concat          = np.empty((len(Wyoming_Ob), 1), dtype=object)
Wyoming_Ob["Join_Array"] = ""
for i in range(len(Concat)):
    NWP_dt            = datetime(Wyoming_Ob["YYYY"][i],Wyoming_Ob["MM"][i],Wyoming_Ob["DD"][i],Wyoming_Ob["HH"][i],0,0)
    NWP_dt            = YYYYMMDDHH_string(NWP_dt)
    Concat[i,0]       = str(wrfout['WBAN'][i]) + "_"+NWP_dt + "_" + str(Wyoming_Ob["P_iso"][i])
    Date_Concat[i,0]  = NWP_dt
Wyoming_Ob["Join_Array"] = pd.DataFrame(Concat)
Wyoming_Ob["Date"]       = pd.DataFrame(Date_Concat)
################################################################################
###JOIN MATRIX
################################################################################
Joined_Array = Wyoming_Ob.merge(wrfout,on='Join_Array')
Joined_Array['Tiso_f'] = Joined_Array['Tiso_f']-273.15
Joined_Array.to_csv("Joined_VFILE.csv", sep=',',index=False)
################################################################################
###PLOTTING
################################################################################
#Unique dates for event
date_n      = Joined_Array.Date.unique()
Titles      = [date_n]
#Set the logic label arrays
ylabel      = np.zeros((1,len(date_n)))
ylabel[0,0] = 1
ylabel      = ylabel[0]
Title_log      = np.zeros((3,len(date_n)))
Title_log[0,:] = 1
SUBPLOT          = np.zeros((1,3*len(date_n)))
for i in range(3*len(date_n)):
    SUBPLOT[0,i] = int(i+1)
SUBPLOT          = SUBPLOT.reshape((3, len(date_n)))
#Set the titles 
Y_Labels       = ['Wind Speed [m/s]','Temperature [$^\circ$C]','Specific Humidity [kg/kg]']
#Set the range of plot
xmin           = [0,-60, 0]
xmax           = [80,40, 0.02]
#plot
fig = plt.figure(figsize=(5*len(date_n),5*3))
for x in range(3):
    for y in range(len(date_n)):
        print(str(x)+" "+str(y))
        #data grabbing
        Obs_h    = ['WSiso_o','Tiso_o','Qiso_o']
        Obs_CHH  = Joined_Array[(Joined_Array['Date']==date_n[y]) & (Joined_Array['WBAN_x']=='CHH')]
        Obs_CHH  = Obs_CHH.sort_values('P_iso_x')
        WRF_h    = ['WSiso_f','Tiso_f','Qiso_f']
        WRF_CHH  = Joined_Array[(Joined_Array['Date']==date_n[y]) & (Joined_Array['WBAN_x']=='CHH')]
        WRF_CHH  = WRF_CHH.sort_values('P_iso_x')
        #plotting 
        ax = fig.add_subplot(3,len(date_n),SUBPLOT[x][y])
        ax = plt.gca()
        ax.set_facecolor('white')
        ax.patch.set_edgecolor('k')  
        ax.patch.set_linewidth('2') 
        #plt.plot(WRF_ALB[WRF_h[0]],WRF_ALB['P_iso'], marker='x', color='r', linewidth=2,linestyle='dashed', label="RAMS/ICLAMS")
        plt.plot(WRF_CHH[WRF_h[x]],WRF_CHH['P_iso_x'], marker='.', color='blue', linewidth=2,linestyle='dashed', label="WRFv4.1.3")
        plt.plot(Obs_CHH[Obs_h[x]],Obs_CHH['P_iso_x'], marker='x', color='k', linewidth=2,linestyle='-', label="Obs.")
        #plt.text(0.06, 0.08,alphabet_idx[x][y],va='center', transform=ax.transAxes, fontsize = 16, color='black',
        #                bbox=dict(facecolor='white', alpha=0.6,edgecolor='white', boxstyle='square,pad=0.20'))
        #Labeling details
        if Title_log[x][y]==1:ax.set_title("KCHH: "+str(date_n[y]), fontsize = 20)
        ax.set_xlabel(Y_Labels[x], fontsize = 12)
        if ylabel[y]==1: ax.set_ylabel("Isobaric Surface [mb]", fontsize = 14)
        else: ax.set_yticklabels([]) 
        if SUBPLOT[x][y]==1: plt.legend(prop={'size': 12},ncol=1,loc='lower right',facecolor='white')
        ax.set_ylim([1000,100])
        ax.set_xlim([xmin[x],xmax[x]])
        plt.grid(b=True,which='both', axis='both',linestyle=':' , linewidth=1 , color='k', alpha=0.6,zorder=4)
        #plt.plot(WRF_OKX[WRF_h[0]],WRF_OKX['P_iso'], marker='*', color='b', linewidth=2,linestyle='dashed', label="WRFv3.8.1")
fig.savefig('Wyoming.png')


# In[ ]:




