#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib        as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from math import atan
from scipy import stats
from sklearn.metrics import mean_squared_error
from datetime import datetime
from datetime import timedelta
import dateutil.parser
from matplotlib import cm
import numpy as np
import pandas as pd
###SED REPLACING
PYTHON_ARRAY=SED_REPLACE
obs_path = PYTHON_ARRAY[0]
wrf_path = PYTHON_ARRAY[1]
upp_path = PYTHON_ARRAY[2]

### CALCULATIONS
################################################################################################################
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
### DATA PROCESSING
################################################################################################################
def YYYYMMDDHHMM_string(Datetime_Value):
    if Datetime_Value.month>9:Month_str = str(Datetime_Value.month)
    else:Month_str = "0"+str(Datetime_Value.month)   
    if Datetime_Value.day>9:Day_str = str(Datetime_Value.day)
    else:Day_str = "0"+str(Datetime_Value.day)
    if Datetime_Value.hour>9:Hour_str = str(Datetime_Value.hour)
    else:Hour_str = "0"+str(Datetime_Value.hour)
    if Datetime_Value.minute>9:Minute_str = str(Datetime_Value.minute)
    else: Minute_str = "0"+str(Datetime_Value.minute)
    return (str(Datetime_Value.year)+Month_str+Day_str+Hour_str+Minute_str)
### HEAT SCATTERPLOT FUNCTION
################################################################################################################
c_min  = 1
c_max  = 1000
line_x = np.arange(-1000,10000,10)
LWIDTH = 2
trp    = 0.6
RTT    = 25.
def Heat_bin_plots(MINXY,MAXXY,INCR,Predictor,Observed,c_min,c_max,xlabel_log,ylabel_log,
                   title_log,yticks_log,NWP_Models,Y_Labels,alphabet_idx,WD_option):
    bins    = (np.arange(MINXY,MAXXY+INCR,step=INCR),np.arange(MINXY,MAXXY+INCR,step=INCR))
    #Removing nan
    X = Observed
    Y = Predictor
    var_nan = np.transpose(np.vstack((X,Y)))          #
    var = var_nan[~np.isnan(var_nan).any(axis=1)]     #
    X = var[:,0]                                      #
    Y = var[:,1]                                      #
    img = plt.hist2d(X, Y,norm=mpl.colors.LogNorm(), bins=bins, cmin = 1,cmap=plt.cm.jet)
    plt.plot(line_x, line_x,color='black',linewidth=LWIDTH);
    #Statistics
    N    = str(len(Y))
    #Mean Bias
    BIAS = np.mean(Y-X)
    #Fit/slope intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    line_y = slope*line_x + intercept
    if intercept<0.001:
        intercept = "{:.2e}".format(intercept)
    else:
        intercept = str(round(intercept,2))
    #RMSE
    RMSE = mean_squared_error(X, Y)**0.5
    #CRMSE
    CRMSE = (RMSE**2-BIAS**2)**0.5
    # imaginary vs real
    if BIAS>RMSE:
        CRMSE = CRMSE.imag
    elif BIAS<RMSE:
        CRMSE = CRMSE.real
    elif BIAS==RMSE:
        CRMSE = 0
    # scientific vs float
    if CRMSE<0.001:
        CRMSE = "{:.2e}".format(CRMSE)
    else:
        CRMSE = str(round(CRMSE,2))
    if BIAS<0.001:
        BIAS = "{:.2e}".format(BIAS)
    else:
        BIAS = str(round(BIAS,2)) 
    if RMSE<0.001:
        RMSE = "{:.2e}".format(RMSE)
    else:
        RMSE = str(round(RMSE,2))
    #Plotting on image
    plt.plot(line_x, line_y,color='gray',linestyle='--',linewidth=LWIDTH)
    if WD_option==0:
        plt.text(0.05, 0.78,alphabet_idx+'\n' "Cor. C. = "+str(round(r_value,2))+'\n' "y = "+str(round(slope,2))+
                 "*x"+" + "+intercept+'\n' "BIAS = "+BIAS+'\n' "RMSE = "+RMSE+'\n' "CRMSE = "+CRMSE+'\n' "N Obs. = "+N,
                 va='center', transform=ax.transAxes, fontsize = 12, color='black',
                bbox=dict(facecolor='white', alpha=trp,edgecolor='white', boxstyle='square,pad=0.20'))
    else:
        plt.text(0.05, 0.85,alphabet_idx+'\n' "Cor. C. = "+str(round(r_value,2))+'\n' "y = "+str(round(slope,2))+"*x"+" + "+intercept+
                 '\n' "N Obs. = "+N,
                 va='center', transform=ax.transAxes, fontsize = 12, color='black',
                bbox=dict(facecolor='white', alpha=trp+0.2,edgecolor='white', boxstyle='square,pad=0.20'))
    #Labeling details
    if title_log==1:
        ax.set_title(NWP_Models, fontsize = titlesize)
    if xlabel_log==1:
        ax.set_xlabel("Obs.", fontsize = titlesize)
    if ylabel_log==1:
        ax.set_ylabel(Y_Labels, fontsize = titlesize)
    if yticks_log==0:
        ax.set_yticklabels([])   
    #Axis and grid options
    plt.xticks(rotation=RTT)
    ax.set_xlim([MINXY,MAXXY])
    ax.set_ylim([MINXY,MAXXY])
    ax.tick_params(axis='both',direction='in')
    plt.grid(b=None, which='major', axis='both',linestyle=':')
    plt.clim(c_min,c_max)
###filter data   
################################################################################################################
def remove_nan(Observed,Predictor):
    X = Observed
    Y = Predictor
    var_nan = np.transpose(np.vstack((X,Y)))
    var = var_nan[~np.isnan(var_nan).any(axis=1)]
    return (var)
#### Dumby data for colorbar
################################################################################################################
x = [i for i in range(20)]
dumby_img = plt.hexbin(x, x,cmap='jet', vmin=1, vmax=c_max,mincnt=1,norm=mpl.colors.LogNorm())
#plt.close(fig=None)
##################################################################################################################################################################
###BEGIN TEMPORAL JOINING
##################################################################################################################################################################
wrf_headers           = ["WBAN","i","YYYY","MM","DD","HH","mn","i_1","HGT_2","HGT_10","MSLP","i_2","SSTSK","SST","T_ETA_0","T_ETA_1","T_ETA_2","T_ETA_3","T_ETA_4","RH_ETA_0","RH_ETA_1","RH_ETA_2","RH_ETA_3","RH_ETA_4","HGT_ETA_MASS_1","HGT_ETA_MASS_2","HGT_ETA_MASS_3","HGT_ETA_MASS_4","TD_ETA_0","TD_ETA_1","TD_ETA_2","TD_ETA_3","TD_ETA_4","TW_ETA_0","TW_ETA_1","TW_ETA_2","TW_ETA_3","TW_ETA_4","WS_ETA_0","WS_ETA_1","WS_ETA_2","WS_ETA_3","WS_ETA_4","WD_ETA_0","WD_ETA_1","WD_ETA_2","WD_ETA_3","WD_ETA_4"]
wrfout                = pd.read_csv(wrf_path,names=wrf_headers,index_col=False)
Concat                = np.empty((len(wrfout), 1), dtype=object)
wrfout["Join_Array"]  = ""
for i in range(len(Concat)):
    NWP_dt            = datetime(wrfout["YYYY"][i],wrfout["MM"][i],wrfout["DD"][i],wrfout["HH"][i],wrfout["mn"][i],0)
    NWP_dt            = YYYYMMDDHHMM_string(NWP_dt)
    Concat[i,0]       = str(wrfout['WBAN'][i]) + "_"+NWP_dt
wrfout["Join_Array"]  = pd.DataFrame(Concat)
################################################################################
upp_head              = ['WBAN','WG_UPP','MM','DD','YYYY','HH','mn','DHH']
postprd               = pd.read_csv(upp_path,names=upp_head)
Concat                = np.empty((len(postprd), 1), dtype=object)
postprd["Join_Array"] = ""
for i in range(len(Concat)):
    UPP_Time_Valid          = datetime(postprd["YYYY"][i],postprd["MM"][i],postprd["DD"][i],postprd["HH"][i],postprd["mn"][i],0)+timedelta(hours=int(postprd['DHH'][i]))
    NWP_dt                  = YYYYMMDDHHMM_string(UPP_Time_Valid)
    Concat[i,0]             = str(postprd['WBAN'][i]) + "_"+NWP_dt
postprd["Join_Array"]       = pd.DataFrame(Concat)
################################################################################
NDBC_Ob              = pd.read_csv(obs_path)
Concat               = np.empty((len(NDBC_Ob), 1), dtype=object)
NDBC_Ob["Join_Array"] = ""
for i in range(len(Concat)):
    NDBC_WBAN_station = str(NDBC_Ob['Station'][i]) + "_"+str(NDBC_Ob['Date_Valid_Hour'][i])
    Concat[i,0]      = NDBC_WBAN_station.replace(".0","")
NDBC_Ob["Join_Array"] = pd.DataFrame(Concat)
################################################################################
Heights      = pd.read_csv("./Templates/Buoy_Staions.csv")
Joined_Array = NDBC_Ob.merge(wrfout,on='Join_Array').merge(postprd,on='Join_Array').merge(Heights,left_on='Station',right_on='STATION_ID')
Joined_Array.to_csv("NDBC_joined.csv", sep=',',index=False)
##################################################################################################################################################################
###BEGIN VERTICAL INTERPOLATION
##################################################################################################################################################################
Model   = "WRF413"
#linear interpolation
def findXPoint(xa,xb,ya,yb,yc):
    m = (xa - xb) / (ya - yb)
    xc = (yc - yb) * m + xb
    return (xc)
def windpowerlaw_00(WS_nwp,Z_anem,Z_nwp):
    WS_anem = WS_nwp*(Z_anem/Z_nwp)**0.11
    return (WS_anem)
########
NWP_HGT       = Joined_Array[['HGT_2','HGT_ETA_MASS_1','HGT_ETA_MASS_2','HGT_ETA_MASS_3','HGT_ETA_MASS_4']]
NWP_Labels    = ['T_ETA_']
OBS_Lables    = ['ATMP']
#################################################################################
### For temperature if station < model 1 layer, use model 1 layer, else interpolate
### b/n the layer above and below using linear interpolation
#################################################################################
Therm_T_readings = np.empty([len(Joined_Array),3], dtype=object)
for i in range(1):#len(Evaluation_Matrix)): #variable
    NWP_Head = []
    for z in range(5):
        NWP_Head.append(NWP_Labels[i]+str(z))
    for j in range(len(Therm_T_readings)):                      #station
        OBS_V     = float(Joined_Array[OBS_Lables[i]][j]) #station variable
        OBZ_Z     = Joined_Array['Temp_e'][j] #station variable
        NWP_Z     = np.asarray(Joined_Array.iloc[j][['HGT_2','HGT_ETA_MASS_1','HGT_ETA_MASS_2',
                                                     'HGT_ETA_MASS_3','HGT_ETA_MASS_4']])
        Stat_diff = abs(NWP_Z-OBZ_Z)
        ##for WRF [below] 
        NWP_Z_log_1 = np.argmin(Stat_diff)
        NWP_H_log_1 = NWP_Head[NWP_Z_log_1]
        if NWP_Z_log_1==0:
            NWP_V = Joined_Array[NWP_H_log_1][j]
        else:
            Stat_diff[NWP_Z_log_1]=np.nan
            NWP_Z_log_2 = np.argmin(Stat_diff)
            NWP_H_log_2 = NWP_Head[NWP_Z_log_2]
            y1    = NWP_Z[NWP_Z_log_1]
            y2    = NWP_Z[NWP_Z_log_2]
            x1    = Joined_Array[NWP_H_log_2][j]
            x2    = Joined_Array[NWP_H_log_1][j]
            qy    = OBZ_Z
            NWP_V = round(findXPoint(x1,x2,y1,y2,qy),2)
        Therm_T_readings[j,:] = OBS_V,NWP_V,OBZ_Z
        ##for WRF [above] 
#################################################################################
### For dp temperature if station < model 1 layer, use model 1 layer, else interpolate
### b/n the layer above and below using linear interpolation
#################################################################################
NWP_Labels    = ['TD_ETA_']
OBS_Lables    = ['DEWP']
Therm_TD_readings = np.empty([len(Joined_Array),3], dtype=object)
for i in range(1):                                               #variable
    NWP_Head = []
    for z in range(5):
        NWP_Head.append(NWP_Labels[i]+str(z))
    for j in range(len(Therm_TD_readings)):                      #station
        OBS_V     = float(Joined_Array[OBS_Lables[i]][j])        #station variable
        OBZ_Z     = Joined_Array['Temp_e'][j]                    #station variable
        NWP_Z     = np.asarray(Joined_Array.iloc[j][['HGT_2','HGT_ETA_MASS_1','HGT_ETA_MASS_2',
                                                     'HGT_ETA_MASS_3','HGT_ETA_MASS_4']])
        
        Stat_diff = abs(NWP_Z-OBZ_Z)        
        ##for WRF [below]
        NWP_Z_log_1 = np.argmin(Stat_diff)
        NWP_H_log_1 = NWP_Head[NWP_Z_log_1]
        if NWP_Z_log_1==0:
            NWP_V = Joined_Array[NWP_H_log_1][j]
        else:
            Stat_diff[NWP_Z_log_1]=np.nan
            NWP_Z_log_2 = np.argmin(Stat_diff)
            NWP_H_log_2 = NWP_Head[NWP_Z_log_2]
            y1    = NWP_Z[NWP_Z_log_1]
            y2    = NWP_Z[NWP_Z_log_2]
            x1    = Joined_Array[NWP_H_log_2][j]
            x2    = Joined_Array[NWP_H_log_1][j]
            qy    = OBZ_Z
            NWP_V = round(findXPoint(x1,x2,y1,y2,qy),2)
        Therm_TD_readings[j,:] = OBS_V,NWP_V,OBZ_Z
        ##for WRF [above] 
#################################################################################
### For Wind Speed if station < model 1 layer, use model 1 layer, else interpolate
### b/n the layer above and below using the wind power law
#################################################################################
NWP_HGT       = Joined_Array[['HGT_10','HGT_ETA_MASS_1','HGT_ETA_MASS_2','HGT_ETA_MASS_3','HGT_ETA_MASS_4']]
NWP_Labels    = ['WS_ETA_']
OBS_Lables    = ['WSPD']
Wind_readings  = np.empty([len(Joined_Array),3], dtype=object)
for i in range(1):#len(Evaluation_Matrix)): #variable
    NWP_Head = []
    for z in range(5):
        NWP_Head.append(NWP_Labels[i]+str(z))
    for j in range(len(Wind_readings)):                      #station
        OBS_V     = float(Joined_Array[OBS_Lables[i]][j]) #station variable
        OBZ_Z     = Joined_Array['Anem_h'][j] #station variable
        NWP_Z     = np.asarray(Joined_Array.iloc[j][['HGT_10','HGT_ETA_MASS_1','HGT_ETA_MASS_2',
                                                     'HGT_ETA_MASS_3','HGT_ETA_MASS_4']])
        Stat_diff = abs(NWP_Z-OBZ_Z)
        NWP_Z_log_1 = np.argmin(Stat_diff)
        NWP_H_log_1 = NWP_Head[NWP_Z_log_1]
        ####
        if NWP_Z_log_1==0: #closest pair is below 10m [power law from 10m to station height]
            Stat_diff   = abs(NWP_Z-OBZ_Z)                #array of distance to station
            NWP_Z_log_2 = np.argmin(Stat_diff)            #closest level
            NWP_H_log_2 = NWP_Head[NWP_Z_log_2]           #header of closest station [HGT_10,GHT_ETA_MASS_1, etc..]
            NWP_u2      = Joined_Array[NWP_H_log_2][j]    #wind speed of closest eta level
            NWP_z2      = NWP_Z[NWP_Z_log_2]              #height of closest eta level [should be 10m in this case]
            NWP_V = round(windpowerlaw_00(NWP_u2,OBZ_Z,NWP_z2),2) #power law
        elif (NWP_Z_log_1==1) & (NWP_Z[0]>NWP_Z[1]): #closest pair is at the first eta layer, but above 10m [power law from eta 1 to station height]
            Stat_diff   = abs(NWP_Z-OBZ_Z)                #array of distance to station
            NWP_Z_log_2 = np.argmin(Stat_diff)            #closest level
            NWP_H_log_2 = NWP_Head[NWP_Z_log_2]           #header of closest station [HGT_10,GHT_ETA_MASS_1, etc..]
            NWP_u2      = Joined_Array[NWP_H_log_2][j]    #wind speed of closest eta level
            NWP_z2      = NWP_Z[NWP_Z_log_2]              #height of closest eta level [should be 10m in this case]
            NWP_V = round(windpowerlaw_00(NWP_u2,OBZ_Z,NWP_z2),2) #power law            
        else: #closest pair is between the first and second eta layers [linear interpolation]
            Stat_diff[NWP_Z_log_1]=np.nan
            NWP_Z_log_2 = np.argmin(Stat_diff)
            NWP_H_log_2 = NWP_Head[NWP_Z_log_2]
            y1    = NWP_Z[NWP_Z_log_1]
            y2    = NWP_Z[NWP_Z_log_2]
            x1    = Joined_Array[NWP_H_log_2][j]
            x2    = Joined_Array[NWP_H_log_1][j]
            qy    = OBZ_Z
            NWP_V = round(findXPoint(x1,x2,y1,y2,qy),2)            
        Wind_readings[j,:] = OBS_V,NWP_V,OBZ_Z
#################################################################################
### For Wind Direction if station < model 1 layer, use model 1 layer, else index 
### the lowest model level
#################################################################################
NWP_Labels    = ['WD_ETA_']
OBS_Lables    = ['WDIR']
Wind_dir_readings  = np.empty([len(Joined_Array),3], dtype=object)
for i in range(1):#len(Evaluation_Matrix)): #variable
    NWP_Head = []
    for z in range(5):
        NWP_Head.append(NWP_Labels[i]+str(z))
    for j in range(len(Wind_dir_readings)):                      #station
        OBS_V     = float(Joined_Array[OBS_Lables[i]][j]) #station variable
        OBZ_Z     = Joined_Array['Anem_h'][j] #station variable
        NWP_Z     = np.asarray(Joined_Array.iloc[j][['HGT_10','HGT_ETA_MASS_1','HGT_ETA_MASS_2',
                                                     'HGT_ETA_MASS_3','HGT_ETA_MASS_4']])
        Stat_diff = abs(NWP_Z-OBZ_Z)
        NWP_Z_log_1 = np.argmin(Stat_diff)
        NWP_H_log_1 = NWP_Head[NWP_Z_log_1]
        NWP_V = Joined_Array[NWP_H_log_1][j]
        Wind_dir_readings[j,:] = OBS_V,NWP_V,OBZ_Z
#################################################################################
#RH and Tw matrix
RH_TW_NWP  = np.empty([len(Joined_Array),5], dtype=object)
#convert to celsius if necessary
if Therm_T_readings[0,1]>150: Therm_T_readings[:,1] = Therm_T_readings[:,1]-273.15
if Therm_TD_readings[0,1]>150: Therm_TD_readings[:,1] = Therm_TD_readings[:,1]-273.15
#calculate RH from interpolated t and td NWP values. fix >100 values
RH_TW_NWP[:,1] = Relative_Humidity(Therm_T_readings[:,1],Therm_TD_readings[:,1])
RH_TW_NWP[RH_TW_NWP[:,1]>100]=100
#index RH observations and fix if necessary
RH_TW_NWP[:,0] = Joined_Array['RELH']
RH_TW_NWP[RH_TW_NWP[:,0].astype(float)>100]=100
#calculate wet-bulb temperature
for j in range(len(RH_TW_NWP)):
    RH_TW_NWP[j,3] = Wet_Bulb_Stull(Therm_T_readings[j,1],RH_TW_NWP[j,1])
#index TW obs and height of observation
RH_TW_NWP[:,2] = Joined_Array['WBTM']
RH_TW_NWP[:,4] = Joined_Array['Temp_e']
#################################################################################
### For Wind Gust and MSLP
#################################################################################
Wind_gust = np.asarray(Joined_Array[['GST','WG_UPP','Anem_h']])
MSLP      = np.asarray(Joined_Array[['PRES','MSLP']])
Headahs = ['MSLP_o','MSLP_f_'+Model,'WG_o','WG_f_UPP','Anem_h','WD_o','WD_f_'+Model,'Anem_h',
           'WS_o','WS_f_'+Model,'Anem_h','TD_o','TD_f_'+Model,
           'Therm_h','T_o','T_f_'+Model,'Therm_h',
           'RH_o','RH_f_'+Model,'TW_o','TW_f_'+Model,'Temp_e']
Buoy_np = np.hstack((MSLP,Wind_gust, Wind_dir_readings,Wind_readings,Therm_TD_readings,Therm_T_readings,RH_TW_NWP)) 
Buoy_pd = pd.DataFrame(Buoy_np)
for col in range(22):
    Buoy_pd.rename(columns={col: Headahs[col]},inplace = True)
Buoy_pd['WBAN']       = Joined_Array['Station']
Buoy_pd['Lat']        = Joined_Array['Lat']
Buoy_pd['Lon']        = Joined_Array['Lon']
Buoy_pd['Join_Array'] = Joined_Array['Join_Array']
Buoy_pd.to_csv("NDBC_Join.csv",index=False)
##################################################################################################################################################################
###BEGIN PLOTTING
##################################################################################################################################################################
#MSLP, WD  , WG
#T<10,TD<10,WS<10
#T>10,TD>10,WS>10
#Buoy_pd['MSLP_f_'+Model] = Buoy_pd['MSLP_f_'+Model]/100
Titles = [["Mean Sea Level Pressure [mb]","Wind Direction [deg.]","Wind Gust [m/s]"],
          ["Temperature [$^\circ$C]"  ,"Dew Point Temperature [$^\circ$C]","Wind Speed [m/s]"],
          ["Temperature [$^\circ$C]"  ,"Dew Point Temperature [$^\circ$C]","Wind Speed [m/s]"]]
OBS = [["MSLP_o" ,"WD_o" ,"WG_o" ],
       ["T_o"    ,"TD_o" ,"WS_o" ],
       ["T_o"    ,"TD_o" ,"WS_o" ]]
NWP =  [["MSLP_f_"+Model ,"WD_f_"+Model ,"WG_f_UPP" ],
       ["T_f_"+Model     ,"TD_f_"+Model ,"WS_f_"+Model ],
       ["T_f_"+Model     ,"TD_f_"+Model ,"WS_f_"+Model ]]
STHGT =  [[""        ,""        ,""       ],
          ["Therm_h" ,"Therm_h" ,"Anem_h" ],
          ["Therm_h" ,"Therm_h" ,"Anem_h" ]]
Y_Labels  = ["pred.", "pred."'\n'"Buoy<10m", "pred."'\n'"Buoy>10m"]
XYMIN     = [[920 ,   0,0 ],
             [-40 , -40,0 ],
             [-40 , -40,0 ]]
XYMAX     = [[1040 , 360, 40],
             [40   ,  40, 40],
             [40   ,  40, 40]]
INCR      = [[1  , 10 , 0.5],
             [0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5]]
fontszt   = 12
titlesize = 16
fontsz = 16
alphabet_idx = [["A","B","C"],
                ["D","E","F"],
                ["G","H","I"]]
WD_option  = [[0,1,0],
              [0,0,0],
              [0,0,0]] 
xlabel_log = [[0,0,0],
              [0,0,0],
              [1,1,1]]
ylabel_log = [[1,0,0],
              [1,0,0],
              [1,0,0]]
title_log  = [[1,1,1],
              [1,1,1],
              [0,0,0]]
yticks_log = [[1,1,1],
              [1,1,1],
              [1,1,1]]
Plot_Seq   = [[1,2,3],
              [4,5,6],
              [7,8,9]]
fig = plt.figure(figsize=(5*3,5*3))
for x in range(3):
    for y in range(3): # variables == row
        SUBPLOT = Plot_Seq[x][y]
        ax = fig.add_subplot(3,3,SUBPLOT)
        #ax = plt.gca()
        if SUBPLOT<4:
            Heat_bin_plots(XYMIN[x][y],XYMAX[x][y],INCR[x][y],Buoy_pd[NWP[x][y]].astype(float),Buoy_pd[OBS[x][y]].astype(float),
                       c_min,c_max,xlabel_log[x][y],ylabel_log[x][y],title_log[x][y],yticks_log[x][y],
                      Titles[x][y],Y_Labels[x],alphabet_idx[x][y],WD_option[x][y])
        elif (SUBPLOT == 4) or (SUBPLOT == 5) or (SUBPLOT == 6): #Station < 10
            PLT_ARRAY = np.asarray(Buoy_pd[[OBS[x][y],NWP[x][y],STHGT[x][y]]])
            BL  = PLT_ARRAY[PLT_ARRAY[:,2]<10]
            NWP_p = BL[:,1].astype(float)
            OBS_p = BL[:,0].astype(float)
            Heat_bin_plots(XYMIN[x][y],XYMAX[x][y],INCR[x][y],NWP_p,OBS_p,
                       c_min,c_max,xlabel_log[x][y],ylabel_log[x][y],title_log[x][y],yticks_log[x][y],
                      Titles[x][y],Y_Labels[x],alphabet_idx[x][y],WD_option[x][y])
        else: #TEMP > 10
            PLT_ARRAY = np.asarray(Buoy_pd[[OBS[x][y],NWP[x][y],STHGT[x][y]]])
            AB  = PLT_ARRAY[PLT_ARRAY[:,2]>10]
            NWP_p = AB[:,1].astype(float)
            OBS_p = AB[:,0].astype(float)
            Heat_bin_plots(XYMIN[x][y],XYMAX[x][y],INCR[x][y],NWP_p,OBS_p,
                       c_min,c_max,xlabel_log[x][y],ylabel_log[x][y],title_log[x][y],yticks_log[x][y],
                      Titles[x][y],Y_Labels[x],alphabet_idx[x][y],WD_option[x][y])
fig.subplots_adjust(bottom=0.12, top=0.97, left=0.08, right=0.95,wspace=0.15, hspace=0.18)
#[left, bottom, width, height],
cb_ax = fig.add_axes([0.08, 0.05,0.86, 0.01])
cbar = fig.colorbar(dumby_img, cax=cb_ax,orientation="horizontal",format='%4.0f')
cbar.set_label('Bin Count',fontsize = 14)
plt.savefig("NDBC_Plot.png")


