#!/usr/bin/env python
# coding: utf-8

# In[28]:


import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.feature as cfeature 
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
##############################################################################################################
#locations of extracted csv data
JOIN_ARRAY=SED_BASH_REPLACE

output_dir           = ""
Path_ISD_extr        = JOIN_ARRAY[0] #output_dir+"/ISD_Files/"
Path_wrf_extr        = JOIN_ARRAY[1] #output_dir+"/WRF_Extraction/"
Path_upp_extr        = JOIN_ARRAY[2] #

Path_static_stations = "/shared/stormcenter/AirMG/ISD_Evaluation/Static_Files/WRFv3.8.1_operational/"
##############################################################################################################
def Spatial_Statistics(Joined_Data_Frame,METRIC):
    Vars      = 9
    Stations = np.unique(np.asarray(Joined_Data_Frame['WBAN']))
    NWP_stats = np.zeros((len(Stations),Vars+1))
    for x in range(len(Stations)):
        NWP_stats[x,9] = Stations[x]
        Stat_Data      = Joined_Data_Frame[Joined_Data_Frame['WBAN']==Stations[x]]
        if len(Stat_Data)>0:
            z         = 0
            for y in range(Vars):
                #PREPROCESSING
                X         = Stat_Data[OBS[y]]
                Y         = Stat_Data[NWP[y]]
                var_nan = np.transpose(np.vstack((X,Y)))
                var = var_nan[~np.isnan(var_nan).any(axis=1)]
                X = var[:,0]
                Y = var[:,1] 
                if (len(X)>2) and (len(Y)>2):
                    #STATISTICS
                    BIAS     = np.nanmean(Y-X) #Mean Bias
                    
                    _, _, r_value, _, _ = stats.linregress(X,Y) #Fit/slope intercept
                    RSQUARED = r_value**2
                    RMSE = mean_squared_error(X, Y)**0.5 #RMSE
                    CRMSE = (RMSE**2-BIAS**2)**0.5 #CRMSE
                    if BIAS>RMSE:
                        CRMSE = CRMSE.imag
                    elif BIAS<RMSE:
                        CRMSE = CRMSE.real
                    elif BIAS==RMSE:
                        CRMSE = 0
                    if METRIC=='RSQUARED': NWP_stats[x,y] = RSQUARED
                    if METRIC=='BIAS'    : NWP_stats[x,y] = BIAS
                    if METRIC=='RMSE'    : NWP_stats[x,y] = RMSE
                    if METRIC=='CRMSE'   : NWP_stats[x,y] = CRMSE
                else:
                    NWP_stats[x,y] = np.nan
        else:
            NWP_stats[x,y] = np.nan
    return (NWP_stats)
##############################################################################################################
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
##############################################################################################################
#heat scatter plot
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
    var_nan = np.transpose(np.vstack((X,Y)))
    var = var_nan[~np.isnan(var_nan).any(axis=1)]
    X = var[:,0]
    Y = var[:,1] 
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
        plt.text(0.05, 0.80,alphabet_idx+'\n' "Cor. C. = "+str(round(r_value,2))+'\n' "y = "+str(round(slope,2))+
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
############################################################################################################## 
#filter data    
def remove_nan(Observed,Predictor):
    X = Observed
    Y = Predictor
    var_nan = np.transpose(np.vstack((X,Y)))
    var = var_nan[~np.isnan(var_nan).any(axis=1)]
    return (var)
##############################################################################################################    
# Dumby data for colorbar
x = [i for i in range(20)]
dumby_img = plt.hexbin(x, x,cmap='jet', vmin=1, vmax=c_max,mincnt=1,norm=mpl.colors.LogNorm())
#plt.close(fig=None)
##############################################################################################################        
def make_colorbar(ax, mappable, **kwargs):
    divider = make_axes_locatable(ax)
    orientation = kwargs.pop('orientation', 'vertical')
    if orientation == 'vertical':
        loc = 'right'
    elif orientation == 'horizontal':
        loc = 'bottom'
    cax = divider.append_axes(loc, '5%', pad='3%', axes_class=mpl.pyplot.Axes)
    ax.get_figure().colorbar(mappable, cax=cax, orientation=orientation)
##############################################################################################################
#shapefiles for cartopy
shp_path  = '/shared/stormcenter/AirMG/ISD_Evaluation/Static_Files/Shapefiles_Cartopy'
fname_usa = shp_path+'/United_States_States_5m/cb_2016_us_state_5m.shp'
fname_chm = shp_path+'/Lake_Champlain/c0100e10-b780-4d5f-b0df-43d1a79a49ad2020410-1-seteko.xnxnj.shp'
fname_can = shp_path+'/Canada_trsmd/canada_tr.shp'

def Cartopy_Plot(B_MIN,B_MAX,B_INC,LAT,LON,Statistic,alphabet_idx,Title_Seq,Titles):
    ax.set_extent([West, East, South, North])  
    #shapefile boundaries usa,canada,lake champlain
    ax.add_geometries(Reader(fname_usa).geometries(), ccrs.PlateCarree(), facecolor="none", edgecolor='black', lw=0.7)
    ax.add_geometries(Reader(fname_can).geometries(), ccrs.PlateCarree(), facecolor="none", edgecolor='black', lw=0.7)
    ax.add_geometries(Reader(fname_chm).geometries(), ccrs.PlateCarree(), facecolor="none", edgecolor='black', lw=0.7)
    #colorbar info
    bins = np.arange(B_MIN,B_MAX+B_INC,B_INC)
    cmap = plt.get_cmap('seismic', len(bins)-1)
    c_norm = mpl.colors.BoundaryNorm(bins, ncolors=len(bins)-1)
    #data processing
    X = LON
    Y = LAT
    Z = Statistic
    var_nan = np.transpose(np.vstack((X,Y,Z)))
    var = var_nan[~np.isnan(var_nan).any(axis=1)]
    X = var[:,0]
    Y = var[:,1] 
    Z = var[:,2] 
    cb = plt.scatter(X,Y,c=Z,transform=ccrs.PlateCarree(),cmap=cmap, norm=c_norm,alpha=1,edgecolors='k',
                     linewidths=0.5,zorder=10)
    make_colorbar(ax, cb)
    #graticule
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.75, color='gray', alpha=0.8, linestyle='--')
    gl.ylocator = mticker.FixedLocator([38, 40, 42, 44, 46, 48])
    gl.xlocator = mticker.FixedLocator([-82, -78, -74, -70, -66])
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}
    if Title_Seq ==1:
        fsize = 20
        plt.title(Titles, fontsize = fsize,x=-10) 
################################################################################
###JOINING
################################################################################
wrf_headers           = ['WBAN','YYYY','MM','DD','HH','mn','i','PSFC_pa_f','T2_K_f','Td_K_f','RH_pct_f','SH_kgkg_f',
                         'Tw_C_f','WS_ms_f','AFWA_WG_ms_f','ECMWF_WG_ms_f','WD_deg_f','QPF_mm_f']
wrfout                = pd.read_csv(Path_wrf_extr,names=wrf_headers)
Concat                = np.empty((len(wrfout), 1), dtype=object)
wrfout["Join_Array"]  = ""
for i in range(len(Concat)):
    NWP_dt            = datetime(wrfout["YYYY"][i],wrfout["MM"][i],wrfout["DD"][i],wrfout["HH"][i],wrfout["mn"][i],0)
    NWP_dt            = YYYYMMDDHHMM_string(NWP_dt)
    Concat[i,0]       = str(wrfout['WBAN'][i]) + "_"+NWP_dt
wrfout["Join_Array"]  = pd.DataFrame(Concat)
################################################################################
upp_head              = ['WBAN','WG_f_UPP','MM','DD','YYYY','HH','mn','DHH']
postprd               = pd.read_csv(Path_upp_extr,names=upp_head)
Concat                = np.empty((len(postprd), 1), dtype=object)
postprd["Join_Array"] = ""
for i in range(len(Concat)):
    UPP_Time_Valid          = datetime(postprd["YYYY"][i],postprd["MM"][i],postprd["DD"][i],postprd["HH"][i],postprd["mn"][i],0)+timedelta(hours=int(postprd['DHH'][i]))
    NWP_dt                  = YYYYMMDDHHMM_string(UPP_Time_Valid)
    Concat[i,0]             = str(postprd['WBAN'][i]) + "_"+NWP_dt
postprd["Join_Array"]       = pd.DataFrame(Concat)
################################################################################
ISD_Ob               = pd.read_csv(Path_ISD_extr)
Concat               = np.empty((len(ISD_Ob), 1), dtype=object)
ISD_Ob["Join_Array"] = ""
for i in range(len(Concat)):
    ISD_WBAN_station = str(ISD_Ob['WBAN'][i]) + "_"+str(ISD_Ob['Time_Valid'][i])
    Concat[i,0]      = ISD_WBAN_station.replace(".0","")
ISD_Ob["Join_Array"] = pd.DataFrame(Concat)
################################################################################
################################################################################
Joined_Array = ISD_Ob.merge(wrfout,on='Join_Array').merge(postprd,on='Join_Array')
Joined_Array['T2_C_f'] = Joined_Array['T2_K_f']-273.15
Joined_Array['PSFC_mb_o'] = Joined_Array['PSFC_pa_o']/100
Joined_Array['PSFC_mb_f'] = Joined_Array['PSFC_pa_f']/100
Joined_Array.to_csv("./"+output_dir+"/eval_joined_WRF381_ISD.csv", sep=',',index=False)

################################################################################
###PLOTTING
################################################################################

Titles = [["2m-Temperature [$^\circ$C]","2m-Dew Point Temperature [$^\circ$C]","2m-Wet-bulb Temperature [$^\circ$C]"],
          ["Station Pressure [mb]","2m-Specific Humidity [kg/kg]","2m-Relative Humidity [%]"],
          ["10m-Wind Speed [m/s]","10m-Wind Gust [m/s]","10m-Wind Direction [deg.]"]]

OBS = [["T2_C_o"    ,"Td_C_o"     ,"Tw_C_o"  ],
       ["PSFC_mb_o" ,"SH_kgkg_o"  ,"RH_pct_o"],
       ["WS_ms_o"   ,"WG_ms_o"    ,"WD_deg_o"]]

NWP = [["T2_C_f"    ,"Td_K_f"     ,"Tw_C_f"  ],
       ["PSFC_mb_f" ,"SH_kgkg_f"  ,"RH_pct_f"],
       ["WS_ms_f"   ,"WG_f_UPP","WD_deg_f"]]
Y_Labels  = ["pred.", "pred.", "pred."]
XYMIN     = [[-40,-40,-40],
             [920,  0,  0],
             [  0,  0,  0]]
XYMAX     = [[40  ,    40,40 ],
             [1050,0.0200,100],
             [40  ,    40,360]]
INCR      = [[0.5,0.5   ,0.5],
             [1  ,0.0001,1  ],
             [0.5,0.5   ,10]]
fontszt   = 12
titlesize = 16
fontsz = 16
alphabet_idx = [["A","B","C"],
                ["D","E","F"],
                ["G","H","I"]]
WD_option  = [[0,0,0],
              [0,0,0],
              [0,0,1]] 
xlabel_log = [[0,0,0],
              [0,0,0],
              [1,1,1]]
ylabel_log = [[1,0,0],
              [1,0,0],
              [1,0,0]]
title_log  = [[1,1,1],
              [1,1,1],
              [1,1,1]]
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
        Heat_bin_plots(XYMIN[x][y],XYMAX[x][y],INCR[x][y],Joined_Array[NWP[x][y]],Joined_Array[OBS[x][y]],c_min,c_max,
                      xlabel_log[x][y],ylabel_log[x][y],title_log[x][y],yticks_log[x][y],
                      Titles[x][y],Y_Labels[x],alphabet_idx[x][y],WD_option[x][y])
#plt.suptitle("Heat Variables"'\n'"Observations vs. NWP Model",y=0.99,fontsize=24)
fig.subplots_adjust(bottom=0.1, top=0.97, left=0.05, right=0.95,wspace=0.15, hspace=0.18)
#[left, bottom, width, height],
cb_ax = fig.add_axes([0.05, 0.045,0.9, 0.01])
cbar = fig.colorbar(dumby_img, cax=cb_ax,orientation="horizontal",format='%4.0f')
cbar.set_label('Bin Count',fontsize = 14)
plt.savefig("eval_heatscatter_plots.png")
#plt.close(fig=None)

OBS = ["T2_C_o","Td_C_o","Tw_C_o","PSFC_mb_o","SH_kgkg_o","RH_pct_o","WS_ms_o","WG_ms_o"     ,"WD_deg_o"]
NWP = ["T2_C_f","Td_K_f","Tw_C_f","PSFC_mb_f","SH_kgkg_f","RH_pct_f","WS_ms_f","WG_f_UPP" ,"WD_deg_f"]

Stats_Headers = ["NWP_T2_C","NWP_Td_C","NWP_Tw_C","NWP_SFCP_mb","NWP_SH_kgkg","NWP_RH_pct",
                 "NWP_WS_ms","NWP_WG_ms","NWP_WD_deg","WBAN"]

ISD_stations                 = pd.read_csv(Path_static_stations+'WRF381_Stations.csv')
Joined_Array['WBAN']         = Joined_Array['WBAN_x']
d01_stats                    = Spatial_Statistics(Joined_Array,'BIAS')
d01_stats[d01_stats == 0]    = 'nan'
d01_stats                    = pd.DataFrame(data=d01_stats, columns=Stats_Headers)
d01_stats                    = pd.merge(d01_stats, ISD_stations, left_on='WBAN', right_on='Station')




Titles        = [["2m-Temperature [$^\circ$C]","2m-Dew Point Temperature [$^\circ$C]","2m-Wet-bulb Temperature [$^\circ$C]"],
                 ["Station Pressure [mb]","2m-Specific Humidity [kg/kg]","2m-Relative Humidity [%]"],
                 ["10m-Wind Speed [m/s]","10m-Wind Gust [m/s]","10m-Wind Direction [deg.]"]]
lats_d01       = d01_stats['Lat']
lons_d01       = d01_stats['Lon']
zoom_scale     = 1.5
bbox_d01       = [np.min(lats_d01)-zoom_scale,np.max(lats_d01)+zoom_scale,np.min(lons_d01)-zoom_scale,np.max(lons_d01)+zoom_scale]
South = np.min(lats_d01)-zoom_scale+1
North = np.max(lats_d01)+zoom_scale-1
West  = np.min(lons_d01)-zoom_scale
East  = np.max(lons_d01)+zoom_scale
Stats_Headers = [["NWP_T2_C"   ,"NWP_Td_C"   ,"NWP_Tw_C"         ],
                 ["NWP_SFCP_mb","NWP_SH_kgkg","NWP_RH_pct"       ],
                 ["NWP_WS_ms"  ,"NWP_WG_ms"  ,"NWP_WD_deg","WBAN"]]                 
alphabet_idx = [["A","B","C"],
                ["C","D","F"],
                ["E","F","I"]]
B_MIN = [[-4  ,-4      ,-4  ],
         [-10 ,-0.0008 ,-20 ],
         [-10 ,-10     ,-90]]
B_MAX = [[4   ,4      ,4  ],
         [10  ,0.0008 ,20 ],
         [10  ,10     ,90]]
B_INC = [[0.5 ,0.5      ,0.5 ],
         [1   ,0.00010  ,5   ],
         [1   ,1        ,10  ]]
Plot_Seq    =  [[1,2,3],
                [4,5,6],
                [7,8,9]]
Title_Seq   =  [[1,1,1],
                [1,1,1],
                [1,1,1]]

plt.figure(figsize=(20, 15))
for x in range(3):
    for y in range(3):
        ax = plt.subplot(3,3,Plot_Seq[x][y], projection=ccrs.Mercator()) #changed from mercator -> not usable in hpc
        stat     = d01_stats[Stats_Headers[x][y]]
        Cartopy_Plot(B_MIN[x][y],B_MAX[x][y],B_INC[x][y],lats_d01,lons_d01,stat,
                     alphabet_idx[x][y],Title_Seq[x][y],Titles[x][y])
plt.suptitle("WRFv3.8.1 Modeled Bias [NWP-Obs.] by Variable",y=0.95,fontsize=28)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.savefig("eval_spatialbias_plots.png")
#plt.close(fig=None)

