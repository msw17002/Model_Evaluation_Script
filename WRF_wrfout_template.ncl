;************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
;************************************************


begin

;replaced by bash script
NCL_ARRAY=SED_REPLACE
;$WRF_Directory","$WRF_Stations","$Domains
;wrf path and analysis domain
station_path = NCL_ARRAY(1)
wrf_path     = NCL_ARRAY(0)
domain       = NCL_ARRAY(2)

;buoy stations for interpolation
Stations = asciiread(station_path,-1,"string")
;format
;STATION_ID,Elev,Temp_e,Anem_h,Barom_h,Lat,Lon
;44007,0,4.5,5,0.5,43.525,-70.141
WBAN_Strings_Deliminated = str_split_csv(Stations,",",2)
WBAN                     = WBAN_Strings_Deliminated(1:,0)
Ylat                     = stringtofloat(WBAN_Strings_Deliminated(1:,5))
Xlon                     = stringtofloat(WBAN_Strings_Deliminated(1:,6))
nstats                   = dimsizes(WBAN)

;get listing of all upp/wrf files
WRF_Listing         = systemfunc("ls "+wrf_path+"/wrfprd/*wrfout_d0"+domain+"*")
npts                = dimsizes(WRF_Listing)

do i=12,npts-1
print(i)
forLoop   = tostring(i)
NCDF_File = addfile(WRF_Listing(i)+".nc","r")
;Get times
	Times          = NCDF_File->Times
        Times_Formated = tostring(wrf_times_c(Times,2))
        YYYY           = rm_single_dims(onedtond(stringtoint(str_get_cols(Times_Formated,0,3)),(/nstats,1/)))
        MM             = rm_single_dims(onedtond(stringtoint(str_get_cols(Times_Formated,4,5)),(/nstats,1/)))
        DD             = rm_single_dims(onedtond(stringtoint(str_get_cols(Times_Formated,6,7)),(/nstats,1/)))
        HH             = rm_single_dims(onedtond(stringtoint(str_get_cols(Times_Formated,8,9)),(/nstats,1/)))
        mn             = rm_single_dims(onedtond(stringtoint(str_get_cols(Times_Formated,10,11)),(/nstats,1/)))
;surface temperature at t2
	T2          = NCDF_File->T2(0,:,:)
;water surface temperature 
	SST	    = NCDF_File->SST(0,:,:)
	SSTSK	    = NCDF_File->SSTSK(0,:,:)
;wind at 10m
	U10         = NCDF_File->U10(0,:,:)
	V10         = NCDF_File->V10(0,:,:)
	WS10        = (U10^2+V10^2)^0.5
	WD10        = wind_direction(U10,V10,1)
;rh at 2m
	RH2         = wrf_user_getvar(NCDF_File,"rh2",0)
;td at 2m
	TD2         = wrf_user_getvar(NCDF_File,"td2",0)
;determine temperature
	TK   = wrf_user_getvar(NCDF_File,"tc",0)
;determine wind speed
	uvm  = wrf_user_getvar(NCDF_File,"uvmet",0) 
	umet = uvm(0,:,:,:) ;is umet, and
        vmet = uvm(1,:,:,:) ;is vmet, and 
	WS   = (umet^2+vmet^2)^0.5
;determine wind direction
	WD   = wind_direction(umet,vmet,1)	
;determine dew point temperature
	TD = wrf_user_getvar(NCDF_File,"td",0)  ; calculate td
;determine RH
	RH  = wrf_user_getvar(NCDF_File,"rh",0)
	RH2 = wrf_user_getvar(NCDF_File,"rh2",0)
;determine Tw
	TW = wrf_user_getvar(NCDF_File,"twb",0)
	TW2= wetbulb_stull((T2-273.15),RH2,(/0,0/), False) ;Celsius
;determine height
	HGT = wrf_user_getvar(NCDF_File,"z",0)
;determine MSLP
	MSLP = wrf_user_getvar(NCDF_File,"slp",0)
;get coords
	Latnc = NCDF_File->XLAT(0,:,:)
	Lonnc = NCDF_File->XLONG(0,:,:)	
;bilinear interpolation of surface variabels
       	INT_MSLP  = rcm2points(Latnc,Lonnc,MSLP,Ylat,Xlon,2)
	INT_T2    = rcm2points(Latnc,Lonnc,T2,Ylat,Xlon,2)-273.15
	INT_SST   = rcm2points(Latnc,Lonnc,SST,Ylat,Xlon,2)-273.15
	INT_SSTSK = rcm2points(Latnc,Lonnc,SSTSK,Ylat,Xlon,2)
       	INT_WS10  = rcm2points(Latnc,Lonnc,WS10,Ylat,Xlon,2)
       	INT_WD10  = rcm2points(Latnc,Lonnc,WD10,Ylat,Xlon,2)
	INT_TD2   = rcm2points(Latnc,Lonnc,TD2,Ylat,Xlon,2)
        INT_RH2  = rcm2points(Latnc,Lonnc,RH2,Ylat,Xlon,2)
        INT_TW2   = rcm2points(Latnc,Lonnc,TW2,Ylat,Xlon,2)
;Preallocate 'Interpolated' fields for extraction.--------------------------------------------
  	INT_TK  = new((/nstats,4/),float)
   	INT_RH  = new((/nstats,4/),float)
	INT_HGT = new((/nstats,4/),float)
	INT_TD  = new((/nstats,4/),float)
        INT_TW  = new((/nstats,4/),float)
        INT_WS  = new((/nstats,4/),float)
	INT_WD  = new((/nstats,4/),float)
;**********************************************************************************************
;The first 'k' loop extracts the 'kth' isobaric level for one hour for the grib file above, bilinearly interpolates the gridded data to the three sounding locations, then places its contents in the 'kth' column of the prealocated variables mentioned above. Certain values have been calculated. 
;**********************************************************************************************
	do k=0,4-1
	print(k)
	;Extract the 'kth' isobaric level-variable----------------------------------------------------
		TEMP_TK     = TK(k,:,:)
	   	TEMP_RH     = RH(k,:,:)
	   	TEMP_HGT    = HGT(k,:,:)
           	TEMP_TD     = TD(k,:,:)
	   	TEMP_TW     = TW(k,:,:)
          	TEMP_WS     = WS(k,:,:)
	 	TEMP_WD     = WD(k,:,:)
	;Bilinearly interpolate the variable at the 3 sounding locations for the 'kth' column---------
	   	INT_TK(:,k)  = rcm2points(Latnc,Lonnc,TEMP_TK,Ylat,Xlon,2)
   		INT_RH(:,k)  = rcm2points(Latnc,Lonnc,TEMP_RH,Ylat,Xlon,2)
		INT_HGT(:,k) = rcm2points(Latnc,Lonnc,TEMP_HGT,Ylat,Xlon,2)
                INT_TD(:,k)  = rcm2points(Latnc,Lonnc,TEMP_TD,Ylat,Xlon,2)
		INT_TW(:,k)  = rcm2points(Latnc,Lonnc,TEMP_TW,Ylat,Xlon,2)
		INT_WS(:,k)  = rcm2points(Latnc,Lonnc,TEMP_WS,Ylat,Xlon,2)
		INT_WD(:,k)  = rcm2points(Latnc,Lonnc,TEMP_WD,Ylat,Xlon,2)
	end do
INT_TW = INT_TW-273.15
;station,height
;**********************************************************************************************
;The second 'k' loop is a preprocessed step. Preallocate new arrays the place it in an array so that the variable becomes vertically concentrated... This part isn't necassary.
;**********************************************************************************************
	;Measurements for the dimensions of the preallocated variables then the variables themselves--
	dimx  = dimsizes(INT_WS)
	nrows = dimx(0)
	Temperature      = new(nrows,string)
	RelativeHumdity  = new(nrows,string)
        Height           = new(nrows,string)
	DewpointTemp     = new(nrows,string)
	WetbulbTemp      = new(nrows,string)
	WindSpeed        = new(nrows,string)
	WindDirection    = new(nrows,string)
;Convert the horizontal orieantated array to a vertically oriantated array for each hours-----
     	do k = 0, nrows-1
		print(k)
        	Temperature(k)     = str_concat(sprintf("%3.2f,",INT_TK(k,:)))
        	RelativeHumdity(k) = str_concat(sprintf("%3.2f,",INT_RH(k,:)))
     		Height(k)          = str_concat(sprintf("%8.2f,",INT_HGT(k,:)))
     		DewpointTemp(k)    = str_concat(sprintf("%3.2f,",INT_TD(k,:)))
		WetbulbTemp(k)     = str_concat(sprintf("%3.2f,",INT_TW(k,:)))
		WindSpeed(k)       = str_concat(sprintf("%3.2f,",INT_WS(k,:)))
		WindDirection(k)   = str_concat(sprintf("%3.1f,",INT_WD(k,:)))
     	end do
;**********************************************************************************************
;Concentrate the interpolated values then create a concentrated .csv file... One file represents one hour for all stations for one event.
;**********************************************************************************************
	Line = (/WBAN+","+i+","+YYYY+","+MM+","+DD+","+HH+","+mn+","+i+","+2+","+10+","+INT_MSLP+","+i+","+INT_SSTSK+","+INT_SST+","+INT_T2+","+Temperature+INT_RH2+","+RelativeHumdity+Height+INT_TD2+","+DewpointTemp+INT_TW2+","+WetbulbTemp+INT_WS10+","+WindSpeed+INT_WD10+","+WindDirection/)
	asciiwrite(""+"VP_"+forLoop+".csv",Line)
	print(Line)
	end do
;**********************************************************************************************
;Move all files to the specific event's directory that was previously created. Concentrate the files then move it to the event's folder as well.
;**********************************************************************************************

ConcentrateFiles = systemfunc("cat VP_*> WRF_NDBC.csv")
MoveAllCompFiles = systemfunc("rm VP_*.csv")

end

