;************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
;************************************************

begin

NCL_ARRAY=SED_REPLACE
UPP_path     = NCL_ARRAY(0) ;"/shared/airmg/WRF_Modeling/Offshore_Windfarm/simulations/20131214_20131216_Tasnim/postprd/"
domain       = NCL_ARRAY(1) ;"2"
Station_loop = (/"a","b","c","d"/) ;change if >4

;**********************************************************************************************
; Read the file 'Stations.csv' then extract its WBANS and their respective coordinates.
;**********************************************************************************************

nstats   = 3                       ;change if >4
Filename = "./Template/SoundingStations.csv" ;add new row for new station
Stations = asciiread(Filename,-1,"string")   
        WBAN_Strings_Deliminated = str_split_csv(Stations,",",2)
        WBAN_sta = WBAN_Strings_Deliminated(1:nstats,0)
        Ylat_sta = stringtofloat(WBAN_Strings_Deliminated(1:nstats,2))
        Xlon_sta = stringtofloat(WBAN_Strings_Deliminated(1:nstats,3))

print("-----------------------------------EVALUATING STATIONS---------------------------------------------------------------")
print(WBAN_sta)
print("-----------------------------------EVALUATING STATIONS---------------------------------------------------------------")

UPP_listing       = systemfunc("ls "+UPP_path+"/*WRFPRS_d0"+domain+"*")
npts              = dimsizes(UPP_listing)

;**********************************************************************************************
;The first 'i' do loop loads the grib file for all hours for one event into the loop
;**********************************************************************************************

do i=12,npts-1 ;LOOP THROUGH TIME
	print(i)
	;Add the grib file for one hour then extract constant variables for future interpolation-----
	   Grib_File        = addfile(UPP_listing(i)+".grb","r")
	   Time_UPP_MSLP    = Grib_File->PRMSL_GDS3_MSL(:,:)
	   Pressure_surface = Grib_File->lv_ISBL3(:)
	   diso             = dimsizes(Pressure_surface)
	   if (i.eq.12) then
		print("-----------------------------------EVALUATING ISOBARIC SURFACES---------------------------------------------------------------")
		print(Pressure_surface)
		print("-----------------------------------EVALUATING ISOBARIC SURFACES---------------------------------------------------------------")
	   end if
	   Lat          = Grib_File->g3_lat_0(:,:)
	   Lon          = Grib_File->g3_lon_1(:,:)

	;Determine the time of the grib file using wind gust
           TIME_UPP_init_1d      = Time_UPP_MSLP@initial_time
           TIME_UPP_init_2d      = str_split(TIME_UPP_init_1d, " ")
           TIME_UPP_init_date    = TIME_UPP_init_2d(0)
           TIME_UPP_init_date    = str_sub_str(TIME_UPP_init_date,"/",",") ;month,day,year
           TIME_UPP_init_hour_1d = TIME_UPP_init_2d(1)
           TIME_UPP_init_hour_1d = str_sub_str(TIME_UPP_init_hour_1d,"(","")
           TIME_UPP_init_hour_1d = str_sub_str(TIME_UPP_init_hour_1d,")","")
           TIME_UPP_init_hour    = str_sub_str(TIME_UPP_init_hour_1d,":",",") ;hour,min
           TIME_UPP_fcts         = Time_UPP_MSLP@forecast_time ;+mins since init
           Time_UPP_str          = TIME_UPP_init_date+","+TIME_UPP_init_hour+","+TIME_UPP_fcts
           Time_UPP_str_Print    = rm_single_dims(onedtond(Time_UPP_str,(/diso,1/)))

	;Preallocate 'Interpolated' fields for extraction.--------------------------------------------
	   INT_Tiso      = new((/diso,1/),float)
	   ;INT_Tdiso     = new((/diso,1/),float)
	   INT_SHiso     = new((/diso,1/),float)
	   ;INT_RHiso     = new((/diso,1/),float)
	   ;Tw            = new((/diso,1/),float)
	   INT_Uiso      = new((/diso,1/),float)
	   INT_Viso      = new((/diso,1/),float)
	   INT_WindSpeed = new((/diso,1/),float)
	   INT_WindDir   = new((/diso,1/),float)
	   ;INT_PotT      = new((/diso,1/),float)

;**********************************************************************************************
;The first 'k' loop extracts the 'kth' isobaric level for one hour for the grib file above, bilinearly interpolates the gridded data to the three sounding locations, then places its contents in the 'kth' column of the prealocated variables mentioned above. Certain values have been calculated. 
;**********************************************************************************************

	do z = 0,nstats-1
        	Ylat = Ylat_sta(z)
        	Xlon = Xlon_sta(z)
        	WBAN = rm_single_dims(onedtond(WBAN_sta(z),(/diso,1/)))
		do k=0,diso-1
		print(k)
        		TEMP_Tiso  = Grib_File->TMP_GDS3_ISBL(k,:,:)
			;TEMP_Tdiso = Grib_File->DPT_GDS3_ISBL(k,:,:)
        		TEMP_SHiso = Grib_File->SPF_H_GDS3_ISBL(k,:,:)
			;TEMP_RHiso = Grib_File->R_H_GDS3_ISBL(k,:,:)
			TEMP_Uiso  = Grib_File->U_GRD_GDS3_ISBL(k,:,:)
			TEMP_Viso  = Grib_File->V_GRD_GDS3_ISBL(k,:,:)
			;TEMP_PotT  = Grib_File->POT_GDS3_ISBL(k,:,:)		

			;Preallocate 'Interpolated' fields for extraction.--------------------------------------------
			INT_Tiso(k,0)      = rcm2points(Lat,Lon,TEMP_Tiso,Ylat,Xlon,2)
   			;INT_Tdiso(k,0)     = rcm2points(Lat,Lon,TEMP_Tdiso,Ylat,Xlon,2)
        		INT_SHiso(k,0)     = rcm2points(Lat,Lon,TEMP_SHiso,Ylat,Xlon,2)
   			;INT_RHiso(k,0)     = rcm2points(Lat,Lon,TEMP_RHiso,Ylat,Xlon,2)
			;INT_PotT(k,0)      = rcm2points(Lat,Lon,TEMP_PotT,Ylat,Xlon,2)
  			;Tw(k,0)            = (INT_Tiso(k,0)-273.15)*atan(0.151977*(INT_RHiso(k,0)+8.313659)^0.5)+atan((INT_Tiso(k,0)-273.15)+INT_RHiso(k,0))-atan(INT_RHiso(k,0)-1.676331)+0.00391838*((INT_RHiso(k,0))^1.5)*atan(0.023101*INT_RHiso(k,0))-4.686035
   			INT_WindSpeed(k,0) = rcm2points(Lat,Lon,sqrt(TEMP_Uiso^2+TEMP_Viso^2),Ylat,Xlon,2)
   			INT_WindDir(k,0)   = rcm2points(Lat,Lon,(wind_direction(TEMP_Uiso,TEMP_Viso,1)),Ylat,Xlon,2)
		end do ;k
	Line = (/WBAN+","+Time_UPP_str_Print+","+Pressure_surface+","+INT_Tiso(:,0)+","+INT_SHiso(:,0)+","+INT_WindSpeed(:,0)+","+INT_WindDir(:,0)/)
	asciiwrite(""+"VP_"+tostring(i)+"_"+Station_loop(z)+".csv",Line)
	print(Line)
	end do ;z
end do ;i

ConcentrateFiles = systemfunc("cat VP_*> WRF_vertical_profile.csv")
MoveAllCompFiles = systemfunc("rm *VP_*.csv")

end

