;************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
;load "./WRFUserARW.ncl"

;************************************************
;/shared/stormcenter/marika/WRF_Sim/Runs/WRFv3.8.1/running/mike_2
;*************************************************************************************
; Direct Output from WRF First *******************************************************
;*************************************************************************************

begin

;**********************************************************************************************
; Read the file 'Stations.csv' then extract its WBANS and their respective coordinates.
;**********************************************************************************************
NCL_ARRAY=SED_REPLACE
WRF381_directory   = NCL_ARRAY(0)   ;"/shared/airmg/WRF_Modeling/Offshore_Windfarm/simulations/HRRR_UTAH_2017102900/wrfout_files"
spinup_time        = 12 ;hours
Station_Path       = NCL_ARRAY(1)   ;"/shared/stormcenter/AirMG/ISD_Evaluation/Static_Files/WRFv3.8.1_operational/WRF381_Stations.csv"
domain_list        = NCL_ARRAY(2)

do k=0,dimsizes(domain_list)-1
        print("Extracting postprd*.grib for d0"+tostring(domain_list(k)))
        print("Extracting stations from"+Station_Path+"WRF381_Stations_d0"+tostring(domain_list(k))+".csv")
        Stations                 = asciiread(Station_Path,-1,"string")
	;STATION_ID,Elev,Temp_e,Anem_h,Barom_h,Lat,Lon
	;44007,0,4.5,5,0.5,43.525,-70.141

        WBAN_Strings_Deliminated = str_split_csv(Stations,",",2)
        WBAN                     = WBAN_Strings_Deliminated(1:,0)
        Ylat                     = stringtofloat(WBAN_Strings_Deliminated(1:,5))
        Xlon                     = stringtofloat(WBAN_Strings_Deliminated(1:,6))
        d                        = dimsizes(WBAN)
        Listing                  = systemfunc("ls "+WRF381_directory+"/postprd/*WRFPRS_d0"+tostring(domain_list(k))+"*")
        npts                     = dimsizes(Listing)
        ;Stations,WBAN_Strings_Deliminated,WBAN,Ylat,Xlon,Listing
	do i=0,npts-1
		print(i)
                if (i.lt.10) then
                        forLoop="0"+tostring(i)
                else
                        forLoop=tostring(i)
                end if
		;Add files for both wrf output and post-processed output.--------------------------------------
		Grib_File  = addfile(""+Listing(i)+".grib","r")   
   		nGRB_Files = dimsizes(Grib_File)	
		;Extract variables from the grib files (post-processed data).----------------------------------
   		PLon    = Grib_File->g3_lon_1(:,:)
   		PLat    = Grib_File->g3_lat_0(:,:)
   		WG_UPP  = Grib_File->GUST_GDS3_SFC(:,:)
		;Interpolate the results above at the METAR stations using bilinear interpolation.-------------
        	INT_WG_UPP  = rcm2points(PLat,PLon,WG_UPP,Ylat,Xlon,2)
      		TIME_UPP_init_1d      = WG_UPP@initial_time
		TIME_UPP_init_2d      = str_split(TIME_UPP_init_1d, " ")
		TIME_UPP_init_date    = TIME_UPP_init_2d(0)
                TIME_UPP_init_date    = str_sub_str(TIME_UPP_init_date,"/",",") ;month,day,year
		TIME_UPP_init_hour_1d = TIME_UPP_init_2d(1)
		TIME_UPP_init_hour_1d = str_sub_str(TIME_UPP_init_hour_1d,"(","")
                TIME_UPP_init_hour_1d = str_sub_str(TIME_UPP_init_hour_1d,")","")
		TIME_UPP_init_hour    = str_sub_str(TIME_UPP_init_hour_1d,":",",") ;hour,min
        	TIME_UPP_fcts         = WG_UPP@forecast_time ;+mins since init
        	Time_UPP_str          = TIME_UPP_init_date+","+TIME_UPP_init_hour+","+TIME_UPP_fcts
        	Time_UPP_str_Print    = rm_single_dims(onedtond(Time_UPP_str,(/d,1/)))
		lines                 =  WBAN+","+sprintf("%3.1f",INT_WG_UPP)+","+Time_UPP_str_Print
		print(lines)
		asciiwrite(""+"comp"+forLoop+".csv",lines)
		;WBAN,WG,MM,DD,YYYY,HH,Mn,D_HH
	end do
	ConcentrateFiles     = systemfunc("cat comp*.csv> UPP_NDBC.csv")
	MoveAllCompFiles     = systemfunc("rm comp*.csv")
	delete([/Stations,WBAN_Strings_Deliminated,WBAN,Ylat,Xlon,Listing,Grib_File,PLon,PLat,WG_UPP,INT_WG_UPP,TIME_UPP_fcts,Time_UPP_str,Time_UPP_str_Print,lines/])
	;postprdHeader         = systemfunc("{ echo "WBAN,WG_UPP_ms_f,MM,DD,YYYY,HH,MN,delta_HH"; cat ./WRF_Output/postprd_extraction.csv; } > postprd_extraction.csv")
end do
end

