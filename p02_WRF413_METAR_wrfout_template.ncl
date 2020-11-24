;************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
;load "./WRFUserARW.ncl"

;*************************************************************************************
; Direct Output from WRF First *******************************************************
;*************************************************************************************

begin

;**********************************************************************************************
; Read the file 'Stations.csv' then extract its WBANS and their respective coordinates.
;**********************************************************************************************
NCL_WRFOUT_ARRAY=SED_WRF_DETAILS
WRF381_directory   = NCL_WRFOUT_ARRAY(0)   ;"/shared/airmg/WRF_Modeling/Offshore_Windfarm/simulations/HRRR_UTAH_2017102900/wrfout_files"
spinup_time        = 12*6 ;hours
Station_Path       = NCL_WRFOUT_ARRAY(1)   ;"/shared/stormcenter/AirMG/ISD_Evaluation/Static_Files/WRFv3.8.1_operational/WRF381_Stations.csv"
domain_list        = NCL_WRFOUT_ARRAY(2:)

do k=0,dimsizes(domain_list)-1
	print("Extracting wrfout*.nc for d0"+tostring(domain_list(k)))
        print("Extracting stations from"+Station_Path+"WRF381_Stations_d0"+tostring(domain_list(k))+".csv")
	Stations                 = asciiread(Station_Path,-1,"string")
	WBAN_Strings_Deliminated = str_split_csv(Stations,",",2)
	WBAN                     = WBAN_Strings_Deliminated(1:,0)
	Ylat                     = stringtofloat(WBAN_Strings_Deliminated(1:,1))
	Xlon                     = stringtofloat(WBAN_Strings_Deliminated(1:,2))
	d                        = dimsizes(WBAN)
	Listing                  = systemfunc("ls "+WRF381_directory+"/wrfprd/*wrfout_d0"+tostring(domain_list(k))+"*")
	npts                     = dimsizes(Listing)
	;Stations,WBAN_Strings_Deliminated,WBAN,Ylat,Xlon,Listing
	do i=spinup_time,npts-1
		print(i)
		forLoop    = tostring(i)
		;Add files for both wrf output and post-processed output.--------------------------------------
   		NCDF_File  = addfile(""+Listing(i)+".nc","r")
   		nNC_Files  = dimsizes(NCDF_File)
		;print(NCDF_File)
		;Get times.------------------------------------------------------------------------------------
   		Times = NCDF_File->Times
   		Times_Formated = tostring(wrf_times_c(Times,2))
		print(Times_Formated)
   		YYYY = rm_single_dims(onedtond(stringtoint(str_get_cols(Times_Formated,0,3)),(/d,1/)))
   		MM = rm_single_dims(onedtond(stringtoint(str_get_cols(Times_Formated,4,5)),(/d,1/)))
   		DD = rm_single_dims(onedtond(stringtoint(str_get_cols(Times_Formated,6,7)),(/d,1/)))
   		HH = rm_single_dims(onedtond(stringtoint(str_get_cols(Times_Formated,8,9)),(/d,1/)))
                mn = rm_single_dims(onedtond(stringtoint(str_get_cols(Times_Formated,10,11)),(/d,1/)))
		;YYYY,MM,DD,HH,
		;Repeat the same process from above for the wrf output files.----------------------------------
   		RAINC    = NCDF_File->RAINC(0,:,:)
   		RAINNC   = NCDF_File->RAINNC(0,:,:) 
   		V10      = NCDF_File->V10(0,:,:)
   		U10      = NCDF_File->U10(0,:,:)
   		cosalpha = NCDF_File->COSALPHA(0,:,:)
   		sinalpha = NCDF_File->SINALPHA(0,:,:)
   		U_star   = NCDF_File->UST(0,:,:)
   		WG_AFWA  = NCDF_File->WSPD10MAX(0,:,:)
           	uc     = U10*cosalpha - V10*sinalpha
	  	vc     = V10*cosalpha + U10*sinalpha
           	WS10   = sqrt(U10^2+V10^2)
           	WindDc = wind_direction(uc,vc,1)
   		SFCPress = NCDF_File->PSFC(0,:,:)
   		Temp2m = NCDF_File->T2(0,:,:)
   		Q2Qv_SH = NCDF_File->Q2(0,:,:)
   		BaseStatPress = NCDF_File->PB(0,0,:,:)
   		PertPress = NCDF_File->P(0,0,:,:)
   		Td = wrf_td((BaseStatPress+PertPress),Q2Qv_SH)
   		RH = wrf_rh(Q2Qv_SH,SFCPress,Temp2m)
  		Tw = (Temp2m-273.15)*atan(0.151977*(RH+8.313659)^0.5)+atan((Temp2m-273.15)+RH)-atan(RH-1.676331)+0.00391838*((RH)^1.5)*atan(0.023101*RH)-4.686035
   		Lon = NCDF_File->XLONG(0,:,:)
   		Lat = NCDF_File->XLAT(0,:,:)
		;NCDF_File,RAINC,RAINNC,V10,U10,cosalpha,sinalpha,U_star,WG_AFWA,vc,uc,WS10,WindDc,SFCPress,Temp2m,Q2Qv_SH,BaseStatPress,PertPress,Td,RH,Tw,Lon,Lat,
           	INT_U_star = rcm2points(Lat,Lon,U_star,Ylat,Xlon,2)
           	INT_SFCP   = rcm2points(Lat,Lon,SFCPress,Ylat,Xlon,2)
	   	INT_T2sfc  = rcm2points(Lat,Lon,Temp2m,Ylat,Xlon,2)
           	INT_Tdsfc  = rcm2points(Lat,Lon,Td,Ylat,Xlon,2) 
           	INT_RH     = rcm2points(Lat,Lon,RH,Ylat,Xlon,2)
	   	INT_SH     = rcm2points(Lat,Lon,Q2Qv_SH,Ylat,Xlon,2)	
           	INT_Twsfc  = rcm2points(Lat,Lon,Tw,Ylat,Xlon,2)
	   	INT_WS10   = rcm2points(Lat,Lon,WS10,Ylat,Xlon,2)
           	INT_WG_AFWA  = rcm2points(Lat,Lon,WG_AFWA,Ylat,Xlon,2)
           	INT_WG_ECMWF = INT_WS10+7.71*INT_U_star ;non-convective half	   
           	INT_WDc    = rcm2points(Lat,Lon,WindDc,Ylat,Xlon,2)
           	INT_QPF    = rcm2points(Lat,Lon,RAINNC,Ylat,Xlon,2)+rcm2points(Lat,Lon,RAINC,Ylat,Xlon,2)
		;YYYY,MM,DD,HH,Listing,Listing,Time_UPP_str_Print,INT_SFCP,INT_T2sfc,INT_Tdsfc,INT_RH,INT_SH,INT_Twsfc,INT_WS10,INT_WS10c,INT_WD,INT_WDc,INT_QPF
		lines =  WBAN+","+sprintf("%4.0f",YYYY)+","+sprintf("%2.0f",MM)+","+sprintf("%2.0f",DD)+","+sprintf("%2.0f",HH)+","+sprintf("%2.0f",mn)+","+sprintf("%2.0f",i-12)+","+sprintf("%14.0f",INT_SFCP)+","+sprintf("%5.2f",INT_T2sfc)+","+sprintf("%5.2f",INT_Tdsfc)+","+sprintf("%3.1f",INT_RH)+","+sprintf("%8.8f",INT_SH)+","+sprintf("%5.2f",INT_Twsfc)+","+sprintf("%3.1f",INT_WS10)+","+sprintf("%3.1f",INT_WG_AFWA)+","+sprintf("%3.1f",INT_WG_ECMWF)+","+sprintf("%3.1f",INT_WDc)+","+sprintf("%5.2f",INT_QPF)
		print(lines)
		asciiwrite(""+"comp"+forLoop+".csv",lines)
		;INT_U_star,INT_SFCP,INT_T2sfc,INT_Tdsfc,INT_RH,INT_SH,INT_Twsfc,INT_WS10,INT_WG_AFWA,INT_WG_ECMWF,INT_WDc,INT_QPF,lines
		;WBAN,YYYY,MM,DD,HH,i,PSFC_pa_f,T2_K_f,Td_K_f,RH_pct_f,SH_kgkg_f,Tw_C_f,WS_ms_f,AFWA_WG_ms_f,ECMWF_WG_ms_f,WD_deg_f,QPF_mm_f
	end do
	ConcentrateFiles     = systemfunc("cat comp*.csv> "+"./wrfout_extraction_d0"+domain_list(k)+".csv")
	MoveAllCompFiles     = systemfunc("rm comp*.csv")
	;wrfoutHeader         = systemfunc("{ echo "WBAN,YYYY,MM,DD,HH,mn,i,SFCP_f,T2_K_f,Td_K_f,RH_pct_f,SH_kgkg_f,Tw_C_f,WS_ms_f,AFWA_WG_ms_f,ECMWF_WG_ms_f,WD_deg_f,QPF_mm_f"; cat ./WRF_Output/wrfout_extraction.csv; } > wrfout_extraction.csv")
end do
end

