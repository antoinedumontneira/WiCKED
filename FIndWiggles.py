### IMPORTS #####
import numpy as np
import sys
from threadpoolctl import threadpool_limits
import multiprocessing
from FitWiggles import power_law_model_fit,power_law_stellar_fit,get_masked_regions,costume_sigmaclip
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq # Fourier Transform to fin spaxels affected by wiggles
from scipy.signal import savgol_filter

####################
### FUNCTIONS TO FIND PIXELS AFFECTED BY WIGGLES
##################

def get_FFT(wave,data):
    """ Simple function to obtain the Fast Fourier Transform of the given part of the spectrum.

    Args:
        wave (array): Wavelenght
        data (array): Spectrum

    Returns:
       array : FFT model
    """
    # Number of sample points
    N = len(wave)
    # sample spacing
    T = wave[1] - wave[0]
    xf = fftfreq(N, T)[:N//2]
    yf = fft(data)
    return xf,yf

def fit_wiggly_spaxels(freq_range,wave,data,masked_lines,con_windows,model_spaxel,detector, smooth_spectrum,do_plots=False):
    
    """ This function takes the Fourier trasnform of a the wiggle residual spectrum of spaxel in a
    JWSST NIRSpec data cube and compares it to a the fourier transofrm of a reference spectrum.
    The output is a comparison of the median amplitud of the spectrum in the frequencies belonging to wiggles xf < 40
    and returns the ratio in units of standard deviaton. 
    """
    ##
    if detector == 2:
        wiggle_window_nrs1 = np.zeros(wave.shape,bool)
        wiggle_window_nrs2 = np.zeros(wave.shape,bool)
        ### define the wavelenght regions to search for wiggles
        wiggle_window_nrs1[150:1700] = True 
        wiggle_window_nrs2[-1400:-50] = True 
        wiggle_window_nrs1[~con_windows] = False ### EXCLUDE REGIONS OF CONTINUUM WITH STRONG EMISSION LINES IN CON_WINDOW
        wiggle_window_nrs2[~con_windows] = False ### EXCLUDE REGIONS OF CONTINUUM WITH STRONG EMISSION LINES IN CON_WINDOW
        masked_lines_for_model_nrs1 =  masked_lines[wiggle_window_nrs1]
        masked_lines_for_model_nrs2 =  masked_lines[wiggle_window_nrs2]
        ################### DEFINE SPECTRUMS IN THE WAVELENGHT RANGE TO BE MODELED ################
        wave_for_model_nrs1 =  wave[wiggle_window_nrs1]
        wave_for_model_nrs2 =  wave[wiggle_window_nrs2]
        spec_for_model_nrs1 =  data[wiggle_window_nrs1] 
        spec_for_model_nrs2 =  data[wiggle_window_nrs2]  
        model_spaxel_nrs1 = model_spaxel[wiggle_window_nrs1]
        model_spaxel_nrs2 = model_spaxel[wiggle_window_nrs2]
        wiggles_spec_nrs1 = spec_for_model_nrs1  - model_spaxel_nrs1
        wiggles_spec_nrs2 = spec_for_model_nrs2  - model_spaxel_nrs2
    else:
        masked_lines_for_model =  masked_lines
        ################### DEFINE SPECTRUMS IN THE WAVELENGHT RANGE TO BE MODELED ################
        wave_for_model =  wave
        spec_for_model =  data
        model_spaxel = model_spaxel
        wiggles_spec = spec_for_model  - model_spaxel
    ######### LEFT THESE PLOTS HERE FOR DEBUGGING 

    if smooth_spectrum == "yes":
        if detector == 2:
            wiggles_spec_nrs1 = savgol_filter(wiggles_spec_nrs1,10,3)   ### APPLY A ~0.018 microns SMOOTHING TO MAKE WIGGLES MORE VISIBLE. ONLY RECOMMENDED FOR LOW-S/N. 
            wiggles_spec_nrs2 = savgol_filter(wiggles_spec_nrs2,10,3)   ### APPLY A ~0.018 microns SMOOTHING TO MAKE WIGGLES MORE VISIBLE. ONLY RECOMMENDED FOR LOW-S/N. 
        else:
            wiggles_spec = savgol_filter(wiggles_spec,10,3)   ### APPLY A ~0.018 microns SMOOTHING TO MAKE WIGGLES MORE VISIBLE. ONLY RECOMMENDED FOR LOW-S/N. 

    ############# FOURIER TRASNFORM OF SPAXEL AND REFERENCE SPECTRUM
    if detector == 2:
        xf_nrs1,yf_spaxel_nrs1 = get_FFT(wave_for_model_nrs1[masked_lines_for_model_nrs1] ,wiggles_spec_nrs1[masked_lines_for_model_nrs1] )
        xf_nrs2,yf_spaxel_nrs2 = get_FFT(wave_for_model_nrs2[masked_lines_for_model_nrs2] ,wiggles_spec_nrs2[masked_lines_for_model_nrs2] )
        N_nrs1 = len(wave_for_model_nrs1[masked_lines_for_model_nrs1]) 
        N_nrs2 =  len(wave_for_model_nrs2[masked_lines_for_model_nrs2])
        ### GET FREQUENCY RANGE OF WIGGLE.  ### I  DIVIDE THE RANGE OF FREQ INTO 2, CAUSE SOMETIMES THERE IS ONLY ONE TYPE OF WIGGLES. SO USING THE WHOLE RANGE IS LESS EFFECTIVE
        min_freq, max_freq = np.max([freq_range[0],5]), np.min([freq_range[1],50])  ### USUALLY FREQ < 5 is just noise
        short_freq_mask_nrs1_1 = ( min_freq <= xf_nrs1) & ( xf_nrs1 <= max_freq/2)
        short_freq_mask_nrs1_2 = ( max_freq/2 < xf_nrs1) & ( xf_nrs1 <= max_freq)
        short_freq_mask_nrs2_1 = ( min_freq <= xf_nrs2) & ( xf_nrs2 <= max_freq/2)
        short_freq_mask_nrs2_2 = ( max_freq/2 < xf_nrs2) & ( xf_nrs2 <= max_freq)
        large_freq_mask_nrs1 = ( (max_freq + 15) < xf_nrs1) & ( xf_nrs1 <= (max_freq + 15) + 100)
        large_freq_mask_nrs2 = ( (max_freq + 15) < xf_nrs2) & ( xf_nrs2 <= (max_freq + 15) + 100)
        
        # HERE I DECIDE IN WHICH NRS DETECTOR WIGGLES ARE MORE PROMINENT  
        mean_ampl_pixel_nrs1_1  =  np.nanmean( (2.0/N_nrs1 * np.abs(yf_spaxel_nrs1[0:N_nrs1//2]) )[short_freq_mask_nrs1_1] )
        mean_ampl_pixel_nrs1_2  = np.nanmean( (2.0/N_nrs1 * np.abs(yf_spaxel_nrs1[0:N_nrs1//2]) )[short_freq_mask_nrs1_2] )  
        mean_ampl_pixel_nrs2_1  =  np.nanmean( (2.0/N_nrs2 * np.abs(yf_spaxel_nrs2[0:N_nrs2//2]) )[short_freq_mask_nrs2_1] )
        mean_ampl_pixel_nrs2_2  = np.nanmean( (2.0/N_nrs2 * np.abs(yf_spaxel_nrs2[0:N_nrs2//2]) )[short_freq_mask_nrs2_2] )
        spaxel_level_nrs1 = np.nanmax([mean_ampl_pixel_nrs1_1,mean_ampl_pixel_nrs1_2])
        spaxel_level_nrs2 = np.nanmax([mean_ampl_pixel_nrs2_1,mean_ampl_pixel_nrs2_2])
        if spaxel_level_nrs1 > spaxel_level_nrs2 : 
            spaxel_level = spaxel_level_nrs1
            mean_ampl_large_freq  = np.mean( 2.0/N_nrs1 * np.abs(yf_spaxel_nrs1[0:N_nrs1//2])[large_freq_mask_nrs1] )
            mean_ampl_large_freq_std = np.std( 2.0/N_nrs1 * np.abs(yf_spaxel_nrs1[0:N_nrs1//2])[large_freq_mask_nrs1] )
            #### DEFINE THRESHOLD TO DECIDE IF WIGGLES ARE PRESENT OR NOT
            one_sigma_level =  mean_ampl_large_freq + mean_ampl_large_freq_std
            color_mean = "r" 
        else:
            spaxel_level = spaxel_level_nrs2
            mean_ampl_large_freq  = np.mean( 2.0/N_nrs2 * np.abs(yf_spaxel_nrs2[0:N_nrs2//2])[large_freq_mask_nrs2] )
            mean_ampl_large_freq_std = np.std( 2.0/N_nrs2 * np.abs(yf_spaxel_nrs2[0:N_nrs2//2])[large_freq_mask_nrs2] )
            #### DEFINE THRESHOLD TO DECIDE IF WIGGLES ARE PRESENT OR NOT
            one_sigma_level =  mean_ampl_large_freq + mean_ampl_large_freq_std     
            color_mean = "darkred"                  
        #########################################
        ## these variables are just for plotting. 
        sorted_indx = np.argsort(np.concatenate([xf_nrs1,xf_nrs2]))
        xf = np.concatenate([xf_nrs1,xf_nrs2])[sorted_indx]
        yf_spaxel = np.concatenate([yf_spaxel_nrs1,yf_spaxel_nrs2])[sorted_indx]
        wave_for_model = np.concatenate([wave_for_model_nrs1,wave_for_model_nrs2])
        spec_for_model = np.concatenate([spec_for_model_nrs1,spec_for_model_nrs2])
        model_spaxel = np.concatenate([model_spaxel_nrs1,model_spaxel_nrs2])
        wiggles_spec = np.concatenate([wiggles_spec_nrs1,wiggles_spec_nrs2])
        masked_lines_for_model = np.concatenate([masked_lines_for_model_nrs1,masked_lines_for_model_nrs2])
    else:
        xf,yf_spaxel = get_FFT(wave_for_model[masked_lines_for_model] ,wiggles_spec[masked_lines_for_model] )
        N = len(wave_for_model[masked_lines_for_model])
        ### GET FREQUENCY RANGE OF WIGGLE
        min_freq, max_freq = np.max([freq_range[0],5]), np.min([freq_range[1],50])  ### USUALLY FREQ < 5 is just noise
        short_freq_mask_1 = ( min_freq <= xf) & ( xf <= max_freq/2)
        short_freq_mask_2 = ( max_freq/2 < xf) & ( xf <= max_freq)
        large_freq_mask = ( (max_freq + 15) < xf) & ( xf <= (max_freq + 15) + 100)  ### Arbitarly add 15 [1/mu] as a buffer zone to sometimes avoid Broad features (emission/absorption)
        mean_ampl_large_freq  = np.mean( 2.0/N * np.abs(yf_spaxel[0:N//2])[large_freq_mask] )
        mean_ampl_large_freq_std = np.std( 2.0/N * np.abs(yf_spaxel[0:N//2])[large_freq_mask] )
        #
        mean_ampl_pixel_1  =  np.nanmean( (2.0/N * np.abs(yf_spaxel[0:N//2]) )[short_freq_mask_1] )
        mean_ampl_pixel_2  = np.nanmean( (2.0/N * np.abs(yf_spaxel[0:N//2]) )[short_freq_mask_2] )
        one_sigma_level =  mean_ampl_large_freq + mean_ampl_large_freq_std
        ### I USUALLY DIVIDE THE RANGE OF FREQ INTO 2, CAUSE SOMETIMES THERE IS ONLY ONE TYPE OF WIGGLES. SO USING THE WHOLE RANGE IS LESS EFFECTIVE
        #### DEFINE THRESHOLD TO DECIDE IF WIGGLES ARE PRESENT OR NOT
        spaxel_level = np.nanmax([mean_ampl_pixel_1,mean_ampl_pixel_2])
        color_mean = "r"
    if do_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(7)
        fig.set_figwidth(20)
        ax1.set_title("Wggles spectrum of the pixel")
        if smooth_spectrum == "no":
            ax1.plot(wave_for_model ,(spec_for_model  - model_spaxel),label="Spectrum",c="royalblue")
        if smooth_spectrum == "yes":
            ax1.plot(wave_for_model ,wiggles_spec,label="Smoothed Spectrum",c="blue")
        ax1.vlines( wave_for_model[~masked_lines_for_model] ,np.nanmin(wiggles_spec) -0.1,np.nanmax(wiggles_spec)+0.1,alpha=0.05,color='red',label="Masked lines")
        ax1.set_xlabel(r'wavelenght [$\mu m$]')
        ax1.set_ylabel('Flux ')
        ax1.set_ylim(np.nanmin(wiggles_spec)+np.nanmin(wiggles_spec),np.nanmax(wiggles_spec)*2)
        ax1.legend()
        ax2.set_title("Fourier Transform")
        if detector == 2:
            ax2.plot(xf_nrs1, 2.0/N_nrs1 * np.abs(yf_spaxel_nrs1[0:N_nrs1//2]) ,color='r',label='FFT Spectrum (NRS1)')
            ax2.plot(xf_nrs2, 2.0/N_nrs2 * np.abs(yf_spaxel_nrs2[0:N_nrs2//2]) ,color='darkred',label='FFT Spectrum (NRS2)')
        else:
            ax2.plot(xf, 2.0/N * np.abs(yf_spaxel[0:N//2]) ,color='r',label='FFT Spectrum')
        ax2.hlines(mean_ampl_large_freq ,min(xf),max(xf),color='green',label='Reference mean amplitude')
        ax2.hlines(one_sigma_level ,min(xf),max(xf),color='green',linestyles='dashed',label='1 sigma enhancement')
        ax2.hlines(spaxel_level ,min(xf),max(xf),color=color_mean,linestyles='solid')
        ax2.axvspan(min_freq, max_freq, alpha=0.15, color='r',label="Wiggle frequency regime")
        ax2.set_xlim(5,150)
        ax2.set_ylim(mean_ampl_large_freq-mean_ampl_large_freq,spaxel_level+spaxel_level)
        ax2.set_xlabel(r'frequency [$\mu m^{-1}$]')
        ax2.legend()
    if spaxel_level >= 0:
        return (spaxel_level - mean_ampl_large_freq)/mean_ampl_large_freq_std
    else:
        return 0.1 ##### IF FIT FAILED, RETURN A SMALL NUMBER I.E NO WIGGLES IS ASSUMED


def get_spec_ref_fft(wave,spec_ref,masked_lines,con_windows,detector):
    """  GET FOURIER TRANSFORM OF REFERENCE SPECTRUM. 
    I do this here one time to avoid calculating the FFT of the reference 
    spetrum everytime inside the loop, saving quite a bit of time. 

    Args:
        wave (array): wavelnght
        spec_ref (array): reference spectrum
        masked_lines (bool array): masked lines
        detector (string): nrs1 or nrs2. Detector where the wiggles are more prnounced. 

    Returns:
        _type_: _description_
    """
    #### GET FOURIER TRANSFORM OF REFERENCE SPECTRUM ###
    ## I DO THIS HERE TO AVID CALCULATING THIS A BUNCH OF TIME IN THE "FIT_WIGGLE_SPAXEL" FUNCTION, SAVING TIME
    ####
    wiggle_window_nrs1 = np.zeros(wave.shape,bool)
    wiggle_window_nrs2 = np.zeros(wave.shape,bool)
    wiggle_window_nrs1[150:1700] = True 
    wiggle_window_nrs2[-1400:-50] = True 
    wiggle_window_nrs1[~con_windows] = False ### EXCLUDE REGIONS OF CONTINUUM WITH STRONG EMISSION LINES IN CON_WINDOW
    wiggle_window_nrs2[~con_windows] = False ### EXCLUDE REGIONS OF CONTINUUM WITH STRONG EMISSION LINES IN CON_WINDOW

    wave_for_model_nrs1 = wave[wiggle_window]
    spec_ref_for_model_nrs1 = spec_ref[wiggle_window]
    masked_lines_for_model_nrs1 = masked_lines[wiggle_window]
    _, power_law_model_spec_ref = power_law_model_fit(wave_for_model, spec_ref_for_model,masked_lines_for_model )
    ######### LEFT THESE PLOTS HERE FOR DEBUGGING 
    #plt.plot(wave_for_model,power_law_model_spec_ref)
    #plt.plot(wave_for_model,spec_ref_for_model)
    wiggles_ref_spec = spec_ref_for_model - power_law_model_spec_ref
    wiggles_ref_spec = savgol_filter(wiggles_ref_spec,10,3)
    xf,yf_reference_spec = get_FFT(wave_for_model[masked_lines_for_model],wiggles_ref_spec[masked_lines_for_model])
    return yf_reference_spec , power_law_model_spec_ref

def get_wiggly_pixels(self,radius = 10, N_Cores = 1,smooth_spectrum="no",do_plots=False):
    """ Main function for finding wiggly spaxels. This function defines the number of Cores for the parallelization
    and passes the wavelenght, spectrum, reference spectrum and masked regions to the functions above. 

    Args:
        self = instace of Pipeline of wiggles, with center, masked_lines, and reference spectrum defined.
        N_Cores (int, optional): _description_. Defaults to 1.
        radius (flaot,optional): Search radius in pixels around the center. Default to 10 pixels. 
        reference_spectrum (str): "Inner" or "Outer". Choose the integrated spectrum to compare the FFT to.
    Returns:
        _type_: _description_
    """
    ### IMPORT ARGUMENTS
    wave, spec_ref_in,spec_ref_out, cube,ecube,masked_lines, nrs_detectors = self.wave, self.spec_ref_in,self.spec_ref_out,self.cube ,self.ecube, get_masked_regions(self), self.nrs_detectors

    #### GET FREQUENCY RANGE OF WIGGLES BASED ON THE POLYNOMIAL FIT OF THE CENTRAL PIXEL
    try:
        min_freq, max_freq = np.min(np.poly1d(self.best_freq_par)(np.array(self.wave))) , np.max(np.poly1d(self.best_freq_par)(np.array(self.wave)))
        freq_range = [min_freq,max_freq]
    except:
        print("NEED TO FIT CENTRAL PIXEL TO GET FREQUENCY TREND OF WIGGLES")
        sys.exit(1)
        
    ### GET FOURIER TRANSFORM FOR ANNULAR SPECTRUM, HERE NAMED SPEC_REF
    #if reference_spectrum == "Inner":
    #    spec_ref = spec_ref_in
    #else:
    #    spec_ref = spec_ref_out
    #
    #yf_spec_ref , power_law_model_spec_ref = get_spec_ref_fft(wave,spec_ref,masked_lines,self.con_windows,detector)
    ##
    pool = multiprocessing.Pool(N_Cores)
    power_law_task = []
    results = []
    power_law_stellar_models = []
    print("\n GETTING BEST-FIT MODELS FOR EACH PIXEL \n")
    #### ADD SPECTRUM TO MULTIPROCCESS TO FIND BEST-FIT MODEL 
    for y in range(self.cube.shape[1]):
        for x in range(self.cube.shape[2]):
             ##### GET PIXELS INSIDE DEFINDED SEARCH RADIUS
            if (y - self.nuc_y)**2 + (x- self.nuc_x)**2 < radius**2:
                spec = np.nan_to_num( cube[:,y,x], nan= np.nanmedian(cube[:,y,x][self.con_windows]))   
                espec =  np.nan_to_num( ecube[:,y,x] , nan= np.nanmedian(ecube[:,y,x][self.con_windows]) )   
                if np.median(spec) > 0:    
                    maxspec = np.median(spec[self.con_windows])
                    spec = spec / maxspec
                    spec = costume_sigmaclip(self.wave,spec) ######### SIGMA CLIPPING !! ##########
                    spec = spec * maxspec
                    if nrs_detectors == 2 :
                        maxspec = np.nanmax([ np.nanmax(spec[self.wave < self.gap_window[0]]), np.nanmax(spec[self.wave > self.gap_window[1]]) ])
                    else:
                        maxspec = np.nanmax(spec)
                    spec = spec / maxspec
                    espec = espec / maxspec * 2
                    espec[espec <= 0] = 1e-2
                    power_law_task.append([ (pool.apply_async( power_law_stellar_fit ,(wave,spec,espec,spec_ref_in,spec_ref_out,masked_lines) )),y,x,spec,espec,maxspec ] )
                else:
                    print("WARNING BAD PIXEL. PIXEL {} {} SKIPPED!".format(y,x))
                    continue
    ###### GETTING BEST-FIT MODEL
    for i in np.arange(len(power_law_task)):
        print("getting best-fit model for task {} of {}".format(i,len(power_law_task)))
        try:
            power_law_stellar_model = power_law_task[i][0].get()
            power_law_stellar_models.append([power_law_stellar_model,power_law_task[i][1:]])          
        except:
            print("could not get a power-law fit. Skipping pixel")
            continue      
    ##### ADD BEST-FIT MODELS AND SPECTRA TO MULTIPROCESS 
    for i in range(len(power_law_stellar_models)):
        ### unpack arguments saved on previous loops, I do this this way in case a pixels is skipped, the arrays of power-law models
        ### models and the rest will have different dimensions.    
        bestmodel_spaxel = power_law_stellar_models[i][0]
        y,x,spec,espec,maxspec = power_law_stellar_models[i][1]       
        results.append( (pool.apply_async( fit_wiggly_spaxels ,(freq_range,wave,spec,masked_lines,self.con_windows,bestmodel_spaxel,nrs_detectors,smooth_spectrum) ),y,x,spec,espec,maxspec,bestmodel_spaxel ) )
    #### GETTING FOURIER TRANFORM OF SPECTRA - MODEL (WIGGLE ARRAY) TO FLAGG WIGGLY PIXELS
    ima_wiggles = []
    Xs = []
    Ys =[]
    spectras = []
    especs = []
    maxspecs = []
    best_models = []
    for i in np.arange(len(results)):
        print("Finding Wiggles on pixel: {} of {}".format( results[i][2],results[i][1]))
        ima_wiggles.append(results[i][0].get())
        Xs.append(results[i][1])
        Ys.append(results[i][2])
        spectras.append(results[i][3])
        especs.append(results[i][4])
        maxspecs.append(results[i][5])
        best_models.append(results[i][6])
    if do_plots:
        fig = plt.figure(figsize=(9,6)) 
        im = plt.scatter(np.array(Ys),np.array(Xs),c=np.array(ima_wiggles),s=200,marker='s',cmap="cool")
        plt.scatter(self.nuc_x,self.nuc_y,marker="X",s=30,color="yellow")
        plt.colorbar(im, label='Wiggle Sigma Ratio')
        plt.xlabel("X PIXEL")
        plt.ylabel("Y PIXEL")
    
    return np.array(Xs),np.array(Ys),np.array(ima_wiggles),np.array(spectras) , np.array(especs),np.array(maxspecs) , np.array(best_models) ### X, Y, Wiggle Ratio, Spectrum, Variance, Normalization,  Best-fit Model

def define_affected_pixels(self,results, threshold=3,save_file=False):
    """ Simple function to create a boolean mask based on the resulting FFT fits and the 
    user defined threshold. I do this seperately and not automated so the user can have
    control on which thershold is more suitable. 

    Args:
        results (array): _description_
        threshold (int, optional): Number of sigma thresholds that defines a pixel affected by wiggles
        . Defaults to 1.

    Returns:
        bool array: array containg the coordinates of the pixels affected by wigglres. 
    """
    nuc_x,nuc_y = self.nuc_x,self.nuc_y 
    affected_pixels_mask =  results[2]  >= threshold
    #### ADD OR EXCLUDE PIXELS DEFINE BY THE USE ###
    if self.add_pixels !=None:
        self.add_pixels = np.array(self.add_pixels)
        for ind in range(len(self.add_pixels)):
            new_pix = np.where((results[1] == self.add_pixels[ind,0]) & ((results[0] == self.add_pixels[ind,1])))[0]
            affected_pixels_mask[new_pix] = True
    if self.exclude_pixels !=None:
        self.exclude_pixels = np.array(self.exclude_pixels)
        for ind in range(len(self.exclude_pixels)):
            new_pix = np.where((results[1] == self.exclude_pixels[ind,0]) & ((results[0] == self.exclude_pixels[ind,1])))[0]
            affected_pixels_mask[new_pix] = False
    fig = plt.figure(figsize=(7,7))
    im = plt.scatter(results[1], results[0] ,c=affected_pixels_mask,s=100,marker='s')
    plt.scatter(self.nuc_x,self.nuc_y,marker="X",s=30,color="red" )
    plt.scatter(0,0,marker="s",s=100,color="yellow",label="Affected Pixels" )
    plt.title("PIXELS AFFECTED FOR WIGGLES")
    #plt.colorbar(im)
    plt.xlabel("X offset")
    plt.ylabel("Y offset")
    plt.legend()
    plt.ylim(min(results[0]),max(results[0]))
    plt.xlim(min(results[1]),max(results[1]))
    affected_pixels_array = []
    for i in range(sum(affected_pixels_mask)):
        affected_pixels_array.append( [results[0][affected_pixels_mask][i],results[1][affected_pixels_mask][i],results[3][affected_pixels_mask][i], results[4][affected_pixels_mask][i], results[5][affected_pixels_mask][i], results[6][affected_pixels_mask][i]   ] )
    print("\n {} PIXELS AFFECTED BY WIGGLES".format(len(affected_pixels_array)))
    if save_file==True:
        np.savetxt(self.pathcube_input+"affected_pixels.csv", affected_pixels_array, delimiter=",", fmt='%s')
    return affected_pixels_array

def plot_wiggle_FFT(self,X, Y,smooth_spectrum):
    """ Quick plot of the Fourier tranform for the a single pixel. This can help determine if the user wants to add a not flagged pixel,etc. 

    Args:
        X (int): X pixel. Please follow the same oriantation as the plot for identifying affected pixels, X is horizontal axis
        Y (int): Y Pixel. Please follow the same oriantation as the plot for identifying affected pixels, Y is vertical axis
        reference_spectrum (str): "Inner" or "Outer". Choose the integrated spectrum to compare the FFT to.
        detector (str, optional): Chip side used to trace wiggles. Defaults to None.
    """
    #if reference_spectrum == "Inner":
    #    reference_spec =self.spec_ref_in        
    #else:
    #    reference_spec = self.spec_ref_out
    #yf_spec_ref , power_law_model_spec_ref = get_spec_ref_fft(self.wave,reference_spec,get_masked_regions(self),self.con_windows,detector)
    #### GET FREQUENCY RANGE OF WIGGLES BASED ON THE POLYNOMIAL FIT OF THE CENTRAL PIXEL
    try:
        min_freq, max_freq = np.min(np.poly1d(self.best_freq_par)(np.array(self.wave))) , np.max(np.poly1d(self.best_freq_par)(np.array(self.wave)))
        freq_range = [min_freq,max_freq]
    except:
        print("WARNING!!! YOU NEED TO FIT CENTRAL PIXEL FIRST TO GET FREQUENCY TREND OF WIGGLES")
        sys.exit(1)
    spec =  np.nan_to_num(self.cube[:,Y,X],nan= np.nanmedian(self.cube[:,Y,X][self.con_windows]))
    maxspec =np.median(spec[self.con_windows])
    spec = spec / maxspec
    spec = costume_sigmaclip(self.wave,spec) ######### SIGMA CLIPPING !! ##########
    spec = spec * maxspec
    if self.nrs_detectors == 2:
        maxspec = np.nanmax([ np.nanmax(spec[self.wave < self.gap_window[0]]), np.nanmax(spec[self.wave > self.gap_window[1]]) ])
    else:
        maxspec = np.nanmax(spec)
    spec = spec / maxspec
    espec =  np.nan_to_num( self.ecube[:,Y,X] , nan= np.nanmedian(self.ecube[:,Y,X][self.con_windows]) )   
    espec = espec / maxspec * 2
    espec[espec <= 0] = 1e-2
    plt.figure(figsize=(15,7))
    plt.title("Full Spectrum for this pixel")
    plt.plot(self.wave ,spec,c="blue")
    plt.xlabel(r'wavelenght [$\mu m$]')
    plt.show(block=True)

    best_model = power_law_stellar_fit(self.wave,spec,espec,self.spec_ref_in,self.spec_ref_out,get_masked_regions(self))
    sigma_ratio = fit_wiggly_spaxels(freq_range,self.wave,spec,get_masked_regions(self),self.con_windows,best_model,self.nrs_detectors,smooth_spectrum=smooth_spectrum,do_plots=True) 
    print("\n Sigma Ratio = {:.3f}".format(sigma_ratio))
    return 
