### IMPORTS #####
import numpy as np
from threadpoolctl import threadpool_limits
import multiprocessing
from FitWiggles import power_law_model_fit,power_law_stellar_fit,get_masked_regions,costume_sigmaclip
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq # Fourier Transform to fin spaxels affected by wiggles

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

def fit_wiggly_spaxels(wave,data,yf_spec_ref,spec_ref,masked_lines,con_windows,model_spaxel,chip_side, do_plots=False):
    
    """ This function takes the Fourier trasnform of a the wiggle residual spectrum of spaxel in a
    JWSST NIRSpec data cube and compares it to a the fourier transofrm of a reference spectrum.
    The output is a comparison of the median amplitud of the spectrum in the frequencies belonging to wiggles xf < 40
    and returns the ratio in units of standard deviaton. 
    """
    ##
    wiggle_window = np.zeros(wave.shape,bool)
    if chip_side == "left":
        wiggle_window[150:1700] = True 
    elif chip_side == "right":
        wiggle_window[-1400:-20] = True 
    wiggle_window[~con_windows] = False ### EXCLUDE REGIONS OF CONTINUUM WITH STRONG EMISSION LINES IN CON_WINDOW
    masked_lines_for_model = masked_lines[wiggle_window]
    ################### DEFINE SPECTRUMS IN THE WAVELENGHT RANGE TO BE MODELED ################
    wave_for_model = wave[wiggle_window]
    spec_for_model = data[wiggle_window]  
    spec_ref_for_model = spec_ref[wiggle_window]
    model_spaxel = model_spaxel[wiggle_window]
    ######### LEFT THESE PLOTS HERE FOR DEBUGGING 
    wiggles_spaxel = spec_for_model  - model_spaxel
    ############# FOURIER TRASNFORM OF SPAXEL AND REFERENCE SPECTRUM
    yf_reference_spec = yf_spec_ref
    xf,yf_spaxel = get_FFT(wave_for_model[masked_lines_for_model] ,wiggles_spaxel[masked_lines_for_model] )
    #### DEFINE THRESHOLD TO DECIDE IF WIGGLES ARE PRESENT OR NOT
    N = len(wave_for_model[masked_lines_for_model])
    short_freq_mask_1 = ( 5 < xf) & ( xf <= 20)
    short_freq_mask_2 = ( 20 < xf) & ( xf <= 40)
    large_freq_mask = ( 75 < xf) & ( xf <= 200)
    mean_ampl_integreted_spec_large_freq  = np.median( (2.0/N * np.abs(yf_spaxel[0:N//2]) - 2.0/N * np.abs(yf_reference_spec[0:N//2]))[large_freq_mask] )
    mean_ampl_integreted_spec_large_freq_std = np.std(( 2.0/N * np.abs(yf_spaxel[0:N//2]) - 2.0/N * np.abs(yf_reference_spec[0:N//2]))[large_freq_mask] )
    #
    mean_ampl_integreted_off_center_freq_1  =  np.mean( (2.0/N * np.abs(yf_spaxel[0:N//2])- 2.0/N * np.abs(yf_reference_spec[0:N//2]) )[short_freq_mask_1] )
    mean_ampl_integreted_off_center_freq_2  = np.mean( (2.0/N * np.abs(yf_spaxel[0:N//2])- 2.0/N * np.abs(yf_reference_spec[0:N//2]) )[short_freq_mask_2] )

    reference_level = mean_ampl_integreted_spec_large_freq_std
    one_sigma_level =  mean_ampl_integreted_spec_large_freq + 1*mean_ampl_integreted_spec_large_freq_std
    #
    spaxel_level_1 = (mean_ampl_integreted_off_center_freq_1 - mean_ampl_integreted_spec_large_freq  ) 
    spaxel_level_2 = (mean_ampl_integreted_off_center_freq_2 - mean_ampl_integreted_spec_large_freq  ) 
    max_ampl_integreted_spaxel = np.max([spaxel_level_1,spaxel_level_2])
    spaxel_level = max_ampl_integreted_spaxel # - mean_ampl_integreted_spec_large_freq 
    if do_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(7)
        fig.set_figwidth(15)
        ax1.set_title("Wggles spectrum of the pixel")
        ax1.plot(wave_for_model[masked_lines_for_model] ,0.06 +spec_ref_for_model[masked_lines_for_model] - np.nanmedian(spec_ref_for_model[masked_lines_for_model]),label="Reference Spectrum",c="orange")
        ax1.plot(wave_for_model[masked_lines_for_model] ,wiggles_spaxel[masked_lines_for_model],label="Spectrum",c="blue")
        ax1.vlines( wave_for_model[masked_lines_for_model] , -0.1,0.2,alpha=0.1,label="Allowed regions for fit")
        ax1.set_xlabel(r'wavelenght [$\mu m$]')
        ax1.set_ylabel('Flux ')
        ax1.legend()
        ax2.set_title("Fourier Transform")
        ax2.plot(xf, 2.0/N * np.abs(yf_spaxel[0:N//2]) - 2.0/N * np.abs(yf_reference_spec[0:N//2]),color='r',label='FFT of spectrum')
        ax2.hlines(mean_ampl_integreted_spec_large_freq ,min(xf),max(xf),color='green',label='reference median')
        ax2.hlines(one_sigma_level ,min(xf),max(xf),color='green',linestyles='dashed',label='1 sigma')
        ax2.hlines(max_ampl_integreted_spaxel ,min(xf),max(xf),color='red',linestyles='solid')
        ax2.set_xlim(5,200)
        ax2.set_xlabel(r'frequency [$\mu m^{-1}$]')
        ax2.legend()
    if spaxel_level >= 0:
        return spaxel_level / mean_ampl_integreted_spec_large_freq_std
    else:
        return 0.1 ##### IF FIT FAILED, RETURN A SMALL NUMBER I.E NO WIGGLES IS ASSUMED

def get_spec_ref_fft(wave,spec_ref,masked_lines,con_windows,chip_side):
    """  GET FOURIER TRANSFORM OF REFERENCE SPECTRUM. 
    I do this here one time to avoid calculating the FFT of the reference 
    spetrum everytime inside the loop, saving quite a bit of time. 

    Args:
        wave (array): wavelnght
        spec_ref (array): reference spectrum
        masked_lines (bool array): masked lines
        chip_side (string): left or right. Chunk of the spectrum where the wiggles are more prnounced. 

    Returns:
        _type_: _description_
    """
    #### GET FOURIER TRANSFORM OF REFERENCE SPECTRUM ###
    ## I DO THIS HERE TO AVID CALCULATING THIS A BUNCH OF TIME IN THE "FIT_WIGGLE_SPAXEL" FUNCTION, SAVING TIME
    ####
    wiggle_window = np.zeros(wave.shape,bool)
    if chip_side == "left":
        wiggle_window[150:1700] = True 
    elif chip_side == "right":
        wiggle_window[-1400:-50] = True 
    wiggle_window[~con_windows] = False ### EXCLUDE REGIONS OF CONTINUUM WITH STRONG EMISSION LINES IN CON_WINDOW
    wave_for_model = wave[wiggle_window]
    spec_ref_for_model = spec_ref[wiggle_window]
    masked_lines_for_model = masked_lines[wiggle_window]
    _, power_law_model_spec_ref = power_law_model_fit(wave_for_model, spec_ref_for_model,masked_lines_for_model )
    ######### LEFT THESE PLOTS HERE FOR DEBUGGING 
    #plt.plot(wave_for_model,power_law_model_spec_ref)
    #plt.plot(wave_for_model,spec_ref_for_model)
    wiggles_ref_spec = spec_ref_for_model - power_law_model_spec_ref
    xf,yf_reference_spec = get_FFT(wave_for_model[masked_lines_for_model],wiggles_ref_spec[masked_lines_for_model])
    return yf_reference_spec , power_law_model_spec_ref

def get_wiggly_pixels(self,N_Cores = 1,chip_side = None,do_plots=False):
    """ Main function for finding wiggly spaxels. This function defines the number of Cores for the parallelization
    and passes the wavelenght, spectrum, reference spectrum and masked regions to the functions above. 

    Args:
        self = instace of Pipeline of wiggles, with center, masked_lines, and reference spectrum defined.
        N_Cores (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if chip_side==None:
        if self.jwst_filter == "f170lp":
            chip_side = "right"
        if self.jwst_filter == "f290lp":
            chip_side = "left"
    ### IMPORT ARGUMENTS
    wave, spec_ref, cube,ecube,masked_lines = self.wave, self.spec_ref,self.cube ,self.ecube, get_masked_regions(self)
    ### GET FOURIER TRANSFORM FOR ANNULAR SPECTRUM, HERE NAMED SPEC_REF
    yf_spec_ref , power_law_model_spec_ref = get_spec_ref_fft(wave,spec_ref,masked_lines,self.con_windows,chip_side)
    ##
    pool = multiprocessing.Pool(N_Cores)
    power_law_task = []
    results = []
    power_law_stellar_models = []
    #### ADD SPECTRUM TO MULTIPROCCESS TO FIND BEST-FIT MODEL 
    for y in np.arange(10,40):
        for x in np.arange(10,40):
            spec = np.nan_to_num( cube[:,y,x], nan= np.nanmedian(cube[:,y,x][self.con_windows]))   
            espec =  np.nan_to_num( ecube[:,y,x] , nan= np.nanmedian(ecube[:,y,x][self.con_windows]) )   
            if np.median(spec) > 0:    
                maxspec = np.median(spec[self.con_windows])
                spec = spec / maxspec
                spec = costume_sigmaclip(self.wave,spec) ######### SIGMA CLIPPING !! ##########
                pec = spec * maxspec
                maxspec = np.nanmax([ np.nanmax(spec[self.wave < self.gap_window[0]]), np.nanmax(spec[self.wave > self.gap_window[1]]) ])
                spec = spec / maxspec
                espec = espec / maxspec * 2
                espec[espec <= 0] = 1e-2
                power_law_task.append([ (pool.apply_async( power_law_stellar_fit ,(wave,spec,spec_ref,masked_lines) )),y,x,spec,espec,maxspec ] )
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
        model_spaxel = power_law_stellar_models[i][0]
        y,x,spec,espec,maxspec = power_law_stellar_models[i][1]       
        results.append( (pool.apply_async( fit_wiggly_spaxels ,(wave,spec,yf_spec_ref,spec_ref,masked_lines,self.con_windows,model_spaxel,chip_side) ),y,x,spec,espec,maxspec,model_spaxel ) )
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
        plt.colorbar(im, label='Wiggle Sigma Ratio (pixel/reference)')
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
    fig = plt.figure(figsize=(7,7))
    im = plt.scatter(results[1], results[0] ,c=affected_pixels_mask,s=100,marker='s')
    plt.scatter(self.nuc_x,self.nuc_y,marker="X",s=30,color="red" )
    plt.scatter(0,0,marker="s",s=100,color="yellow",label="Affected spaxels" )
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

def plot_wiggle_FFT(self,X, Y ,chip_side = None):
    """ Quick plot of the Fourier tranform for the a single pixel. This can help determine if the user wants to add a not flagged pixel,etc. 

    Args:
        X (int): X pixel. Please follow the same oriantation as the plot for identifying affected pixels, X is horizontal axis
        Y (int): Y Pixel. Please follow the same oriantation as the plot for identifying affected pixels, Y is vertical axis
        chip_side (str, optional): Chip side used to trace wiggles. Defaults to None.
    """
    if chip_side==None:
        if self.jwst_filter == "f170lp":
            chip_side = "right"
        if self.jwst_filter == "f290lp":
            chip_side = "left"
    yf_spec_ref , power_law_model_spec_ref = get_spec_ref_fft(self.wave,self.spec_ref,get_masked_regions(self),self.con_windows,chip_side)
    
    spec =  np.nan_to_num(self.cube[:,Y,X],nan= np.nanmedian(self.cube[:,Y,X][self.con_windows]))
    maxspec =np.median(spec[self.con_windows])
    spec = spec / maxspec
    spec = costume_sigmaclip(self.wave,spec) ######### SIGMA CLIPPING !! ##########
    spec = spec * maxspec
    maxspec = np.nanmax([ np.nanmax(spec[self.wave < self.gap_window[0]]), np.nanmax(spec[self.wave > self.gap_window[1]]) ])
    spec = spec / maxspec
    plt.figure(figsize=(15,7))
    plt.title("Full Spectrum for this pixel")
    plt.plot(self.wave ,spec,c="blue")
    plt.xlabel(r'wavelenght [$\mu m$]')
    plt.show(block=True)

    best_model = power_law_stellar_fit(self.wave,spec,self.spec_ref,get_masked_regions(self))
    sigma_ratio = fit_wiggly_spaxels(self.wave,spec,yf_spec_ref,self.spec_ref,get_masked_regions(self),self.con_windows,best_model,chip_side,do_plots=True) 
    print("\n Sigma Ratio = {:.3f}".format(sigma_ratio))
    return 
