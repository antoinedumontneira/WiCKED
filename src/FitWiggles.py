## LIBRARY IMPORTS ####
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy import constants as k
from threadpoolctl import threadpool_limits
from mgefit import cap_mpfit as mpfit
from scipy.signal import savgol_filter
from astropy.io import fits
import random as rn
from photutils.centroids import  centroid_com
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt,find_peaks #  peak finder
from astropy.stats import sigma_clip # Remove ramaining outliers left by the standard pipeline
from scipy.fft import fft, fftfreq # Fourier Transform to fin spaxels affected by wiggles
import multiprocessing
import warnings
####### ####### ####### ####### ####### ####### #######
###### Functions for the Modelling of the wiggles #######
####### ####### ####### ####### ####### ####### #######
def costume_sigmaclip(wave,data,N_window=10,sigma=3):
    """ Costume sigma clipping based on the astropy.stats.sigma_clip package.
    The code looks for outliers bsed on the provided sigma level and replace them with the mean value around a user define window. 

    Args:
        wave (np.array): Wavelength array. Must be same lenght as data array
        data (np.array): data. 
        N_window (int, optional): Windown around outlier to calculate mean value. Defaults to 10.
        sigma (int, optional): sigma threshold for the sigma_clip. Defaults to 3.

    Returns:
        sigma_clipped_data: data with outliers removed and replaced with local mean.
    """
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    sigma_clip_mask = sigma_clip(data,masked=True,sigma=sigma).mask
    sigma_clipped_daata = data.copy()
    lendata = len(data)
    for ind in np.nonzero(sigma_clip_mask)[0]:
        loop_ind = ind
        ind_low = ind - round(N_window/2)
        ind_up = ind + round(N_window/2)
        if ind_low < 0:
            ind_low = 0
        if ind_up > lendata:
            ind_up = lendata - 1
            if ind_low - N_window > 0:
                ind_low =  ind - round(N_window)
                ind_up = ind
                loop_ind = ind - round(N_window/2)
        try:
            mean_ind = np.nanmean( [data[ind_low:ind].mean(),data[loop_ind:ind_up].mean() ]  )
        except:
            print('WARNING: cannot calculate local mean for outlier at wave = {:.2f} \n'.format(wave[ind]))
            print('will replace with global mean')
            plt.plot(wave[ind_low:ind_up],data[ind_low:ind_up])
            plt.xlabel(r'wavelength [$\mu m$]')
            plt.ylabel('Flux ')
            plt.show()
            mean_ind =  np.nanmean(data)
        sigma_clipped_daata[ind] = mean_ind
    return sigma_clipped_daata

def get_masked_regions(instance):
    """Get boolean array of the Wavelength to be masked during the power-law fit.
    
    Args:
       self: The instance of the FittWiggleClass with the ".lines_to_be_flagged" already defined. 
    Returns:
       allowed_wavelenghts: array with masked regions, gap + masked lines
    """
    if instance.nrs_detectors == 2:
        wave,gap_window,lines_to_be_flagged = instance.wave,instance.gap_window,instance.lines_to_be_flagged
        allowed_wavelenghts = np.ones(wave.shape,bool)
        allowed_wavelenghts[ (wave > gap_window[0]) & (wave < gap_window[1]) ] = False # exclude gap
    else:
        wave,lines_to_be_flagged = instance.wave,instance.lines_to_be_flagged
        allowed_wavelenghts = np.ones(wave.shape,bool)           
    for iltbf in range(len(lines_to_be_flagged)):
        allowed_wavelenghts[(wave > lines_to_be_flagged[iltbf][0]) & (wave < lines_to_be_flagged[iltbf][1])] = False
    return allowed_wavelenghts



############################################
########### POWER LAW FUNCTION ######
############################################ 
#### This models the power-law sprectum type of an AGN. I do not assign a specific physical meaning to this fit.
#### and it is basically to model difference in continuum between the reference spectrum and the fited spectrum.

def power_law_model_fit(wave,flux, masked_regions, do_plots = False) :
    
    """ This models the power-law sprectum type of an AGN. 
      I do not assign a specific physical meaning to this fit.
     and it is basically to model difference in continuum between
     the reference spectrum and the fited spectrum.
    
    Returns:
        flaot: Best fit power-law model.
    """


    ### IGNORE ANOYYING PYTHON WARNINGS
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide') 
    warnings.filterwarnings(action='ignore', message='overflow encountered in power') 
    warnings.filterwarnings(action='ignore', message='overflow encountered in multiply') 
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')

    flux_for_model = costume_sigmaclip(wave,flux)
    ########### POWER LAW FUNCTION ######
    power_law = lambda x,a1,b1,c: b1*x**a1 + c ### POWER-LAW
    ######################################
    wave_for_model = wave[masked_regions]
    flux_for_model = flux[masked_regions]
    
    size = 20
    flux_start = np.nanmedian(flux_for_model[:size])
    flux_end = np.nanmedian(flux_for_model[:size])
        
    if flux_start >  flux_end:
        guess = -0.5
    else:
        guess = 0.5
            
    N = 10  #### NUMBER OF ITERATIONS TO REPEAT POWER-LAW FIT WITH RANDOM GUESS SLOPE, WITH CENTER AT MU = GUESS     

    slope_guesses = np.random.normal( guess ,  0.1, N) 
    A1s = []
    B1s = []
    Cs = []
    for guess_i in slope_guesses:
        guesses = [guess_i,1,np.nanmedian(flux_for_model[:30])]
        power_law_guess = power_law(wave_for_model, guesses[0], guesses[1], guesses[2])
        popt,pcov = curve_fit(power_law, wave_for_model,flux_for_model, p0 = guesses,  method='trf', max_nfev = 10000) # need to set max iterations to larger than default else fit fails erroneously 
        A1s.append(popt[0]) ; B1s.append(popt[1]) ; Cs.append(popt[2])
    ###### RUN POWER-LAW FIT WITH MEAN GUESS OF RANDOM INITIAIONS    
    guesses = [np.mean(A1s),np.mean(B1s),np.mean(Cs)]
    power_law_guess = power_law(wave_for_model, guesses[0], guesses[1], guesses[2])
    popt,pcov = curve_fit(power_law, wave_for_model,flux_for_model, p0 = guesses,  method='trf', max_nfev = 10000) # need to set max iterations to larger than default else fit fails erroneously 
    a1_opt,b1_opt,c_opt = popt[0],popt[1],popt[2]
    #perr = np.sqrt(np.diag(pcov))
    #a_err,b_err,c_err = perr
    power_law_flux = power_law(wave,a1_opt,b1_opt,c_opt)
    if do_plots:
        plt.figure(figsize=(10,8))
        plt.plot(wave,power_law_flux, label = 'Power Law Fit')
        plt.plot(wave,flux,label='data')
        plt.xlabel(r'Wavelength [$\mu m$]',fontsize=15)
        plt.ylabel('Flux (a.u)',fontsize=15)
        if len(masked_regions):
            plt.vlines(wave[masked_regions], ymin = 0.02, ymax=max(flux), alpha=0.05, colors='k',label='Masked pixels')
        plt.legend()
    return flux, power_law_flux


def power_law_stellar_fit(wavelenght, data,espec,spec_ref_in,spec_ref_out,masked_regions,gap_mask=None,smooth_model=False):
    """ We assume that the fitted spectrum is a combination of the A*reference_sepctrum + B*power-law. 
    This routine finds the weights A and B that best fits the data. 

    Args:
        Wavelength (flaot): Wavelength array_
        data (float): spectrum to be fitted
        stellar_template (float): stellar reference spectrum
        masked_regions (bool): emission lines or outliers to be excluded during the fit. 
        normalize (bool, optional): If spectrum is not normalized, set to "True". Defaults to False.

    Returns:
        float: It returns an array with the best fit model. The best fit model is A*reference_sepctrum + B*power-law. 
    """
    ### IGNORE ANOYYING PYTHON WARNINGS
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide')    
    flux , power_law_template =  power_law_model_fit(wavelenght,data,masked_regions)
    ########## check if data was taken with 1 or 2 NRS detectors, to exclude the instrument gap. 
    ### define variables for the fitting
    if np.any(gap_mask)!=None:
        wave = wavelenght[gap_mask]
        data_fit = data[gap_mask]
        power_law_template_fit = power_law_template[gap_mask]
        spec_ref_in_fit = spec_ref_in[gap_mask]
        spec_ref_out_fit = spec_ref_out[gap_mask]
        espec_fit = espec[gap_mask]
    else: 
        wave = wavelenght
        data_fit = np.nan_to_num(data)
        power_law_template_fit = power_law_template
        spec_ref_in_fit = spec_ref_in
        spec_ref_out_fit = spec_ref_out
        espec_fit = espec
    if smooth_model==True: ## smooth the continuum of the spectrum to create a template to fit the data
        smooth_data_fit = savgol_filter(np.nan_to_num(data_fit),int(len(data_fit)/2),3) ## smoothing template with a width of 1.1 microns
        smooth_data = savgol_filter(np.nan_to_num(data),int(len(data)/2),3) ## smoothing template with a width of 1.1 microns
        def power_law_plus_annular_model(x,A,B,C,D,f1,f2,f3):
            total_model = A*power_law_template_fit + B*spec_ref_in_fit +C*spec_ref_out_fit + D*smooth_data_fit + f1*x**2 + f2*x + f3
            return total_model
        def power_law_plus_annular_reconstruc(x,A,B,C,D,f1,f2,f3):
            total_model = A*power_law_template + B*spec_ref_in +C*spec_ref_out + D*smooth_data + f1*x**2 + f2*x + f3
            return total_model
        guesses_polynomial = np.polyfit(wave, data_fit- smooth_data_fit, 2)
        guesses = [0.5,0.5,0.5,1.0,guesses_polynomial[0],guesses_polynomial[1],guesses_polynomial[2]] ## ASUME EUQAL CONTRIBUTION FROM BOTH TEMPLATES. We set the amplitud guess to 1.0
        popt,pcov = curve_fit(power_law_plus_annular_model,xdata= wave,ydata=data_fit,sigma=espec_fit,absolute_sigma= True, p0 = guesses,method='trf', max_nfev = 10000) # need to set max iterations to larger than default else fit fails erroneously 
        A_opt,B_opt,C_opt,D_opt,f1_opt,f2_opt,f3_opt = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6]
        power_law_flux_model = power_law_plus_annular_reconstruc(wavelenght,A_opt,B_opt,C_opt,D_opt,f1_opt,f2_opt,f3_opt)

    if smooth_model == False:
        def power_law_plus_annular_model(x,A,B,C,d1,d2,d3):
            total_model = A*power_law_template_fit + B*spec_ref_in_fit +C*spec_ref_out_fit + d1*x**2 + d2*x + d3 
            return total_model
        def power_law_plus_annular_reconstruc(x,A,B,C,d1,d2,d3):
            total_model = A*power_law_template + B*spec_ref_in +C*spec_ref_out + d1*x**2 + d2*x + d3 
            return total_model
        guesses_polynomial = np.polyfit(wave, data_fit - power_law_template_fit, 2)
        guesses = [0.33,0.33,0.33,guesses_polynomial[0],guesses_polynomial[1],guesses_polynomial[2]] ## ASUME EUQAL CONTRIBUTION FROM BOTH TEMPLATES
        popt,pcov = curve_fit(power_law_plus_annular_model,xdata= wave,ydata=data_fit,sigma=espec_fit,absolute_sigma= True, p0 = guesses,method='trf', max_nfev = 10000) # need to set max iterations to larger than default else fit fails erroneously 
        A_opt,B_opt,C_opt,d1_opt,d2_opt,d3_opt = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]
        power_law_flux_model = power_law_plus_annular_reconstruc(wavelenght,A_opt,B_opt,C_opt,d1_opt,d2_opt,d3_opt)
    #### AD EXTRA LOW-ORDER POLYNOMIAL FOR CONTINUUM MISTMATCHES 
    
    return power_law_flux_model

def make_plots(dictionary,lines_to_be_flagged, gap_window ,x,y):
                                                
    ### READ DICTIONARY
    wave = dictionary['wave']
    final_model = dictionary['final_model']
    wiggle_spectrum = dictionary["wiggles_spec"]
    spec_ref_in = dictionary["spec_ref_in"]
    spec_ref_out = dictionary["spec_ref_out"]
    power_law_stellar_model = dictionary["pw_law_stellar_model"]
    spec = dictionary["original_spec"]
    spec_corr = dictionary["corrected_spec"]
    nrs_detectors = 2
    if gap_window == 1:
        nrs_detectors = 1
    ########################################
    plt.close('fig')
    #global fig, ax, ax, cx, wave_um, lines_to_be_flagged, gap_window
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, height_ratios=[1, 0.5, 1])  # The middle subplot will be 1/2 as tall as the others
    fig.subplots_adjust(hspace=0.03)
    #panel with original spectra, and oscillations
    ax = plt.subplot(3, 1, 1)
    ax.set_ylim([-.08, 1.2])
    ax.set_xlim([wave[0], wave[-1]])
    ax.set_ylabel('flux (a.u.)',fontsize=15)
    ax.set_xticks([])
    ax.set_title("PIXEL = {} , {}".format(x,y))
    bx = plt.subplot(3, 1, 2)
    bx.set_ylim([-.3, 0.3])
    bx.set_xlim([wave[0], wave[-1]])
    bx.set_ylabel('flux (a.u.)',fontsize=15)
    #panel with corrected spectrum
    cx = plt.subplot(3,1,3)
    cx.set_ylim([-.08, 1.2])
    cx.set_xlim([wave[0], wave[-1]])
    cx.set_ylabel('flux (a.u.)',fontsize=15)
    cx.set_xticks([])

    for iltbf in range(len(lines_to_be_flagged)):
        ax.axvspan(lines_to_be_flagged[iltbf][0], lines_to_be_flagged[iltbf][1], alpha=0.1, color='red')
        cx.axvspan(lines_to_be_flagged[iltbf][0], lines_to_be_flagged[iltbf][1], alpha=0.1, color='red')
    if nrs_detectors == 2 :   
        ax.axvspan(gap_window[0], gap_window[1], alpha=0.2, color='orange')
        bx.axvspan(gap_window[0], gap_window[1], alpha=0.2, color='orange')
        cx.axvspan(gap_window[0], gap_window[1], alpha=0.2, color='orange')
    ax.plot(wave, spec_ref_in, color = 'k', label = 'Aperture spectrum',alpha=0.8)
    ax.plot(wave, spec_ref_out, color = 'orange', label = 'Annulus spectrum',alpha=0.5)
    ax.plot(wave, spec, label = 'single-spaxel spectrum')
    ax.plot(wave, power_law_stellar_model, color = 'blue', label = 'Best-Fit model',alpha=0.8)
    ax.legend(loc='upper right',prop={'size': 8}, mode = "expand", ncol = 3)
    bx.plot(wave, wiggle_spectrum, label = 'Wiggles', markersize= 1,color='gray')
    bx.plot(wave, final_model, label = 'Wiggles model',color='red',alpha=0.6)
    cx.plot(wave, spec_corr - power_law_stellar_model, color='gray', label='Reesidual', alpha=1)
    cx.plot(wave, spec_corr, color='darkblue', label='Corrected single-pixel spectrum', alpha=1)
    cx.legend(loc='upper right',prop={'size': 10}, mode = "expand", ncol = 2)
    plt.show(block=False)
    return 

def set_plot_panels(wave,lines_to_be_flagged,gap_window  ):
    plt.close('fig')
    #global fig, ax, ax, cx, wave_um, lines_to_be_flagged, gap_window
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, height_ratios=[1, 0.5, 1])  # The middle subplot will be 1/2 as tall as the others
    fig.subplots_adjust(hspace=0.03)
    nrs_detectors = 2
    if gap_window == 1:
        nrs_detectors = 1
    #panel with frequencies
    #ax = plt.subplot(3,1,3)
    #ax.set_xlim([wave[0],wave[-1]])
    #ax.set_ylim([-9,69])
    #ax.set_xlabel(r'obs.-frame wavelength ($\mu$m)')
    #ax.set_ylabel(r'frequency ($\mu$m$^{-1}$)')

    #panel with original spectra and templates
    ax = plt.subplot(3, 1, 1)
    ax.set_ylim([-.08, 1.2])
    ax.set_xlim([wave[0], wave[-1]])
    ax.set_ylabel('flux (a.u.)',fontsize=15)
    ax.set_xticks([])

    #panel with wiggles & wiggle model
    bx = plt.subplot(3, 1, 2)
    bx.set_ylim([-.3, 0.3])
    bx.set_xlim([wave[0], wave[-1]])
    bx.set_xticks([])

    #panel with corrected spectrum
    cx = plt.subplot(3,1,3)
    cx.set_ylim([-.08, 1.2])
    cx.set_xlim([wave[0], wave[-1]])
    cx.set_ylabel('flux (a.u.)',fontsize=15)
    cx.set_xlabel(r'wavelength ($\mu$m)', fontsize=15)
    

    for iltbf in range(len(lines_to_be_flagged)):
        #ax.axvspan(lines_to_be_flagged[iltbf][0], lines_to_be_flagged[iltbf][1], alpha=0.1, color='red')
        ax.axvspan(lines_to_be_flagged[iltbf][0], lines_to_be_flagged[iltbf][1], alpha=0.1, color='red')
        cx.axvspan(lines_to_be_flagged[iltbf][0], lines_to_be_flagged[iltbf][1], alpha=0.1, color='red')

    #ax.axvspan(gap_window[0], gap_window[1], alpha=0.2, color='orange')
    if nrs_detectors == 2:
        ax.axvspan(gap_window[0], gap_window[1], alpha=0.2, color='orange')
        cx.axvspan(gap_window[0], gap_window[1], alpha=0.2, color='orange')
    return ax,bx, cx 

# cos_function without continuum, for spectrum reconstruction
def cos_fn_reconstruc(p,x_model):
    """ Cosine function without added "extra" continuum. The best parameters are dereived
    using the function cos_fn() which includes a polynomial continuum. This continuum can 
    produce weird shapes in the continuum of the corrected spectrum, thus we leave it out. 

    Args:
        p (list): list of best fitted cosine parameters obtained for function cos_fn()
        x_model (array): best fited cosine function of the wigglers

    Returns:
        _type_: _description_
    """
    ### IGNORE ANOYYING PYTHON WARNINGS
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide') 
    
    model_freq   = p[0]
    Amp = p[1]
    y = Amp * np.cos(2 * np.pi * model_freq * x_model + np.pi/2*p[2])
    return y
def cos_fn_reconstruc2(p,x_model):
    """ Cosine function without added "extra" continuum. The best parameters are dereived
    using the function cos_fn() which includes a polynomial continuum. This continuum can 
    produce weird shapes in the continuum of the corrected spectrum, thus we leave it out. 

    Args:
        p (list): list of best fitted cosine parameters obtained for function cos_fn()
        x_model (array): best fited cosine function of the wigglers

    Returns:
        _type_: _description_
    """
    ### IGNORE ANOYYING PYTHON WARNINGS
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide') 
    
    model_freq1 = p[0]
    model_freq2 = p[1]
    Amp1 = p[2]
    Amp2 = p[3]

    y = Amp1 * np.cos(2 * np.pi * model_freq1 * x_model + np.pi/2*p[4]) + Amp2 * np.sin(2 * np.pi * model_freq2 * x_model)
    return y
def model(p,x_model, x=None , y=None, err=None,  fjac=None):
    ### IGNORE ANOYYING PYTHON WARNINGS
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide') 
    return (y- cos_fn_reconstruc(p,x_model))/err

def chisquare(p,x_model, x, y, err):
    ### IGNORE ANOYYING PYTHON WARNINGS
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide') 
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in scalar divide')
    mod = cos_fn_reconstruc(p,x_model)
    return np.sum((y-mod)**2/err**2.)

def chisquare_final(mod, y, err):
    ### IGNORE ANOYYING PYTHON WARNINGS
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide') 
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in scalar divide')
    return np.sum((y-mod)**2/err**2.)
def loop_for_fit_wiggles(spec,wiggle_spectrum,espec,power_law_stellar_model,maxspec,args,iplotF=1):   
    """ Loop section for fitting the wiggles.

    Args:
        spec (_type_): _description_
        wiggle_spectrum (_type_): _description_
        espec (_type_): _description_
        power_law_stellar_model (_type_): _description_
        maxspec (_type_): _description_
        args (_type_): _description_
        iplotF (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    wave,spec_ref_in,spec_ref_out,f_walls,lines_to_be_flagged,gap_window,x_model,N_rep,f0,df0i,bfi,center_spec = args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8],args[9],args[10],args[11]
    nrs_detectors = 2
    if gap_window == 1:
        nrs_detectors = 1
    global best_freq_par
    if center_spec == "no":
        best_freq_par = args[12]
    # correct_px is set to `r`
    # to repeat the fit as many times as needed to optimise the frequency trend 
    # This loop stops after 1 interation if f_walls is set to 2
    ### IGNORE ANOYYING PYTHON WARNINGS
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide') 
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    correct_px = 'r'
    while correct_px == 'r':

        # inizialization
        final_model = np.zeros((N_rep, spec_ref_in.size))
        totalL_bins = []
        totalF_bins = [] 
        succesful_Nrep = []
        # define the spectrum to be corrected
        if f_walls < 2: 
            if f_walls == 0:
                print("Constraining frequency trend of wiggles. \n")
        elif f_walls > len(df0i):
            print('too many iterations!')
            break
        
        if (iplotF == 0) & (f_walls==2):
            ax, bx, cx = set_plot_panels(wave, lines_to_be_flagged, gap_window)
        ###### THIS PARTS IS FOR FINDING THE OPTIOMAL WAY OF SPLITTING THE SPECTRUM INTO CHUNKS BASED ON IT'S PEAKS
        filter_spec = savgol_filter(wiggle_spectrum,10,3) ### smothing of ~0.02 microns for finding peaks easily
        if nrs_detectors == 2:
            filter_spec[(wave > gap_window[0]) & (wave < gap_window[1])] = 0.0  # exclude gap
        max_peaks, _ = find_peaks(filter_spec, distance=60,height=np.std(filter_spec)*0.1) ### peaks must be ~ 0.1 microns apart
        min_peaks, _ = find_peaks(filter_spec*-1, distance=60,height=np.std(filter_spec)*0.1)
        combined_peaks = np.sort(np.concatenate([min_peaks,max_peaks]))
        if f0 ==30: ## default prior is 30. This steps is skipped if users define it's own freq. prior
            f0 = 0.5/(np.mean(np.diff(combined_peaks))*(wave[1]-wave[0])) ### FREQ Prior based on the average distance between peaks. 
        if f_walls == 0:
            print("Freq. prior of wiggles = {:.1f} [1/mu]".format(f0))
        # repeat the fit for N_rep to increase the quality...
        for iN in range(N_rep):
            l_bins = []
            f_bins = []
            peak_selector = np.random.choice([0,1,2])
            if peak_selector == 0:
                min_random, max_random =  np.min([(int(max_peaks[0]) - 2 ) , 20 ]), np.min([ (wave.size-1) - max_peaks[-1], 20 ] )
                random_shift = np.random.randint(-min_random,max_random) ### RANDOMLY CHANGE THE INDEX OF THE PEAK BY ~0.04 MICRONS
                peaks = (max_peaks + random_shift)
            if peak_selector == 1:
                min_random, max_random =   np.min([(int(min_peaks[0]) - 2 ) , 20 ]), np.min([ (wave.size-1) - min_peaks[-1], 20 ] )
                random_shift = np.random.randint(-min_random,max_random) ### RANDOMLY CHANGE THE INDEX OF THE PEAK BY ~0.04 MICRONS
                peaks = (min_peaks + random_shift)
            if peak_selector == 2:
                min_random, max_random =   np.min([(int(combined_peaks[0]) - 2 ) , 20 ]), np.min([ (wave.size-1) - combined_peaks[-1], 20 ] )
                random_shift = np.random.randint(-min_random,max_random) ### RANDOMLY CHANGE THE INDEX OF THE PEAK BY ~0.04 MICRONS
                peaks = (combined_peaks + random_shift)
            peaks = np.insert(peaks,0,0) ### ADD FIRST ELEMENT, SO WE CAN FIT BETWEEN THE BEGINNING AND THE 1ST PEAK
                
            ##### START THE LOOP TO FIT WIGGLES WITH SIN FUNCTIONS
            for w_i in np.arange(len(peaks)):                     
                if (w_i) == len(peaks) - 1: ## FOR THE LAST PEAK IN THE SPECTRUM
                    if peaks[w_i] >=wave.size:  ## FOR THE LAST PEAK IN THE SPECTRUM
                        peaks[w_i] = wave.size - 60 #### IF THE RANDOM IND IS LARGER THAN THE WAVE, FIT IN A 60 PIX WINDOWN
                    flag_fit = (wave > wave[peaks[w_i]]) & (wave < wave[wave.size-1]) 
                    flag_mod = (wave > wave[peaks[w_i]]) & (wave < wave[wave.size-1])      
                else: ## FOR THE REST OF THE PEAKS IN THE SPECTRUM      
                    flag_fit = (wave > wave[peaks[w_i]]) & (wave < wave[peaks[w_i +1]]) 
                    flag_mod = (wave > wave[peaks[w_i]]) & (wave < wave[peaks[w_i +1]])                                 
                ### EXCLUDE MAKSED REGIONS FROM THE FITTING WAVELENGHT CHUNK.   
                if nrs_detectors == 2:
                    flag_fit[(wave > gap_window[0]) & (wave < gap_window[1])] = False  # exclude gap
                for iltbf in range(len(lines_to_be_flagged)):
                    flag_mod[(wave > lines_to_be_flagged[iltbf][0]) & (wave < lines_to_be_flagged[iltbf][1])] = False
                if nrs_detectors == 2:    
                    flag_mod[(wave > gap_window[0]) & (wave < gap_window[1])] = False  # exclude gap
                if np.isnan(wave[flag_mod].mean()) == True :
                    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
                    continue
                
                # model params ................................
                # 
                p = []
                # frequency
                try:
                    # if already defined in a previous fit, the frequency will be initialised using the 
                    # trend fw(lambda) shown in the 3rd panel of the figure below
                    f0 = np.poly1d(best_freq_par)(wave[flag_mod].mean())
                    p.append([np.max([f0,5]), np.max([f0-df0i[f_walls],5]), f0 + df0i[f_walls]])
                except NameError:
                    # otherwise, the frequency will be initialised using the parameter defined in the json file
                    
                    p.append([f0,  bfi[0], bfi[1]])
                    
                p.append([0.09, 0.001, 0.5])    # Amplitude 1
                p.append([0.5,  0.005, 0.995])  # phase
                parinfo = []
                parinfo = [{'value': p[i][0], 'fixed': 0, 'limited': [1, 1], 'limits': [p[i][1], p[i][2]]} for i in range(len(p))]
                # ................................
                fa = {'x_model':x_model[flag_mod],'x': wave[flag_mod], 'y': wiggle_spectrum[flag_mod] , 'err': espec[flag_mod]}
                m = mpfit.mpfit(model, parinfo=parinfo, functkw=fa, ftol=1e-15, xtol=1e-15, quiet=1)
                if (m.status <= 0):
                    #print ('error message = ', m.errmsg)        
                    continue
                best_chi2_mod = chisquare(m.params,x_model[flag_mod], wave[flag_mod], wiggle_spectrum[flag_mod], espec[flag_mod])/(wave[flag_mod].size - m.params.size)
                # repeat the fit, with random initializations for the initial parameters
                for k in range(30): #### ORIGINALLY SET TO 60
                    pbin = 20
                    for i in range(len(p)):
                        p[i][0]      = (int(rn.uniform(1, pbin))*(p[i][1]-p[i][2])/pbin +p[i][2]+1e-6)

                    parinfo = []
                    parinfo = [{'value': p[i][0], 'fixed': 0, 'limited': [1, 1], 'limits': [p[i][1], p[i][2]]} for i in range(len(p))]
                    m0 = mpfit.mpfit(model, parinfo=parinfo, functkw=fa, ftol=1e-15, xtol=1e-15, quiet=1)
                    
                    chi2_mod_new = chisquare(m0.params,x_model[flag_mod], wave[flag_mod], wiggle_spectrum[flag_mod] , espec[flag_mod])/(wave[flag_mod].size - m0.params.size)

                    if chi2_mod_new < best_chi2_mod:
                        m = m0
                        best_chi2_mod = chi2_mod_new

                    final_model[iN][flag_fit] = cos_fn_reconstruc(m.params,x_model[flag_fit])
                if (f_walls == 2) & (iplotF == 0):
                    ax.plot(wave[flag_mod], wiggle_spectrum[flag_mod], 'x', markersize=0.3,color='gray')
                l_bins.append(wave[flag_mod].mean())
                f_bins.append(m.params[0])
                #if i_ch == wave.size -1: i_ch +=1 # to avoid repetion of last step in the loop
            if iN <=1 :  ## SET REFERENCE FOR A CHI2 FOR THE WHOLE WIGGLE SPECTRUM
                best_chi2 =  chisquare_final(final_model[0,:],wiggle_spectrum,espec)
            else:
                #### HERE IT WILL COMPARE EACH LOOP MODEL AND ONLY SAVE THE ONES WITH BEST CHI2.
                #### ALSO TRAIN THE FREQUENCY TREND BASED ON THOSE MODELS. 
                loop_chi2 = chisquare_final(final_model[iN],wiggle_spectrum,espec)            
                if (len(l_bins)>0) & (loop_chi2<=(best_chi2 + 0.01*best_chi2)):    
                    succesful_Nrep.append(iN)
                    #best_chi2 = loop_chi2
                    for i in np.arange(len(l_bins)):
                        totalL_bins.append(l_bins[i])
                        totalF_bins.append(f_bins[i])
                    inds = np.array(totalL_bins).argsort()
                    sorted_f_bins = np.array(totalF_bins)[inds]
                    sorted_l_bins = np.array(totalL_bins)[inds]
                    mask_freq_channels = np.ones_like(sorted_l_bins, dtype='bool')
                    if nrs_detectors == 2:
                        mask_freq_channels[(sorted_l_bins > gap_window[0]) & (sorted_l_bins < gap_window[1])] = False
                    freq_par = np.polyfit(sorted_l_bins[mask_freq_channels],sorted_f_bins[mask_freq_channels] , 3)
                    freq_mod = np.poly1d(freq_par)(np.array(sorted_l_bins))
                    best_freq_par = freq_par
                
            if (iplotF == 0) & (f_walls==2) : 
                if (iN == 0):
                    ax.plot(wave[flag_mod], wiggle_spectrum[flag_mod], 'x', markersize=0.3,color='gray',label="Wiggles")
                    ax.plot(wave, spec_ref_in, color = 'k', label = 'Aperture spectrum',alpha=0.8)
                    ax.plot(wave, spec_ref_out, color = 'orange', label = 'Annulus spectrum',alpha=0.5)
                    ax.plot(wave, spec, color="red" ,label = 'Single-spaxel spectrum')
                    ax.plot(wave, power_law_stellar_model, color = 'blue', label = 'Best-Fit model',alpha=0.8)
                    bx.plot(wave, wiggle_spectrum, label = 'Wiggles', markersize= 1,color='gray')
                    bx.plot(wave[max_peaks],wiggle_spectrum[max_peaks],"X",c="green", markersize=5)
                    bx.plot(wave[min_peaks],wiggle_spectrum[min_peaks],"X",c="limegreen", markersize=5)
                    

        best_wiggle_model = savgol_filter(np.nanmedian(final_model, axis=0),5,3 ) #np.nanmedian(final_model, axis=0)
        
        spec_corr = spec -  best_wiggle_model#np.nanmean(final_model, axis=0) # Substract WIGGLE MODEL (COSINE) WITHOUT CONTINUUM
        min_freq_wiggle,max_frequ_wiggle =  str(np.min(np.round(sorted_f_bins))),str(np.max(np.round(sorted_f_bins))) # Get MIN & MAX freuqnecy of Wiggles to display them in the plot
        meam_frequ_wiggle =  str(np.round(np.nanmedian(sorted_f_bins))) # Get MIN & MAX freuqnecy of Wiggles to display them in the plot
        if (iplotF == 0) & (f_walls==2): 
            ax.legend(loc='upper right', prop={'size': 8}, mode = "expand", ncol = 3)
            #bx.text(.05, .93, 'Freq. range wiggles = ' + min_freq_wiggle + ' - ' +max_frequ_wiggle + r' [1/$\mu$]', ha='left', va='top',transform=bx.transAxes, bbox=dict(facecolor='none', edgecolor='red', boxstyle='round,pad=0.4'))
            bx.text(.05, .93, 'Median Freq. wiggles = ' + meam_frequ_wiggle + r' [1/$\mu$]', ha='left', va='top',transform=bx.transAxes, bbox=dict(facecolor='none', edgecolor='red', boxstyle='round,pad=0.4'))

            bx.plot(wave,best_wiggle_model, color = 'red',label="Wiggle Model" ,alpha = 0.6)
            bx.plot(wave,np.zeros(wave.size),"k--")
            bx.legend(loc='upper right')
            cx.plot(wave, spec_corr - power_law_stellar_model, color='gray', label='Residual', alpha=1)
            cx.plot(wave, spec_corr, color='red', label='Corrected single-spaxel spectrum', alpha=0.9)
            cx.legend(loc='upper right',prop={'size': 10}, mode = "expand", ncol = 2)
            plt.show(block=False)

        if f_walls != 2:
            print('50% OF THE {} ITERATIONS FINISHED'.format(N_rep))  
            correct_px = 'r' ### BY DEFAULT IT WILL RE ITERATE OVER TO GET A BETTER FIT
        else:
            correct_px = 'y' # with f_walls = 2 we directly fit the oscillations with solid constraints, and correct for them without looking at results
            
        f_walls += 2

        if correct_px == 'y':
            if center_spec == "yes":
                return spec_corr*maxspec, best_freq_par
            else:
                return spec_corr*maxspec , {"wave":wave,"wiggles_spec":wiggle_spectrum,"spec_ref_in":spec_ref_in,"spec_ref_out":spec_ref_out,"pw_law_stellar_model":power_law_stellar_model,
                                                "original_spec":spec,"corrected_spec":spec_corr,"final_model":best_wiggle_model
                                                }
    return


def fitwiggles(self,affected_pixels,nuc_y=None,nuc_x=None,N_rep=15,N_Cores=1,smooth_spectrum="y",do_plots=False):
    """ Main function to substract wiggles from NIRSpec cube. 
    This function prepares the spectrum for the pixels that have been flaged as affected by wiggles 
    with the function "define_affected_pixels()" and will find the best fit parameters with 
    "loop_for_fit_wiggles()". The division of the loop section for the fit is done for easines for Parallelization
    
    Args:
        affected_pixels (bool): array containing a 1s for pixels in the datacube affected by wiggles. 
        nuc_y (int, optional): X coordinte of the center of the datacube. Defaults to None.
        nuc_x (int, optional): Y coordinate of the center of the datacube. Defaults to None.
        N_rep (int, optional): Number of random intiations around a wavalenght chunk of the spectrum
        to find best-fit parameters of the cosine function. Defaults to 30.
        N_Cores (int, optional): Numbers of CPU cores use during the parallel fiting routine. Defaults to 1.
        do_plots (bool, optional): _description_. Defaults to False.
    """
    #### GET GLOBAL PARAMETERS FOR THE FITTING ######
    dt = self.wave[1] - self.wave[0]
    x_model = np.arange(self.wave.size)*dt
    f0 = self.frequency_prior
    df0i = self.df0i
    bfi = self.bfi
    con_windows = self.con_windows
    con_model_order = self.con_model_order
    spec_ref_in = self.spec_ref_in
    spec_ref_out = self.spec_ref_out
    wave = self.wave
    lines_to_be_flagged = self.lines_to_be_flagged
    if self.nrs_detectors == 2:
        gap_window = self.gap_window
    else:
        gap_window = self.nrs_detectors
    masked_lines = get_masked_regions(self)
    best_freq_par = self.best_freq_par
    f_walls = 2
    center_spec = "no"
    args = [wave,spec_ref_in,spec_ref_out,f_walls,lines_to_be_flagged,gap_window,x_model,N_rep,f0,df0i ,bfi,center_spec,best_freq_par]
    sorted_lf_bins = self.sorted_lf_bins
    #############################################################################
    ### RUN FOR REST OF FLAGGED SPAXELS ########
    pool = multiprocessing.Pool(N_Cores)   
    tasks = []
    PLOTS = []
    print("Adding affected spaxels to multiprocessing.pool \n")
    for i in range(len(affected_pixels)):
        ### unpack arguments saved on previous loops, I do this this way in case a pixels is skipped, the arrays of power-law models
        ### models and the rest will have different dimensions.
        power_law_stellar_model = affected_pixels[i][5]
        spec = affected_pixels[i][2]
        espec = affected_pixels[i][3]
        maxspec = affected_pixels[i][4]
        iy, ix = affected_pixels[i][0], affected_pixels[i][1]
        if ~((ix == self.nuc_y) & (iy == self.nuc_x)):
            if smooth_spectrum == "y":
                wiggle_spectrum = savgol_filter(spec - power_law_stellar_model,10,3)   ### APPLY A ~0.018 microns SMOOTHING TO MAKE WIGGLES MORE VISIBLE. ONLY RECOMMENDED FOR LOW-S/N. 
            else:
                wiggle_spectrum = costume_sigmaclip(wave,spec) - power_law_stellar_model ######### SIGMA CLIPPING OUTLIERS IN SPECTRUM!! THE ORIGINAL SPECTRUM IS NOT ALTERED, IT IS ONLY FOR THE WIGGLE SPECTRUM 
            #wiggle_spectrum = costume_sigmaclip(wave,wiggle_spectrum,sigma=3)
            tasks.append( (pool.apply_async( loop_for_fit_wiggles ,(  spec,wiggle_spectrum,espec,power_law_stellar_model,maxspec,args, 1 ) ),iy,ix ) ) 
    print("\n ##### START WIGGLE CORRECTION ##### \n")
    for i in np.arange(len(tasks)):
        print("Correcting wiggles on task: {} of {}".format(i,len(tasks)))
        try:
            corrected_spectrum,plot = tasks[i][0].get()
            sorted_lf_bins.append([tasks[i][2],tasks[i][1],corrected_spectrum])
            PLOTS.append(plot)
        except:
            print("could not get a fit. skipping this pixel")
            continue

    if do_plots:
        for i in range(len(PLOTS)):
            make_plots(PLOTS[i],lines_to_be_flagged,gap_window,tasks[i][2],tasks[i][1])
    
    ############ WRITE RESULTS ON DATA CUBE ########################################
    print("ADDING CORRECTED SPECTRA TO DATA CUBE \n")
    # read data cube 
    cube_output = self.pathcube_input + self.cube_name[:-5] + '_wicked.fits'
    hdu_01  = fits.open(self.pathcube_input + self.cube_name)
    for i in range(len(sorted_lf_bins)):
        hdu_01[1].data[:,sorted_lf_bins[i][1],sorted_lf_bins[i][0]] = sorted_lf_bins[i][2]

    hdu_01.writeto(cube_output, overwrite=True)

    print("DATA CUBE SAVED ON INPUT FOLDER, WITH NAME: {} \n".format(cube_output))
    print('\n FINISHED!')
    hdu_01.close()
    return

