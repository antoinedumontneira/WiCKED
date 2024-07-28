## LIBRARY IMPORTS ####
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as k
from threadpoolctl import threadpool_limits
from mgefit import cap_mpfit as mpfit
from scipy.signal import savgol_filter
from astropy.io import fits
import random as rn
from photutils.centroids import  centroid_com
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt #  peak finder
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
        wave (np.array): wavelenght array. Must be same lenght as data array
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
            plt.xlabel(r'wavelenght [$\mu m$]')
            plt.ylabel('Flux ')
            plt.show()
            mean_ind =  np.nanmean(data)
        sigma_clipped_daata[ind] = mean_ind
    return sigma_clipped_daata

def get_masked_regions(instance):
    """Get boolean array of the wavelenghts to be masked during the power-law fit.
    
    Args:
       self: The instance of the FittWiggleClass with the ".lines_to_be_flagged" already defined. 
    Returns:
       allowed_wavelenghts: array with masked regions, gap + masked lines
    """
    wave,gap_window,lines_to_be_flagged = instance.wave,instance.gap_window,instance.lines_to_be_flagged
    allowed_wavelenghts = np.ones(wave.shape,bool)
    allowed_wavelenghts[ (wave > gap_window[0]) & (wave < gap_window[1]) ] = False # exclude gap
                        
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
        plt.xlabel(r'Wavelenght [$\mu m$]')
        plt.ylabel('Flux')
        if len(masked_regions):
            plt.vlines(wave[masked_regions], ymin = 0.02, ymax=max(flux), alpha=0.05, colors='k',label='Masked pixels')
        plt.legend()
    return flux, power_law_flux


def power_law_stellar_fit(wavelenght, data,stellar_template,masked_regions,normalize=False):
    """ We assume that the fitted spectrum is a combination of the A*reference_sepctrum + B*power-law. 
    This routine finds the weights A and B that best fits the data. 

    Args:
        wavelenght (flaot): wavelenght array_
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
    
    if normalize == True : 
        maxspec =  np.nanmax([ np.nanmax(data[wavelenght < gap_window[0]]), np.nanmax(data[wavelenght > gap_window[1]]) ])
        data = data / maxspec
        
    flux , power_law_template =  power_law_model_fit(wavelenght,data,masked_regions)
    
    def power_law_plus_annular_model(x,A,B,C,d1,d2,d3):
        stellar_template_depper_lines = stellar_template**2/np.median(stellar_template**2)
        total_model = A*power_law_template + B*stellar_template +C*stellar_template_depper_lines + d1*x**2 + d2*x + d3
        return total_model
    data = np.nan_to_num(data)
    guesses_polynomial = np.polyfit(wavelenght, np.nan_to_num(data), 2)
    guesses = [0.33,0.33,0.33,guesses_polynomial[0],guesses_polynomial[1],guesses_polynomial[2]] ## ASUME EUQAL CONTRIBUTION FROM BOTH TEMPLATES
    popt,pcov = curve_fit(power_law_plus_annular_model,xdata= wavelenght,ydata=data, p0 = guesses,method='trf', max_nfev = 10000) # need to set max iterations to larger than default else fit fails erroneously 
    A_opt,B_opt,C_opt,d1_opt,d2_opt,d3_opt = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]

    power_law_flux_model = power_law_plus_annular_model(wavelenght,A_opt,B_opt,C_opt,d1_opt,d2_opt,d3_opt)
    #### AD EXTRA LOW-ORDER POLYNOMIAL FOR CONTINUUM MISTMATCHES 
    
    return power_law_flux_model

def make_plots(dictionary,lines_to_be_flagged, gap_window ,x,y):
                                                
    ### READ DICTIONARY
    wave = dictionary['wave']
    final_model = dictionary['final_model']
    spec_to_be_modelled = dictionary["wiggles_spec"]
    spec_ref = dictionary["spec_ref"]
    power_law_stellar_model = dictionary["pw_law_stellar_model"]
    spec = dictionary["original_spec"]
    spec_corr = dictionary["corrected_spec"]
    
    ########################################
    plt.close('fig')
    #global fig, ax, bx, cx, wave_um, lines_to_be_flagged, gap_window
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.03)
    #panel with original spectra, and oscillations
    bx = plt.subplot(2, 1, 1)
    bx.set_ylim([-.08, 1.2])
    bx.set_xlim([wave[0], wave[-1]])
    bx.set_ylabel('flux (a.u.)')
    bx.set_xticks([])
    bx.set_title("PIXEL = {} , {}".format(x,y))
    #panel with corrected spectrum
    cx = plt.subplot(2,1,2)
    cx.set_ylim([-.08, 1.2])
    cx.set_xlim([wave[0], wave[-1]])
    cx.set_ylabel('flux (a.u.)')
    cx.set_xticks([])

    for iltbf in range(len(lines_to_be_flagged)):
        bx.axvspan(lines_to_be_flagged[iltbf][0], lines_to_be_flagged[iltbf][1], alpha=0.1, color='red')
        cx.axvspan(lines_to_be_flagged[iltbf][0], lines_to_be_flagged[iltbf][1], alpha=0.1, color='red')
    bx.axvspan(gap_window[0], gap_window[1], alpha=0.2, color='orange')
    cx.axvspan(gap_window[0], gap_window[1], alpha=0.2, color='orange')
    bx.plot(wave, final_model, color = 'purple', alpha = 0.05)
    bx.plot(wave, spec, label = 'single-spaxel spectrum')
    bx.plot(wave, power_law_stellar_model, color = 'fuchsia', label = 'Power-law + integrated spectrum model')
    bx.plot(wave, spec_ref, color = 'orange', label = 'ntegrated spectrum model')
    bx.plot(wave, spec_to_be_modelled, label = 'wiggles', markersize= 1,color='grey')
    bx.plot(wave, final_model, label = 'wiggles model',color='red')
    bx.legend(loc='upper right')
    cx.plot(wave, spec_corr, color='darkblue', label='corrected single-spaxel spectrum', alpha=0.9)
    cx.plot(wave, spec_corr - power_law_stellar_model , color='grey', label='residuals', alpha=0.7)
    cx.legend(loc='upper right')
    plt.show(block=False)
    return 

def set_plot_panels(wave,lines_to_be_flagged,gap_window  ):
    plt.close('fig')
    #global fig, ax, bx, cx, wave_um, lines_to_be_flagged, gap_window
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.03)

    #panel with frequencies
    ax = plt.subplot(3,1,3)
    ax.set_xlim([wave[0],wave[-1]])
    ax.set_ylim([-9,69])
    ax.set_xlabel(r'obs.-frame wavelength ($\mu$m)')
    ax.set_ylabel(r'frequency ($\mu$m$^{-1}$)')

    #panel with original spectra, and oscillations
    bx = plt.subplot(3, 1, 1)
    bx.set_ylim([-.08, 1.2])
    bx.set_xlim([wave[0], wave[-1]])
    bx.set_ylabel('flux (a.u.)')
    bx.set_xticks([])

    #panel with corrected spectrum
    cx = plt.subplot(3,1,2)
    cx.set_ylim([-.08, 1.2])
    cx.set_xlim([wave[0], wave[-1]])
    cx.set_ylabel('flux (a.u.)')
    cx.set_xticks([])

    for iltbf in range(len(lines_to_be_flagged)):
        ax.axvspan(lines_to_be_flagged[iltbf][0], lines_to_be_flagged[iltbf][1], alpha=0.1, color='red')
        bx.axvspan(lines_to_be_flagged[iltbf][0], lines_to_be_flagged[iltbf][1], alpha=0.1, color='red')
        cx.axvspan(lines_to_be_flagged[iltbf][0], lines_to_be_flagged[iltbf][1], alpha=0.1, color='red')

    ax.axvspan(gap_window[0], gap_window[1], alpha=0.2, color='orange')
    bx.axvspan(gap_window[0], gap_window[1], alpha=0.2, color='orange')
    cx.axvspan(gap_window[0], gap_window[1], alpha=0.2, color='orange')
    return ax, bx, cx 
# cos_function 
def cos_fn(p,x_model):
    """ Function to model the residual spectrum containing the wiggles. 

    Args:
        p (list): initial guess parameter
        x_model (array): best fit model + continuum

    Returns:
        _type_: _description_
    """
    ### IGNORE ANOYYING PYTHON WARNINGS
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide') 
    model_freq   = p[0]
    Amp = p[1]
    y = Amp * np.cos(2 * np.pi * model_freq * x_model + np.pi/2*p[2]) + p[3]
    return y
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
    y = Amp * np.cos(2 * np.pi * model_freq * x_model + np.pi/2*p[2]) + p[3]*0
    return y
def model(p,x_model, x=None , y=None, err=None,  fjac=None):
    ### IGNORE ANOYYING PYTHON WARNINGS
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide') 

    return (y- cos_fn(p,x_model))/err

def chisquare(p,x_model, x, y, err):
    ### IGNORE ANOYYING PYTHON WARNINGS
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide') 
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in scalar divide')
    mod = cos_fn(p,x_model)
    return np.sum((y-mod)**2/err**2.)

def loop_for_fit_wiggles(spec,spec_to_be_modelled,espec,power_law_stellar_model,maxspec,args,iplotF=1):   
    """ Loop section for fitting the wiggles.

    Args:
        spec (_type_): _description_
        spec_to_be_modelled (_type_): _description_
        espec (_type_): _description_
        power_law_stellar_model (_type_): _description_
        maxspec (_type_): _description_
        args (_type_): _description_
        iplotF (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    wave,spec_ref,f_walls,lines_to_be_flagged,gap_window,x_model,N_rep,f0,df0i,bfi,center_spec = args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8],args[9],args[10]
    global best_freq_par,sorted_l_bins_cen,freq_mod_cen
    if center_spec == "no":
        best_freq_par,sorted_l_bins_cen,freq_mod_cen = args[11],args[12], args[13]
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
        final_model = np.zeros((N_rep, spec_ref.size))
        final_model_plot = np.zeros((N_rep, spec_ref.size))
        l_bins = []
        f_bins = []

        # define the spectrum to be corrected
        if f_walls < 2: 
            if f_walls == 0:
                print("Fitting routine requires 3 iterationes to achieve a good result. \n")
            print('iteration n. ', f_walls)
        elif f_walls > len(df0i):
            print('too many iterations!')
            break
        
        if (iplotF == 0) & (f_walls==2):
            #set plot panels
            ax, bx, cx = set_plot_panels(wave, lines_to_be_flagged, gap_window)

        # repeat the fit for N_rep to increase the quality...
        for iN in range(N_rep):
            Delta = 200  + np.random.randint(-60,200)
            for i_ch in range(Delta, wave.size + int(Delta / 2), int(Delta / 2)):

                # here I define two masks:
                # flag_mod, to define the model for the fit, and excluding all emission lines and features
                # which could affect the quality of the fit;
                # flag_fit, for the reconstruction of the oscillations with the best-fit results
                # the latter will present interpolations for the flag_mod masked regions
                if i_ch >= wave.size:
                    #pdb.set_trace()
                    i_ch = wave.size -1
                    flag_fit = (wave > wave[i_ch -int(Delta/2)]) & (wave < wave[i_ch]) # at long wave, wiggles have higher freq, and we can use smaller ranges.
                    flag_mod = (wave > wave[i_ch - int(Delta/2)]) & (wave < wave[i_ch])
                else:
                    
                    flag_fit = (wave > wave[i_ch - Delta]) & (wave < wave[i_ch])
                    #define another array for the modelled oscillations
                    flag_mod = (wave > wave[i_ch - Delta]) & (wave < wave[i_ch])
                flag_fit[(wave > gap_window[0]) & (wave < gap_window[1])] = False  # exclude gap
                
                for iltbf in range(len(lines_to_be_flagged)):
                    flag_mod[(wave > lines_to_be_flagged[iltbf][0]) & (wave < lines_to_be_flagged[iltbf][1])] = False
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
                    
                p.append([0.09, 0.005, 0.3])    # Amplitude
                p.append([0.5,  0.005, 0.995])  # phase
                p.append([0.03, -0.09, 0.09])   # continuum

                parinfo = []
                parinfo = [{'value': p[i][0], 'fixed': 0, 'limited': [1, 1], 'limits': [p[i][1], p[i][2]]} for i in range(len(p))]
                # ................................

                fa = {'x_model':x_model[flag_mod],'x': wave[flag_mod], 'y': spec_to_be_modelled[flag_mod], 'err': espec[flag_mod]}

                m = mpfit.mpfit(model, parinfo=parinfo, functkw=fa, ftol=1e-15, xtol=1e-15, quiet=1)
                if (m.status <= 0):
                    print ('error message = ', m.errmsg)

                chi2red = chisquare(m.params,x_model[flag_mod], wave[flag_mod], spec_to_be_modelled[flag_mod], espec[flag_mod])/(wave[flag_mod].size - m.params.size)

                # repeat the fit, with random initializations for the initial parameters
                for k in range(50): #### ORIGINALLY SET TO 60
                    pbin = 20
                    for i in range(len(p)):
                        p[i][0]      = (int(rn.uniform(1, pbin))*(p[i][1]-p[i][2])/pbin +p[i][2]+1e-6)

                    parinfo = []
                    parinfo = [{'value': p[i][0], 'fixed': 0, 'limited': [1, 1], 'limits': [p[i][1], p[i][2]]} for i in range(len(p))]
                    m0 = mpfit.mpfit(model, parinfo=parinfo, functkw=fa, ftol=1e-15, xtol=1e-15, quiet=1)

                    chi2red_new = chisquare(m0.params,x_model[flag_mod], wave[flag_mod], spec_to_be_modelled[flag_mod], espec[flag_mod])/(wave[flag_mod].size - m0.params.size)

                    if chi2red_new < chi2red:
                        m = m0
                        chi2red = chi2red_new

                    final_model[iN][flag_fit] = cos_fn_reconstruc(m.params,x_model[flag_fit])
                    final_model_plot[iN][flag_fit] = cos_fn(m.params,x_model[flag_fit])
                if (f_walls == 2) & (iplotF == 0):
                    bx.plot(wave[flag_mod], spec_to_be_modelled[flag_mod], 'x', markersize=0.3,color='grey')
                l_bins.append(wave[flag_mod].mean())
                f_bins.append(m.params[0])
                if i_ch == wave.size -1: i_ch +=1 # to avoid repetion of last step in the loop


            if (iplotF == 0) & (f_walls==2) : 
                if (iN == 0):
                    bx.plot(wave, spec, label = 'single-spaxel spectrum')
                    bx.plot(wave, power_law_stellar_model, color = 'fuchsia', label = 'Power-law + integrated spectrum model')
                    bx.plot(wave, spec_ref, color = 'orange', label = 'ntegrated spectrum model')
                    bx.plot(wave, spec_to_be_modelled, label = 'wiggles', markersize= 1,color='grey')
                bx.plot(wave, final_model[iN], color = 'purple', alpha = 0.05)


        l_bins = np.array(l_bins)
        f_bins = np.array(f_bins)

        inds = l_bins.argsort()
        sorted_f_bins = f_bins[inds]
        sorted_l_bins = l_bins[inds]
        # here I mask the wave channels close to the gap between detectors, and close to Ha emission,
        # where the fit is more difficult and the frequencies may significantly vary from the general trend
        mask_freq_channels = np.ones_like(sorted_l_bins, dtype='bool')
        mask_freq_channels[(sorted_l_bins > gap_window[0]) & (sorted_l_bins < gap_window[1])] = False

        freq_par = np.polyfit(sorted_l_bins[mask_freq_channels],sorted_f_bins[mask_freq_channels] , 5)
        freq_mod = np.poly1d(freq_par)(np.array(sorted_l_bins))

        if (iplotF == 0)  & (f_walls==2) : 
            ax.plot(sorted_l_bins, sorted_f_bins, 'o', markersize=3, label = 'best-fit frequency')
            ax.plot(sorted_l_bins, freq_mod, 'black', label = 'polynomial fit')
        
        if center_spec == "yes":
            freq_mod_cen = freq_mod + 0
            sorted_l_bins_cen = sorted_l_bins + 0
            
        spec_corr = spec - np.nanmean(final_model, axis=0) #+ model_con
        if (iplotF == 0) & (f_walls==2): 
            
            ax.plot(sorted_l_bins, np.interp(sorted_l_bins,sorted_l_bins_cen,freq_mod_cen), alpha = 0.5)
            bx.plot(wave, np.mean(final_model_plot,axis=0),  label = 'model', color = 'red')
            bx.legend(loc='upper right')
            cx.plot(wave, spec_corr, color='darkblue', label='corrected single-spaxel spectrum', alpha=0.9)
            cx.plot(wave, spec_corr - power_law_stellar_model, color='grey', label='residuals', alpha=0.7)
            cx.legend(loc='upper right')

            plt.show(block=False)

        if f_walls == 0: #only for central spaxel spectrum
            update_freq_par = 'y' #input('To update the reference trend for frequency, enter `y`')
            if update_freq_par == 'y':
                best_freq_par = freq_par


        if f_walls != 2:
            #print('Note that a few iterations might be required to decrease the scatter in the 3rd panel')
            print('ITERATION {} OF 2 FINISHED'.format(f_walls))            
            correct_px = 'r' ### BY DEFAULT IT WILL RE ITERATE OVER TO GET A BETTER FIT
        else:
            correct_px = 'y' # with f_walls = 2 we directly fit the oscillations with solid constraints, and correct for them without looking at results
            
        f_walls += 1

        if correct_px == 'y':
            if center_spec == "yes":
                return spec_corr*maxspec, best_freq_par, sorted_l_bins_cen, freq_mod_cen
            else:
                return spec_corr*maxspec , {"wave":wave,"wiggles_spec":spec_to_be_modelled,"spec_ref":spec_ref,"pw_law_stellar_model":power_law_stellar_model,
                                                "original_spec":spec,"corrected_spec":spec_corr,"final_model":np.mean(final_model_plot,axis=0),
                                                }
    return

def FitWiggles(self,affected_pixels,nuc_y=None,nuc_x=None,N_rep=15,N_Cores=1,do_plots=False):
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
    f0 = self.f0
    df0i = self.df0i
    bfi = self.bfi
    con_windows = self.con_windows
    con_model_order = self.con_model_order
    spec_ref = self.spec_ref
    wave = self.wave
    f_walls = 0
    lines_to_be_flagged = self.lines_to_be_flagged
    gap_window = self.gap_window
    masked_lines = get_masked_regions(self)
    best_freq_par = self.best_freq_par
    sorted_l_bins_cen = self.sorted_l_bins_cen
    freq_mod_cen = self.freq_mod_cen
    f_walls = 2
    center_spec = "no"
    args = [wave,spec_ref,f_walls,lines_to_be_flagged,gap_window,x_model,N_rep,f0,df0i ,bfi,center_spec,best_freq_par, sorted_l_bins_cen, freq_mod_cen]
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
            spec_to_be_modelled = spec - power_law_stellar_model 
            #spec_to_be_modelled = costume_sigmaclip(wave,spec_to_be_modelled,sigma=3)
            tasks.append( (pool.apply_async( loop_for_fit_wiggles ,(  spec,spec_to_be_modelled,espec,power_law_stellar_model,maxspec,args, 1 ) ),iy,ix ) ) 
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
    cube_output = self.pathcube_input + self.cube_input[:-5] + '_wigglycorrected.fits'
    hdu_01  = fits.open(self.pathcube_input + self.cube_input)
    for i in range(len(sorted_lf_bins)):
        hdu_01[1].data[:,sorted_lf_bins[i][1],sorted_lf_bins[i][0]] = sorted_lf_bins[i][2]

    hdu_01.writeto(cube_output, overwrite=True)

    print("DATA CUBE SAVED ON INPUT FOLDER, WITH NAME: {} \n".format(cube_output))
    print('\n FINISHED!')
    hdu_01.close()
    return

