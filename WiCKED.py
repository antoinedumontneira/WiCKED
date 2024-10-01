## IMPORTS ####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import savgol_filter
import random as rn
import json, csv ,sys
import warnings
from astropy import constants as k
from threadpoolctl import threadpool_limits
from photutils.centroids import  centroid_quadratic
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt #  peak finder
from astropy.stats import sigma_clip # Remove ramaining outliers left by the standard pipeline
import multiprocessing
from FitWiggles import loop_for_fit_wiggles, costume_sigmaclip, get_masked_regions,power_law_model_fit,power_law_stellar_fit

class WICKED:
    """
    A breif description of the main parameters 

    - Object_name : name of the target (e.g. 'NGC4395')
    - pathcube : path of the input cube
    - cube_path : name of the input cube
    - redshift : redshift of the target
    - jwst_filter : name of NIRSpec filter ('f170lp' or 'f290lp')
    - nrs_detectors: 1 or 2. Number of NRS detectors in the data, i.e nrs1 & nrs2 or only one. 
    """
    def __init__(self,pathcube, cube_path,redshift):
        self.redshift = redshift
        self.pathcube_input = pathcube
        self.cube_input = cube_path
        # read data cube 
        hdu_01  = fits.open(self.pathcube_input + self.cube_input)
        self.cube = (hdu_01[1].data)
        self.ecube = (hdu_01[2].data)
        self.wave= ( hdu_01[1].header['CRVAL3']+ hdu_01[1].header['CDELT3']*np.arange(0, hdu_01[1].header['NAXIS3'])) #um
        self.jwst_filter = hdu_01[0].header["FILTER"]
        if hdu_01[0].header["DETECTOR"] == 'MULTIPLE':
            self.nrs_detectors = 2
        else:
            self.nrs_detectors = 1
        if self.jwst_filter == 'F100LP':
            if self.nrs_detectors == 2:
                self.gap_window = [1.435, 1.475]
            self.good_continuum = [[1.02,1.42],[1.5,1.86]] ## Default, excludes only instrument gap and edges
        if self.jwst_filter == 'F170LP':
            if self.nrs_detectors == 2:
                self.gap_window = [2.39, 2.475]
            self.good_continuum = [[1.7,2.36],[2.5,3.15]] ## Default, excludes only instrument gap and edges
        if self.jwst_filter == 'F290LP':
            if self.nrs_detectors == 2:
                self.gap_window = [3.99, 4.16]
            self.good_continuum = [[2.88,3.98],[4.2,5.26]] ###  Default, excludes only instrument gap       
        if self.nrs_detectors == 2:
            self.gap_mask = ~((self.wave > self.gap_window[0]) & (self.wave < self.gap_window[1]))
        # frequency parameters
        self.frequency_prior = 30 # initial estimate for the frequency of the wiggles. reasonable values are 20-40
        self.bfi = [0.5,50] # boundaries for the fit for the wiggles frequency, to be used for the 1d spectrum with highest S/N
        self.df0i =  [15, 10, 5, 5] # delta-frequence for the wiggles modelling
        self.DV = 600 # define the wavelength range around each emission line in km/s
        # parameters for the automatic detection of peaks
        self.min_peak_snr = 3
        self.peak_width = 1
        self.con_model_order = 3
        self.linefeatures = None ### PASS A .DAT FILE WITH KNOWN EMISSION/ABSORPTION LINES, must be a .dat file with 1 column lines in angstrom(vacuum) and the next with their names
        ### create empty arrays for flagged pixels
        self.exclude_pixels = None
        self.add_pixels = None
    ############################
    ### Functions ############
    ############################
    def mask_lines_and_peaks(self,data,do_plots = False): 
        """ Function to automatically detect emission lines. 
        ### Important:
          The user can also pass a file "linefeatures_vac.dat" containing a two colum file with 
          the wavelenght of known emission or absoprtion lines (in Angstroms) to be masked.
          If there is no file on the directory of the Main.py this step is skipped. 

        Args:
            data (array): Spectrum
            min_snr (int, optional): Minimum Singal-to-Noise ratio to be detected as an
            emission line and not just flagg noise. Defaults to 3.
            width (int, optional): Width in pixels of the emission line. Defaults to 1.

        Returns:
            N/A
        """
        c = k.c.to('km/s').value
        ### FLATTEN THE SPECTRUM TO FIND OUTLIERS

        wave_cont = self.wave
        model_cont_coef = np.polyfit(wave_cont,data,deg=9)
        model_cont = np.poly1d(model_cont_coef)(wave_cont)
        data = data - model_cont
        ###### Identify continuum emission and line features in the 1d spectrum  ################## 
        peaks_abs = find_peaks_cwt(-data, min_snr = self.min_peak_snr, widths = self.peak_width)
        peaks_em = find_peaks_cwt(data, min_snr = self.min_peak_snr, widths = self.peak_width)
        try:
            peaks = np.concatenate([peaks_em,peaks_abs])
        except:
            if peaks_em.length != 0:
                peaks = peaks_em
            else:
                peaks = peaks_abs
        ## the file linefeatures_vac.dat cointains the list of transitions that will be masked during the fit
        linefeatures = self.linefeatures
        if linefeatures != None:
            list_em_lines = pd.read_csv(linefeatures,sep=" ",header=None)[0].values
            self.list_em_lines = list_em_lines*(1+self.redshift)/1.e4 # obs-frame in micron
            self.list_em_lines = np.append(self.list_em_lines,self.wave[peaks])
        if linefeatures == None:
            self.list_em_lines = self.wave[peaks]
        self.lines_to_be_flagged = [[]]*len(self.list_em_lines)
        for i in range(len(self.list_em_lines)):
            self.lines_to_be_flagged[i] = [self.list_em_lines[i]*(1-self.DV/c), self.list_em_lines[i]*(1+self.DV/c)]
        
        if do_plots == True:
            fig2 = plt.figure(figsize=(7,7))
            zx1 = fig2.add_subplot(1,1,1) 
            if len(peaks):
                zx1.vlines(self.wave[peaks], ymin = 0.02, ymax=1, colors='k',label='fitted lines')
            zx1.vlines(list_em_lines,ymin = 0.02, ymax=1, colors='r',label='given mask lines')
            zx1.plot(self.wave,data)
            zx1.set_xlabel(r'obs.-frame wavelength ($\mu$m)')
            zx1.set_ylabel('flux (a.u.)')
            zx1.legend()
            plt.show(block=False)
        return


    def get_center(self,do_plots=False):
        """ Code to find the brightes x,y coordinates of the brightest pixels in tbe data cube.
        """
        #Find the center of the galaxy
        signal = np.zeros((np.shape(self.cube)[1], np.shape(self.cube)[2]))
        noise = np.zeros((np.shape(self.cube)[1], np.shape(self.cube)[2]))
        minlambda = min(self.wave)
        maxlambda = max(self.wave)
        for k in range(8,np.shape(self.cube)[1]-8):
            for l in range(8,np.shape(self.cube)[2]-8):
                    if self.nrs_detectors == 2:
                        uspec = np.nan_to_num( self.cube[self.gap_mask, k, l] )
                    else:
                        uspec = np.nan_to_num( self.cube[:, k, l] )
                    uspec = costume_sigmaclip(self.wave,uspec)
                    mean, sigma, subs = meanclip(uspec, clipsig=3, returnSubs=True)
                    signal[k, l] = mean
                    noise[k, l] = sigma
        xcen, ycen = centroid_quadratic(signal)
        self.nuc_x, self.nuc_y = round(xcen), round(ycen)
        if do_plots==True:
            fig = plt.figure(figsize=(7,7))
            zx0 = fig.add_subplot(1,1,1) 
            zx0.imshow(signal, origin = 'lower',cmap="Reds")
            zx0.plot(self.nuc_x,self.nuc_y,c="yellow",marker='X',markersize=10,label='Best-fit center')
            zx0.set_xlabel("X [offset]")
            zx0.set_ylabel("Y [offset]")
            zx0.legend()
            plt.show(block=False)
        print( '\n Center is  x, y = ', self.nuc_x, self.nuc_y)
        return   
        
    
    def get_reference_spectrum(self,in_radius=2,out_radius=4,nuc_x=None,nuc_y=None,do_plots=False):
        """ This code returns the integrated spectrum over a radii = nuc_r in pixels. 

        Args:
            nuc_x (int): X coordinate of the center of the cube. in pixel coordinates.
            nuc_y (int): Y coordinate of the center of the cube. in pixel coordinates.
            nuc_r (int): radius for aperture extraction.
        """
        if (nuc_x == None) & (nuc_y ==  None):
            try:
                nuc_y = self.nuc_y
                nuc_x = self.nuc_x
            except:
                print("WARNING!!!! : NO PIXELS PROVIDED FOR THE CENTER! ")
        global k
        self.con_windows = np.full((len(self.wave)), False)                    
        for iltbf in range(len(self.good_continuum)):
            self.con_windows[(self.wave > self.good_continuum[iltbf][0]) & (self.wave < self.good_continuum[iltbf][1])] = True
        
        if in_radius<=0:
            print("WARNING! in_radius must be >0!")
            print("\n Fix this! Stopping here")
            sys.exit()  # Exiting the script here
        
        spec_ref_in = np.zeros(self.cube.shape[0])
        spec_ref_out = np.zeros(self.cube.shape[0])
        for iy in range(self.cube.shape[1]):
            for jx in range(self.cube.shape[2]):
                if (iy - nuc_y)**2 + (jx- nuc_x)**2 < in_radius**2:
                    single_spec = np.nan_to_num(self.cube[:,iy,jx],nan= np.nanmedian(self.cube[:,iy,jx][self.con_windows]))  
                    median_spec = np.nanmedian(single_spec)
                    single_spec = costume_sigmaclip(self.wave,single_spec/median_spec) * median_spec
                    spec_ref_in += single_spec 
                if in_radius**2 <= (iy - nuc_y)**2 + (jx- nuc_x)**2 <= out_radius**2:
                    single_spec = np.nan_to_num(self.cube[:,iy,jx],nan= np.nanmedian(self.cube[:,iy,jx][self.con_windows]))  
                    median_spec = np.nanmedian(single_spec)
                    single_spec = costume_sigmaclip(self.wave,single_spec/median_spec) * median_spec
                    spec_ref_out += single_spec 
        #normalise the spectrum it its maximum (in det1 or det2)
        ##### IDENTIFY EMISSION LINES TO BE MASKED
        if self.nrs_detectors == 2:
            WICKED.mask_lines_and_peaks(self,data=spec_ref_in[self.gap_mask])  
            maxspec_ref_in = np.nanmax([np.nanmax(spec_ref_in[self.wave < self.gap_window[0]]), np.nanmax(spec_ref_in[self.wave > self.gap_window[1]])])
            maxspec_ref_out = np.nanmax([np.nanmax(spec_ref_out[self.wave < self.gap_window[0]]), np.nanmax(spec_ref_out[self.wave > self.gap_window[1]])])
        else:
            WICKED.mask_lines_and_peaks(self,data=spec_ref_in)  
            maxspec_ref_in = np.nanmax(spec_ref_in)
            maxspec_ref_out = np.nanmax(spec_ref_out)  

        spec_ref_in = spec_ref_in / maxspec_ref_in   
        spec_ref_out = spec_ref_out / maxspec_ref_out
        if do_plots==True:
            fig2 = plt.figure(figsize=(10,7))
            zx1 = fig2.add_subplot(1,1,1) 
            zx1.plot(self.wave,self.con_windows,linewidth=5,label='continuum window')
            zx1.plot(self.wave,spec_ref_in, label="Inner Integrated Spectrum",c="k",alpha=1)
            zx1.plot(self.wave,spec_ref_out, label="Outer Integrated Spectrum",c="orange",alpha=0.7)
            for iltbf in range(len(self.lines_to_be_flagged)):
                zx1.axvspan(xmin=self.lines_to_be_flagged[iltbf][0],xmax=self.lines_to_be_flagged[iltbf][1],ymin=0,ymax=1,color='red',alpha=0.2)
            zx1.axvspan(xmin=0,xmax=0.1,ymin=0,ymax=1,color='red',alpha=0.1,label='Masked Lines')
            if self.nrs_detectors == 2:
                zx1.axvspan(xmin=self.gap_window[0],xmax=self.gap_window[1],ymin=0,ymax=1,color='gold',alpha=0.2,label='Detector Gap')
            zx1.set_xlabel(r'obs.-frame wavelength ($\mu$m)')
            zx1.set_ylabel('flux (a.u.)')
            zx1.legend()
            zx1.set_ylim([-.08, 1.2])
            zx1.set_xlim(min(self.wave),max(self.wave))
            plt.show(block=False)
        self.spec_ref_in = spec_ref_in
        self.spec_ref_out = spec_ref_out
        return

    ##########################################
    ###### Modelling of the wiggles #######
    ##########################################
    
    # this is the function used to model the wiggles in 1d spectra.
    def FitWigglesCentralPixel(self,nuc_x=None,nuc_y=None,N_rep=30,iplotF = 0):
        """_summary_

        Args:
            affected_pixels (_type_): _description_
            N_rep (int, optional): _description_. Defaults to 30.
            N_Cores (int, optional): _description_. Defaults to 1.
            iplotF (int, optional): _description_. Defaults to 0.
        """
        #### GET GLOBAL PARAMETERS FOR THE FITTING ######
        dt = self.wave[1] - self.wave[0]
        x_model = np.arange(self.wave.size)*dt
        #ix, iy, f_walls = args[0], args[1], args[2]
        f0 = self.frequency_prior
        df0i = self.df0i
        bfi = self.bfi
        con_windows = self.con_windows
        con_model_order = self.con_model_order
        spec_ref_in = self.spec_ref_in
        spec_ref_out = self.spec_ref_out
        wave = self.wave
        f_walls = 0
        lines_to_be_flagged = self.lines_to_be_flagged
        if self.nrs_detectors == 2:
            gap_window = self.gap_window
        else:
            gap_window = self.nrs_detectors
        masked_lines = get_masked_regions(self)
        if nuc_y==None:
            nuc_y,nuc_x = self.nuc_y , self.nuc_x
        #############################################################################
        #### START THE FITTING FOR CENTRAL PIXEL, TO DEFINE THE FREQUENCY TREND #####
        sorted_lf_bins = []
        center_spec = "yes"
        ## READ SPECTRUM and ERROR SPECTRUM
        spec = np.nan_to_num(self.cube[:, nuc_y, nuc_x],nan= np.nanmedian(self.cube[:,nuc_y,nuc_x][self.con_windows]))
        maxspec =np.median(spec[con_windows])
        spec = spec / maxspec
        spec = costume_sigmaclip(wave,spec) ######### SIGMA CLIPPING !! ##########
        spec = spec * maxspec
        if self.nrs_detectors == 2:
            maxspec = np.nanmax([ np.nanmax(spec[wave < gap_window[0]]), np.nanmax(spec[wave > gap_window[1]]) ])
        else:
            maxspec = np.nanmax(spec) 
        spec = spec / maxspec
        espec =np.nan_to_num( self.ecube[:, nuc_y, nuc_y], nan=np.nanmedian(self.ecube[:, nuc_y, nuc_y][self.con_windows])  ) / maxspec * 2
        espec[espec <= 0] = 1e-2
        power_law_stellar_model = power_law_stellar_fit(wave,spec,espec,spec_ref_in,spec_ref_out,masked_lines)
        spec_to_be_modelled = spec - power_law_stellar_model 
        ### RUN FIT
        args = [wave,spec_ref_in,spec_ref_out,f_walls,lines_to_be_flagged,gap_window,x_model,N_rep,f0,df0i ,bfi,center_spec ]
        corrected_central_spectrum, self.best_freq_par  = loop_for_fit_wiggles(spec,spec_to_be_modelled,espec,power_law_stellar_model,maxspec,args, iplotF=0)
        sorted_lf_bins.append([self.nuc_x,self.nuc_y,corrected_central_spectrum ] )
        self.sorted_lf_bins = sorted_lf_bins
        return 


################################################################
### Function required for the setup of the pipeline ############
################################################################
        
def meanclip(image, 
        clipsig=3, maxiter=5,
        converge_num=0.02, verbose=False,
        returnSubs=False):
    """Computes an iteratively sigma-clipped mean on a data set
    Clipping is done about median, but mean is returned.
    Converted from IDL to Python.
    #D. Jones - 1/13/14
    This code is from the IDL Astronomy Users Library
    CALLING SEQUENCE:
        mean,sigma = meanclip( data, clipsig=, maxiter=,
                        converge_num=, verbose=,
                        returnSubs=False)
        mean,sigma,subs = meanclip( data, clipsig=, maxiter=,
                                converge_num=, verbose=,
                                returnSubs=True)
    INPUT PARAMETERS:
        data           -  Input data, any numeric array
    OPTIONAL INPUT PARAMETERS:
        clipsig        -  Number of sigma at which to clip.  Default=3
        maxiter        -  Ceiling on number of clipping iterations.  Default=5
        converge_num   -  If the proportion of rejected pixels is less
                        than this fraction, the iterations stop.  Default=0.02, i.e.,
                        iteration stops if fewer than 2% of pixels excluded.
        verbose        -  Set this flag to get messages.
        returnSubs     -  if True, return subscript array for pixels finally used
    
    RETURNS:
        mean           -  N-sigma clipped mean.
        sigma          -  Standard deviation of remaining pixels.
"""

    prf = 'MEANCLIP:  '

    #image = image.reshape(np.shape(image)[0]*np.shape(image)[1])
    
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in scalar divide')
    subs = np.where(np.isfinite(image))[0]
    ct = len(subs)
    iter=0

    for i in range(maxiter):
        skpix = image[subs]
        iter = iter + 1
        lastct = ct
        medval = np.nanmedian(skpix)
        mom = [np.nanmean(skpix),np.std(skpix)]
        sig = mom[1]
        wsm = np.where(np.abs(skpix-medval) < clipsig*sig)[0]
        ct = len(wsm)
        if ct > 0: subs = subs[wsm]         
        if (float(np.abs(ct-lastct))/lastct <= converge_num) or \
            (iter > maxiter) or (ct == 0):
            break
    #mom = moment(image[subs],double=double,max=2)
    mean = np.nanmean(image[subs])
    sigma = np.std(image[subs])
    if verbose:
        print(prf+strn(clipsig)+'-sigma clipped mean')
        print(prf+'Mean computed in ',iter,' iterations')
        print(prf+'Mean = ',mean,',  sigma = ',sigma)
    if not returnSubs:
        return(mean,sigma)
    else:
        return(mean,sigma,subs)         