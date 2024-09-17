# WICKED: **W**iggle **C**orrector tool**K**it for NIRSp**E**c **D**ata

## Introduction
WiCKED (**W**iggle **I**nterference **C**orrector tool**K**it for **N**IRSp**E**c **D**ata) is a python package designed to remove sinusoidal wiggles, also known as Moire patterns, that appear in NIRSpec IFS data. These patterns arise due to the undersampling of the Point Spread Function (PSF).

## Overview
The Moire pattern can be modeled as a series of different sinusoidal waves. WiCKED uses a two different integrated spectrum templates a power-law and a second degree polynomial to model single-pixel spectra. The residual of the best-fit and the single-pixel spectrum is then fitted  a series of sinusoidal waves plus a constant to effectively remove these wiggles.

To flag pixels affected by wiggles in the datacube, WiCKED calculates the Fourier Transfrom for the residual between the best-fit and the single-pixel spectrum and compares the mean amplitude at frequency were wiggles dominate (tipically f < 50 [1/micron]) and at longer wavelenghts. Pixels with a larger ratio between these two windows are flagged. The typical frequency of the wiggles is calculated based on the wiggles of the brightest pixel. 

## Usage Examples
Detailed examples on how to use WICKED are shown in the jupyter notebooks. Here's a basic example of how to use WICKED:
from wicked import Wicked
 ```
#### Initialize the WICKED corrector
run WiCKED.py
corrector = Wicked(Object_name=sourcename,pathcube=pathcube_input,cube_path=cube_input,redshift=z,jwst_filter=jwst_filter)

#### Get center and integrated spectrum templates
corrector.get_center(do_plots=True)
corrector.get_reference_spectrum(do_plots=True)

### Fit Central Pixel
corrector.FitWigglesCentralPixel()
### Flag single pixels affected by wiggles
from FIndWiggles import get_wiggly_pixels,define_affected_pixels

results = get_wiggly_pixels(corrector, N_Cores=NUMBER_OF_CPU,do_plots=True)
affected_pixels = define_affected_pixels(corrector,results)

#### Apply the wiggle correction to flagged pixels
from FitWiggles import FitWiggles
FitWiggles(corrector,affected_pixels,N_Cores=NUMBER_OF_CPU,do_plots=True)
#### DATA IS SAVED IN SAME FOLDER AS DATACUBE WITH THE _WIGGLECORRECTED EXTENSION

```


## Features
- **Accurate Moire Pattern Correction**: Models and removes sinusoidal wiggles from NIRSpec IFS data.
- **Integrated Spectrum Templates**: Utilizes spectrum templates and power-laws for precise modeling.
- **Single-Pixel Spectrum Fitting**: Fits residuals with a series of sinusoidal waves for enhanced accuracy.

## Installation
To install WICKED, clone the repository on you local folder. There is no installation required for WiCKED, and example jupyter notebooks are also attached to the github repository showing how to run it. 

git clone https://github.com/antoinedumontneira/WiCKED.git

## Dependencies
WICKED requires the following Python libraries:
- cap_mpfit
- scipy
- photutils
- astropy

## Credits

I got the inspiration to use a polynomial fit to characterize the frequency trend of Wiggles from the code by M. Perna. You can find their original repository [here](https://github.com/micheleperna/JWST-NIRSpec_wiggles/tree/main). Their code fits a polynomial to the frequency of wiggles at different wavelengths for the central and brightest pixel in the datacube, and then uses this as a prior for the rest of the pixels. I use a similar approach, but the polynomial is also tuned based on a chi-square fit for each pixel, since wiggles in different pixels sometimes have different frequencies.

## Citation
A full description of WICKED will be presented in the paper \italic{dumont et al 2024 (in prep.)}

## Contact
This python package is still under development. For any questions or feedback, please contact me via dumont (at) mpia.de
