# WICKED: **W**iggle **C**orrector tool**K**it for NIRSp**E**c **D**ata

## Introduction
WiCKED (**W**iggle **I**nterference **C**orrector tool**K**it for **N**IRSp**E**c **D**ata) is a python package designed to remove sinusoidal wiggles, also known as Moire patterns, that appear in NIRSpec IFS data. These patterns arise due to the undersampling of the Point Spread Function (PSF).

## Overview
The Moire pattern can be modeled as a series of different sinusoidal waves. WiCKED uses a two different integrated spectrum templates a power-law and a second degree polynomial to model single-pixel spectra. The residual of the best-fit and the single-pixel spectrum is then fitted  a series of sinusoidal waves plus a constant to effectively remove these wiggles.

To flag pixels affected by wiggles in the datacube, WiCKED calculates the Fourier Transfrom for the residual between the best-fit and the single-pixel spectrum and compares the mean amplitude at frequency were wiggles dominate (tipically f < 50 [1/micron]) and at longer wavelenghts. Pixels with a larger ratio between these two windows are flagged. The typical frequency of the wiggles is calculated based on the wiggles of the brightest pixel. 

## Usage Examples
A Detailed example on how to use WICKED is shown in the [example notebook](https://github.com/antoinedumontneira/WiCKED/blob/dev/Examples/Example_notebook.ipynb) in the Example folder.

## Installation
To install WICKED, you can pip install the repository on you local folder. There is no installation required for WiCKED.


pip install git+https://github.com/antoinedumontneira/WiCKED.git

## Dependencies
WICKED requires the following Python libraries:
- cap_mpfit
- threadpoolctl
- scipy
- photutils
- astropy

**WICKED** uses the  Michelle Cappellari version of the **mpfit** library included in the **mgefit** version 5.0.15 available for installation  [HERE!](https://pypi.org/project/mgefit/5.0.15/#files). **WICKED** will automatically install mgefit 5.0.15 in the enviroment

## Credits

I got the inspiration to use a polynomial fit to characterize the frequency trend of Wiggles from the code by M. Perna. You can find their original repository [here](https://github.com/micheleperna/JWST-NIRSpec_wiggles/tree/main). Their code fits a polynomial to the frequency of wiggles at different wavelengths for the central and brightest pixel in the datacube, and then uses this as a prior for the rest of the pixels. I use a similar approach, but the polynomial is also tuned based on a chi-square fit for each pixel, since wiggles in different pixels sometimes have different frequencies.

## Citation  

If you use **WICKED**, please cite our paper:  

**[WIggle Corrector Kit for NIRSpEc Data: WICKED]** – [Dumont] et al. (2025)   
https://ui.adsabs.harvard.edu/abs/2025arXiv250309697D/abstract 

## Contact
This python package is still under development. For any questions or feedback, please contact me via dumont (at) mpia.de
