# WICKED: Wiggle Corrector Toolkit for NIRSpec Data

## Introduction
WiCKED (**W**iggle **I**nterference **C**orrector tool**K**it for **N**IRSp**E**c **D**ata) is a python package designed to remove sinusoidal wiggles, also known as Moire patterns, that appear in NIRSpec IFS data. These patterns arise due to the undersampling of the Point Spread Function (PSF).

## Overview
The Moire pattern can be modeled as a series of different sinusoidal waves. WiCKED uses a two different integrated spectrum templates a power-law and a second degree polynomial to model single-pixel spectra. The residual of the best-fit and the single-pixel spectrum is then fitted  a series of sinusoidal waves plus a constant to effectively remove these wiggles.

To flag pixels in the datacube, WiCKED calculates the Fourier Transfrom for the residual between the best-fit and the single-pixel spectrum with the Fourier Transform of the outer integrated spectrum (which has minimal wiggles due to it's larger integration aperture) and compares them. Pixels that present Moire patterns or wiggles, have significantly higher peaks at frequencies <50 [1/microns]

## Usage Examples
Here's a basic example of how to use WICKED:
from wicked import Wicked

# Initialize the WICKED corrector
corrector = Wicked()

# Load your NIRSpec IFS data
data = load_your_data_function()

# Apply the wiggle correction
corrected_data = corrector.correct(data)

# Save or analyze your corrected data
save_corrected_data_function(corrected_data)


## Features
- **Accurate Moire Pattern Correction**: Models and removes sinusoidal wiggles from NIRSpec IFS data.
- **Integrated Spectrum Templates**: Utilizes spectrum templates and power-laws for precise modeling.
- **Single-Pixel Spectrum Fitting**: Fits residuals with a series of sinusoidal waves for enhanced accuracy.

## Installation
To install WICKED, clone the repository on you local folder. There is no installation required for WiCKED, and example jupyter notebooks are also attached to the github repository showing how to run it. 

git clone https://github.com/yourusername/WICKED.git

## Dependencies
WICKED requires the following Python libraries:
- mgefit
- scipy
- photutils
- astropy

## Credits
Part of the code that fits the wiggles with many sinusoidal was inspired by the work of M. Perna. You can find their original repository https://github.com/micheleperna/JWST-NIRSpec_wiggles/tree/main.

## Contact
This python package is still under development. For any questions or feedback, please contact me via dumont@mpia.com or open an issue on GitHub.
