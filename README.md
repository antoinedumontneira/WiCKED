# WICKED: Wiggle Corrector Kit for NIRSpec Data

## Introduction
WiCKED (**W**iggle **I**nterference **C**orrector **K**it for **N**IRSp**E**c **D**ata) is a python package designed to remove sinusoidal wiggles, also known as Moire patterns, that appear in NIRSpec IFS data. These patterns arise due to the undersampling of the Point Spread Function (PSF).

## Overview
The Moire pattern can be modeled as a series of different sinusoidal waves. WiCKED uses a two different integrated spectrum templates a power-law and a second degree polynomial to model single-pixel spectra. The residual of the best-fit and the single-pixel spectrum is then fitted  a series of sinusoidal waves plus a constant to effectively remove these wiggles.

To flag pixels in the datacube, WiCKED calculates the Fourier Transfrom for the residual between the best-fit and the single-pixel spectrum with the Fourier Transform of the outer integrated spectrum (which has minimal wiggles due to it's larger integration aperture) and compares them. Pixels that present Moire patterns or wiggles, have significantly higher peaks at frequencies <50 [1/microns]

## Features
- **Accurate Moire Pattern Correction**: Models and removes sinusoidal wiggles from NIRSpec IFS data.
- **Integrated Spectrum Templates**: Utilizes spectrum templates and power-laws for precise modeling.
- **Single-Pixel Spectrum Fitting**: Fits residuals with a series of sinusoidal waves for enhanced accuracy.

## Installation
To install WICKED, clone the repository on you local folder. There is no installation required for WiCKED, and example jupyter notebooks are also attached to the github repository showing how to run it. 

git clone https://github.com/yourusername/WICKED.git

## Contact
This python package is still under development. For any questions or feedback, please contact me via dumont@mpia.com or open an issue on GitHub.
