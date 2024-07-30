# WICKED: Wiggle Corrector Kit for NIRSpec Data

## Introduction
WICKED (**W**iggle **I**nterference **C**orrector **K**it for **N**IRSp**E**c **D**ata) is a software package designed to remove sinusoidal wiggles, also known as Moire patterns, that appear in NIRSpec IFS data. These patterns arise due to the undersampling of the Point Spread Function (PSF).

## Overview
The Moire pattern can be modeled as a series of different sinusoidal waves. WICKED uses a mixture of integrated spectrum templates and power-laws to model single-pixel spectra and fits the residuals with a series of sinusoidal waves to effectively remove these wiggles.

## Features
- **Accurate Moire Pattern Correction**: Models and removes sinusoidal wiggles from NIRSpec IFS data.
- **Integrated Spectrum Templates**: Utilizes spectrum templates and power-laws for precise modeling.
- **Single-Pixel Spectrum Fitting**: Fits residuals with a series of sinusoidal waves for enhanced accuracy.

## Installation
To install WICKED, clone the repository on you local folder. There is no installation required for WiCKED, and example jupyter notebooks are also attached to the github repository showing how to run it. 

git clone https://github.com/yourusername/WICKED.git

