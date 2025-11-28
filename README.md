# SCAD: Spectral Correction via Adaptive Deconvolution

## Quick Overview
A robust MATLAB framework for correcting poor microphone audio - removes echoes, resonances, and "flabby" bass response through stabilized inverse filtering.

## The Problem
Poor quality recordings often introduce linear distortions:
- Short echoes and reflections
- Resonant frequency peaks/dips  
- Temporal smearing ("flabby" sound)
- Comb filtering artifacts

## How It Works
1. **Blind estimation** of microphone impulse response
2. **Automatic model selection** using BIC
3. **Multi-criteria optimization** for stability
4. **Stabilized Wiener filtering** with practical safeguards

## Key Features
- ✅ Automatic parameter tuning
- ✅ Handles real-world noisy recordings
- ✅ Prevents common inverse filtering artifacts
- ✅ Multiple safety modes (safe/best/aggressive)
- ✅ Production-ready MATLAB code

## Usage
- Enter name/paths of input and output files
- Let the script run and listen to results

## Curious for more information? 
- Read the white paper
- Adjust parameters as needed for your system (ex custom impulse response)
