# SCAD: Spectral Correction via Adaptive Deconvolution

## Quick Overview
SCAD is a robust MATLAB framework for restoring poor-quality audio using stabilized inverse filtering. It is designed for recordings affected by:
- short echoes and reflections
- resonant peaks and dips
- temporal smearing / "flabby" low end
- comb-filtering and mic coloration
- stationary or slowly varying linear distortion

## What SCAD Does
SCAD automatically searches for informative regions in the audio, estimates a stable inverse model from multiple candidate segments, and applies the learned correction to the full recording.

A web demo is available which processes files entirely locally, however beware it is **SLOW**. 

👉 https://scad.theivanr.duckdns.org/

For any large processing and big files use matlab code for a 10-100x speedup in performance

### Core pipeline
1. **Multi-resolution segment selection** across the full signal
2. **Conditioning-aware candidate ranking** to avoid weak or unstable windows
3. **Blind inverse model estimation** with alternating Wiener / least-squares updates
4. **Robust consensus kernel aggregation** across multiple segments
5. **Outer refinement passes** for self-consistent re-selection and retraining
6. **Stabilized Wiener deconvolution** with practical gain limits

## Key Features
- Automatic segment discovery across the whole file
- Multi-scale window analysis for transients and steady-state content
- Robust kernel averaging across multiple training regions
- Conditioning checks to reject poor inverse problems
- Outer-loop refinement for better consistency
- Stability safeguards against spectral blow-up
- Useful for vocals, guitars, ribbon microphones, vinyl, and other linear-ish restoration tasks

## Typical Use Cases
SCAD works especially well when the recording chain behaves approximately like a smooth, causal linear system, such as:
- old vocals and spoken word
- guitars and amp / cabinet coloration
- ribbon microphone recordings
- vinyl restoration material
- room coloration and mild echo correction

## Usage
1. Set input and output file paths.
2. Adjust the configuration knobs if needed.
3. Run the script.
4. Listen to the restored output.

## Notes
- SCAD assumes the dominant distortion is approximately linear and time-invariant over the selected regions.
- It is not intended for hard clipping, heavy saturation, or strongly time-varying effects.
- Better training windows usually come from energetic, structured, non-silent audio.

## Curious for More?
- Read the white paper
- Experiment with the configuration values
- Adapt the impulse-response length and selection strategy to your material

![SCAD Workflow](scad.png)
