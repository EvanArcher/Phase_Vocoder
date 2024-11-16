#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 21:05:48 2024

@author: evana
"""
import numpy as np
import sounddevice as sd

#print out devices avaliable
print(sd.query_devices())

# Parameters
fs = 44100  # Sampling rate
channels = 1 # Mono audio
buffer_size = 1024  # Block size

# Set default output device to your headphones (replace with correct device index)
sd.default.device = (1, 2)  # (input_device, output_device), only set output in this case

# Callback function for audio input and output
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print("Status:", status)  # Print any buffer overflows or other issues
    
    # Pass-through for real-time audio input to output
    outdata[:] = indata

# Set up the audio stream for both input and output
with sd.Stream(samplerate=fs, blocksize=buffer_size, channels=channels, dtype='float32',
               callback=audio_callback):
    print("Streaming audio. Press Ctrl+C to stop.")
    
    # Keep the stream running until the user interrupts (Ctrl+C)
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nStream stopped.")
        
        