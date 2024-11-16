#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:18:03 2023

@author: evana

this code is meant to implmenet the overlap and add basic method, except now 
we will do this on a live incoming signal, this will require extra hardware 
such as a raspberry pi
"""

import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import resample
import os
from Audio_Effects_Class import AudioEffects

print(sd.query_devices())
# Set default output device to your headphones (replace with correct device index)
sd.default.device = (1, 2)  # (input_device, output_device), only set output in this case

samp_rate = (44100)*2 # 192kHz sampling rate
chunk = 2**12  # 100ms of data at 192kHz
delay_ms = 500 # Delay time in milliseconds
delay_gain = 0.5

# Create an instance of the AudioEffects class with delay settings
effects = AudioEffects()
effects.basic_delay_init(delay_ms=delay_ms, samplerate=samp_rate, delay_gain=delay_gain)

#%% Callback function to handle streaming
# this is where all the magic is
def callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    # Apply the delay effect to the incoming block of audio
    delayed_output = effects.basic_delay(indata)  # Call the class method to process the audio

    # Output the processed signal (you can also adjust gain here)
    outdata[:] = delayed_output.reshape(outdata.shape)  # Amplify the delayed output if needed

#%% Start of stream this is where we input audio and get it back as outdata  
with sd.Stream(samplerate=samp_rate, blocksize=chunk, channels=1, callback=callback, latency=(0.001,0.001)):
    print("Press Enter to stop streaming...")
    input()

print("Streaming terminated.")







