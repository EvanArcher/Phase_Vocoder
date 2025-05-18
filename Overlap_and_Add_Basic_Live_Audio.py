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
import matplotlib.pyplot as plt
from Audio_Effects_Class import AudioEffects, LFO

print(sd.query_devices())
# # Set default output device to your headphones (replace with correct device index)
# sd.default.device = (0, 1)  # (input_device, output_device), only set output in this case

# samp_rate = (44100)*2 # 192kHz sampling rate
# chunk = 2**12  # 100ms of data at 192kHz
# delay_ms = 100 # Delay time in milliseconds
# delay_gain = .5

# # Create an instance of the AudioEffects class with delay settings
# effects = AudioEffects()
# effects.basic_delay_init(delay_ms=delay_ms, samplerate=samp_rate, delay_gain=delay_gain)
# effects.basic_noise_filter_init(chunk, 0.5)

# #%% Callback function to handle streaming
# # this is where all the magic is
# def callback(indata, outdata, frames, time, status):
#     if status:
#         print(status)
    
#     # Apply the delay effect to the incoming block of audio
#     # output_signal = effects.basic_delay_with_feedback(indata)  # Call the class method to process the audio
#     # output_signal = effects.basic_delay(indata)
    
#     # Messing with making sound noisy then denoising it with denoise function
#     signal_std = np.std(indata) #signal standard devitation
#     desired_noise_ratio  = .5 #NSR lol = noise/signal
        
#     noise = np.random.normal(0,1,indata.shape)
#     scaled_noise = noise * (signal_std * desired_noise_ratio)
    
#     output_signal = indata + scaled_noise
#     output_signal = effects.basic_noise_filter(output_signal) # feed in noisy signal
    
#     # print(delayed_output)
#     # Output the processed signal (you can also adjust gain here)
#     outdata[:] = output_signal.reshape(outdata.shape)  # Amplify the delayed output if needed

# #%% Start of stream this is where we input audio and get it back as outdata  
# with sd.Stream(samplerate=samp_rate, blocksize=chunk, channels=1, callback=callback, latency=(0.001,0.001)):
#     print("Press Enter to stop streaming...")
#     input()

# print("Streaming terminated.")








# LFO Testing
# Create an LFO instance (5 Hz sine wave, amplitude 1.0)
# Create an LFO instance: 5 Hz sine wave at 44100 Hz sample rate
sample_rate = 44100
effects = AudioEffects()
effects.vibrato_init(amplitude=0.10, frequency=2.0, sample_rate=44100, waveform='sine')


effects.vibrato_lfo_test()





