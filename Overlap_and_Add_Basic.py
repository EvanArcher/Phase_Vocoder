#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:23:15 2023

@author: evana

This code is meant to implement the overlap and add method in python reading in a sample audio file then playing
No processing is done in this code rather it be used as an outline
"""

import numpy as np
import soundfile as sf
import sounddevice as sd
import time 
import os
import matplotlib.pyplot as plt
from Audio_Effects_Class import AudioEffects, LFO


#set audio class paramaters

samp_rate = (44100)*1 # 192kHz sampling rate
chunk = 2**12  # 100ms of data at 192kHz
delay_ms = 500 # Delay time in milliseconds
delay_gain = .9

effects = AudioEffects()
effects.basic_delay_init(delay_ms=delay_ms, samplerate=samp_rate, delay_gain=delay_gain)

effects.basic_noise_filter_init(chunk, .01)

#Vibrato set up
effects.vibrato_init(amplitude=0.1, frequency=100.0, sample_rate=samp_rate, waveform='square')


#chunk size and hop size
N = chunk
hop_size = int(N/2)
# 1. Read in signal 
signal, signal_sample_rate = sf.read('My_vocals.wav')
try:
    if signal.shape[1] == 2: # check if impulse is dual signal
        signal = np.mean(signal, axis=1)
except:
    None

#pad signal by mod N size to allow easier math and code below
signal_pad_length = len(signal)%N
signal = np.pad(signal,(0, signal_pad_length),mode='constant')
output_signal = np.zeros(len(signal)) # holds our output for adding back to it

# all of this is in a loop
window = np.hamming(N)
# 2. loop over signal in hop_size inrements
for i in range(0, len(signal), hop_size):
    chunk = signal[i:i+N] # load up chunk
# 3. apply window to signal
    chunk = np.pad(chunk, (0, N - len(chunk)), mode='constant')
    windowed_signal = chunk*window
# 4. apply fft to signal of size M>=N, we will use size N
    # fft_signal = np.fft.fft(windowed_signal)

# 5. this is where processing would go with the fft output
    processed_signal = chunk # clean signal
    processed_signal = effects.basic_noise_filter( effects.vibrato(chunk) )
    # processed_signal = effects.vibrato(chunk)
    
#6. apply IFFT to signal
    # ifft_signal = np.fft.ifft(fft_signal)

# 7. Move signal to output buffer, we will map it to a array the length of our 
# input signal. 
    try:
        output_signal[i:i+N] = processed_signal + output_signal[i:i+N]
    except:
        None
# 8. Now move by Hop size samples and repeat process, Hop size is N/2 in our case

#play sound to see how it is
# sd.play(signal, signal_sample_rate)
# sd.wait()  # Wait until the sound is finished playing
sd.play(output_signal, signal_sample_rate)
sd.wait()  # Wait until the sound is finished playing