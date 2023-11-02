#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:07:08 2023

@author: evana
This code is meant to implement the overlap and add method in python reading in a sample audio file then playing it
the robotization effect is used in this code
"""

import numpy as np
import soundfile as sf
import sounddevice as sd
import time 
import os
import matplotlib.pyplot as plt

#chunk size and hop size
N = 1024
hop_size = int(N/2)
# 1. Read in signal 
signal, signal_sample_rate = sf.read('My_Vocals.wav')
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
    fft_signal = np.fft.fft(windowed_signal)

# 5. this is where processing would go with the fft output
# apply the robotization effect, which is where we remove phase from
# our fft so essentially abs(fft_values)
    Robotic_Signal = abs(fft_signal)
#6. apply IFFT to signal
    ifft_signal = np.fft.ifft(Robotic_Signal)

# 7. Move signal to output buffer, we will map it to a array the length of our 
# input signal. 
    try:
        output_signal[i:i+N] = ifft_signal + output_signal[i:i+N]
    except:
        None
# 8. Now move by Hop size samples and repeat process, Hop size is N/2 in our case

#play sound to see how it is
sd.play(signal, signal_sample_rate)
sd.wait()  # Wait until the sound is finished playing
sd.play(output_signal, signal_sample_rate)
sd.wait()  # Wait until the sound is finished playing

