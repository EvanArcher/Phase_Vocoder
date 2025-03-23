#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:07:08 2023

@author: evana
This code is meant to implement the overlap and add method in python reading in a sample audio file then playing it
and using the YIN algorithim method for determining the frequency of the incoming audio signal
"""

import numpy as np
import soundfile as sf
import sounddevice as sd
import time 
import os
import matplotlib.pyplot as plt

# #chunk size and hop size
# N = 1024 # 1024 seems to be the optimal window size
# hop_size = int(N/2)
# # 1. Read in signal 
# signal, signal_sample_rate = sf.read('My_vocals.wav')
# try:
#     if signal.shape[1] == 2: # check if impulse is dual signal
#         signal = np.mean(signal, axis=1)
# except:
#     None

# #pad signal by mod N size to allow easier math and code below
# signal_pad_length = len(signal)%N
# signal = np.pad(signal,(0, signal_pad_length),mode='constant')
# output_signal = np.zeros(len(signal)) # holds our output for adding back to it

# # all of this is in a loop
# window = np.hamming(N)
# # 2. loop over signal in hop_size inrements
# for i in range(0, len(signal), hop_size):
#     chunk = signal[i:i+N] # load up chunk
# # 3. apply window to signal
#     chunk = np.pad(chunk, (0, N - len(chunk)), mode='constant')
#     windowed_signal = chunk*window
# # 4. apply fft to signal of size M>=N, we will use size N
#     fft_signal = np.fft.fft(windowed_signal)

# # 5. this is where processing would go with the fft output
# # apply the robotization effect, which is where we remove phase from
# # our fft so essentially abs(fft_values)
#     Robotic_Signal = abs(fft_signal)
# #6. apply IFFT to signal
#     ifft_signal = np.fft.ifft(Robotic_Signal)

# # 7. Move signal to output buffer, we will map it to a array the length of our 
# # input signal. 
#     try:
#         output_signal[i:i+N] = ifft_signal + output_signal[i:i+N]
#     except:
#         None
# # 8. Now move by Hop size samples and repeat process, Hop size is N/2 in our case

# #play sound to see how it is
# # sd.play(signal, signal_sample_rate)
# # sd.wait()  # Wait until the sound is finished playing
# sd.play(output_signal, signal_sample_rate)
# sd.wait()  # Wait until the sound is finished playing




# #Below are the funcitons needed for the YIN algorithim

# def CalculateDifferenceFunction(audioSignal, windowSize):
#     difference = np.zeros(windowSize)
#     for tau in range(0,windowSize-1):
#         for j in range(0,windowSize-1):
#             if j + tau < len(audioSignal):
#                 difference[tau] += (audioSignal[j] - audioSignal[j + tau])**2
#     return difference    
            
# def CalculateCumlativeMeanNormalizedDifferenceFunction(difference, windowSize):
#     d_prime = np.zeros(windowSize)
#     d_prime[0] = 1
#     for tau in range(1,windowSize-1):
#         runningSum = 0
#         for j in range(1,tau):
#             runningSum += difference[j]
#         d_prime[tau] = difference[tau]/(runningSum / tau)
#     return d_prime

# def AbsoluteThreshold(d_prime, threshold):
#     for tau in range(2,len(d_prime)-1):
#         if d_prime[tau] < threshold:
#             if all(d_prime[tau] < d_prime[tau + 1: tau+2]):
#                 return[tau]
#     return -1 #indicates no pitch found

def f(x):
    f_0 = 1
    envelope = lambda x: np.exp(-x)
    return np.sin(x * np.pi * 2 * f_0) * envelope(x)

def ACF(f, W, t, lag):
    return np.sum(
        f[t: t + W] *
        f[lag + t: lag + t + W]
        )

def DF(f, W, t, lag):
    return ACF(f, W, t, 0)\
        + ACF(f, W, t+lag, 0)\
            - ( 2 * ACF(f, W, t, lag))
            
def CMNDF(f, W, t, lag):
    if lag == 0:
        return 1 
    try:
        return DF(f, W, t, lag)\
            / np.sum([DF(f, W, t, j + 1 ) for j in range( lag ) ] ) * lag
    except:
        return 1

def detect_pitch(f, W, t, sample_rate, bounds, threshold = 0.1):
    CMNDF_vals = [CMNDF(f, W, t, i) for i in range(*bounds)]
    sample = None
    for i, val in enumerate(CMNDF_vals):
        if val < threshold:
            sample = i + bounds[0]
            break
    if sample is None:
        sample = np.argmin(CMNDF_vals) + bounds[0]
    return sample_rate / sample



def main():
    signal, signal_sample_rate = sf.read('My_vocals.wav')
    try:
        if signal.shape[1] == 2: # check if impulse is dual signal
            signal = np.mean(signal, axis=1)
    except:
        None
    window_size = int(5 / 2000 * signal_sample_rate)
    bounds = [20,2000]
    
    pitches = []
    for i in range(signal.shape[0] // (window_size + 3)):
        print(pitches)
        pitches.append(
            detect_pitch(
                signal,
                window_size,
                i * window_size,
                signal_sample_rate,
                bounds)
            )
    
    print(pitches)

main()
    


