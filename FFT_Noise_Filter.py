#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:25:08 2024

@author: evana
This code is made to do overlap and add but use the fft to cancel out noise
"""


import numpy as np
import soundfile as sf
import sounddevice as sd
import time 
import os
import matplotlib.pyplot as plt

#chunk size and hop size
N = 2**9
hop_size = int(N/2)
# 1. Read in signal 
signal, signal_sample_rate = sf.read('My_vocals.wav')
try:
    if signal.shape[1] == 2: # check if impulse is dual signal
        signal = np.mean(signal, axis=1)
except:
    None
    
signal_std = np.std(signal) #signal standard devitation
desired_noise_ratio  = .5 #NSR lol = noise/signal
    
noise = np.random.normal(0,1,signal.shape)
scaled_noise = noise * (signal_std * desired_noise_ratio)

noisy_signal = signal + scaled_noise



#pad signal by mod N size to allow easier math and code below
signal_pad_length = len(noisy_signal)%N
noisy_signal = np.pad(noisy_signal,(0, signal_pad_length),mode='constant')
output_signal = np.zeros(len(noisy_signal)) # holds our output for adding back to it

#______________________________________________________________
fft_noisy = np.fft.fft(scaled_noise[0:N]) # reference for power of noisy signal
PSD = np.abs(fft_noisy * np.conj(fft_noisy) / N) # <- Power spectrum density per frame, why we divide by N
max_noise_level = max(np.abs(fft_noisy))
max_noise_PSD = max(PSD)
case = 4


# all of this is in a loop
window = np.hamming(N)
# 2. loop over signal in hop_size inrements
for i in range(0, len(noisy_signal), hop_size):
    chunk = noisy_signal[i:i+N] # load up chunk
# 3. apply window to noisy_signal
    chunk = np.pad(chunk, (0, N - len(chunk)), mode='constant')
    windowed_noisy_signal = chunk*window
# 4. apply fft to noisy_signal of size M>=N, we will use size N
    fft_noisy_signal = np.fft.fft(windowed_noisy_signal)

    if case == 1:
        # 5. this is where processing would go with the fft output
        # we will process away the noise, or atleast attempt to, by subtracting the noise floor
        fft_magnitude = np.abs(fft_noisy_signal) - max_noise_level
        fft_magnitude = np.maximum(fft_magnitude,0)
        filtered_signal = fft_magnitude * np.exp(1j * np.angle(fft_noisy_signal))
    if case == 2:       
        # 5b. this is another attempt to process the information
        # here we will zero out unused frequencies
        threshold = max_noise_level *0.25 #don't want to do max bad results
        filtered_signal = np.where(np.abs(fft_noisy_signal) >= threshold, fft_noisy_signal, 0)
    if case == 3:
        # 5c. lets try a smoothing filter, we will take averages of the signal as we go through
        # take averages of it, this is a time domain filter
        smoothing_window_size = 8
        kernel = np.ones(smoothing_window_size)/smoothing_window_size
        padded_kernel = np.pad(kernel,(0,N-len(kernel)),mode='constant')
        kernel_fft = np.fft.fft(padded_kernel)
        filtered_signal = fft_noisy_signal*kernel_fft #convolve together
    if case == 4: 
        #5d. Now we are gonna use the PSD to find our signal in the noise
        signal_PSD = np.abs(fft_noisy_signal * np.conj(fft_noisy_signal)) / N 
        indicies = signal_PSD > max_noise_PSD #grab indicies greater than noise, should be signal
        signal_PSD_Clean = signal_PSD * indicies
        filtered_signal = fft_noisy_signal * indicies

#6. apply IFFT to noisy_signal
    ifft_noisy_signal = np.fft.ifft(filtered_signal)

# 7. Move noisy_signal to output buffer, we will map it to a array the length of our 
# input noisy_signal. 
    try:
        output_signal[i:i+N] = ifft_noisy_signal + output_signal[i:i+N]
    except:
        None
# 8. Now move by Hop size samples and repeat process, Hop size is N/2 in our case

#play sound to see how it is
# sd.play(signal, signal_sample_rate)
# sd.wait()  # Wait until the sound is finished playing

sd.play(noisy_signal, signal_sample_rate)
sd.wait()  # Wait until the sound is finished playing

sd.play(output_signal, signal_sample_rate)
sd.wait()  # Wait until the sound is finished playing


plt.figure(figsize=(10, 6))
t = np.linspace(0, len(signal)/signal_sample_rate, int(len(signal)), endpoint=False)
plt.plot(t,signal)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Time clean Signal')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
t = np.linspace(0, len(noisy_signal)/signal_sample_rate, int(len(noisy_signal)), endpoint=False)
plt.plot(t,noisy_signal)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Time noisy Signal')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
t = np.linspace(0, len(output_signal)/signal_sample_rate, int(len(output_signal)), endpoint=False)
plt.plot(t,output_signal)
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Time Output Signal')
plt.legend()
plt.grid()
plt.show()



plt.figure(figsize=(10, 6))
freq = np.fft.fftfreq(len(np.fft.fft(signal)), 1/signal_sample_rate)
t = np.linspace(0, len(np.fft.fft(signal))/signal_sample_rate, int(len(np.fft.fft(signal))), endpoint=False)
plt.plot(freq,np.fft.fft(signal))
plt.xlabel('frequency')
plt.ylabel('Magnitude')
plt.title('frequency clean np.fft.fft(signal)')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
freq = np.fft.fftfreq(len(np.fft.fft(noisy_signal)), 1/signal_sample_rate)
t = np.linspace(0, len(np.fft.fft(noisy_signal))/signal_sample_rate, int(len(np.fft.fft(noisy_signal))), endpoint=False)
plt.plot(freq,np.fft.fft(noisy_signal))
plt.xlabel('frequency')
plt.ylabel('Magnitude')
plt.title('frequency noisy Signal')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
freq = np.fft.fftfreq(len(np.fft.fft(output_signal)), 1/signal_sample_rate)
t = np.linspace(0, len(np.fft.fft(output_signal))/signal_sample_rate, int(len(np.fft.fft(output_signal))), endpoint=False)
plt.plot(freq,np.fft.fft(output_signal))
plt.xlabel('frequency')
plt.ylabel('Magnitude')
plt.title('frequency Output Signal')
plt.legend()
plt.grid()
plt.show()

