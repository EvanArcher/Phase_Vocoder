#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:14:45 2024

@author: evana

#Pitch Detection algorithim
"""


import time
import numpy as np
import math
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd



RATE = 64000
CHUNK = 1024	# 1024
signal, signal_sample_rate = sf.read('My_vocals.wav')

all_pitches = []

stream_buffer = []



def empty_frame(length):
	"""Returns an empty 16bit audio frame of supplied length."""
	frame = np.zeros(length, dtype='i2')
	return to_raw_data(frame)


def to_int_data(raw_data):
	"""Converts raw bytes data to ."""
	return np.array([int.from_bytes(raw_data[i:i+2], byteorder='little', signed=True) for i in range(0, len(raw_data), 2)])

def to_raw_data(int_data):
	data = int_data.clip(-32678, 32677)
	data = data.astype(np.dtype('i2'))
	return b''.join(data.astype(np.dtype('V2')))

def process(raw_data):
	st = time.time()
	data = to_int_data(raw_data)
	data = data*4		# raise volume
	detect_pitch(data)
	et = time.time()
	return to_raw_data(data)

def normal_distribution(w):
	width = w+1
	weights = np.exp(-np.square([2*x/width for x in range(width)]))
	weights = np.pad(weights, (width-1,0), 'reflect')
	weights = weights/np.sum(weights)
	return weights

def detect_pitch(int_data):
	if 'avg' not in detect_pitch.__dict__:
		detect_pitch.avg = 0
	WIND = 10
	CYCLE = 400
	weights = normal_distribution(WIND)
	windowed_data = np.pad(int_data, WIND, 'reflect')
	smooth_data = np.convolve(int_data, weights, mode='valid')
	smooth_pitches = [0]+[np.mean(smooth_data[:-delay] - smooth_data[delay:]) for delay in range(1,CYCLE)]

	dips = [x for x in range(WIND, CYCLE-WIND) if smooth_pitches[x] == np.min(smooth_pitches[x-WIND:x+WIND])]
	if len(dips) > 1:
		av_dip = np.mean(np.ediff1d(dips))
		cheq_freq = signal_sample_rate / av_dip
		detect_pitch.avg = detect_pitch.avg*0.5 + cheq_freq*0.5
		all_pitches.append(int(detect_pitch.avg))
		print('\r'+str(int(detect_pitch.avg))+' Hz      ', end='')


if __name__ == "__main__":
    #chunk size and hop size=
    N = 1024
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
        fft_signal = np.fft.fft(windowed_signal)

    # 5. this is where processing would go with the fft output

    #6. apply IFFT to signal
        ifft_signal = np.fft.ifft(fft_signal)
        process(ifft_signal)
    # 7. Move signal to output buffer, we will map it to a array the length of our 
    # input signal. 
        try:
            output_signal[i:i+N] = ifft_signal + output_signal[i:i+N]
        except:
            None
    # 8. Now move by Hop size samples and repeat process, Hop size is N/2 in our case

    #play sound to see how it is
    # sd.play(signal, signal_sample_rate)
    sd.wait()  # Wait until the sound is finished playing
    sd.play(output_signal, signal_sample_rate)
    # sd.wait()  # Wait until the sound is finished playing
	

    plt.plot(all_pitches)
    plt.show()

