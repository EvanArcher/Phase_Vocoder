#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:23:15 2023

@author: evana

This code is meant to implement the overlap and add method in python reading in a sample audio file then playing
No processing is done in this code rather it be used as an outline
"""


# 1. Read in signal 


# all of this is in a loop
# 2. loop over signal in N size chunks/samples

# 3. apply window to signal

# 4. apply fft to signal of size M>=N, we will use size N

# 5. this is where processing would go with the fft output

#6. apply IFFT to signal

# 7. Move signal to output buffer, we will map it to a array the length of our 
# input signal. 

# 8. Now move by Hop size samples and repeat process, Hop size is N/2 in our case