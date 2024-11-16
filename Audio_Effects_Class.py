import numpy as np
import scipy.signal 

class AudioEffects:
    def __init__(self):
        None    
    def basic_delay_init(self,delay_ms=250, samplerate=44100, delay_gain=0.5):
        """
        Initialize the audio effect class with a delay buffer.
        
        Args:
        delay_ms (float): The delay time in milliseconds.
        samplerate (int): The sample rate of the audio signal (e.g., 44100 Hz).
        feedback (float): The feedback amount (default is 0.5 for moderate feedback).
        """
        self.samplerate = samplerate
        self.delay_ms = delay_ms
        self.delay_gain = delay_gain
        
        # Convert delay time from milliseconds to samples
        self.delay_samples = int((delay_ms / 1000) * samplerate)
        
        # Initialize a circular buffer for delay (2x delay size for wrap-around)
        self.delay_buffer = np.zeros(self.delay_samples * 2)
        
        # Track the position in the circular buffer
        self.buffer_pos = 0
    def basic_delay(self, signal):
        """
        Apply a simple delay effect to the audio signal with feedback.

        Args:
        signal (numpy array): The input audio signal (current block).
        delay_ms (float): The delay time in milliseconds.
        samplerate (int): The sample rate of the audio (used to calculate delay).
        feedback (float): The feedback amount (default is 0.5 for moderate feedback).

        Returns:
        numpy array: The audio signal with delay applied.
        """
        # Ensure signal is 1D (flatten if it's a 2D array with shape (512, 1))
        signal = signal.flatten()

        signal_len = len(signal)
        
        # Output signal to hold the processed result
        output_signal = np.zeros_like(signal)

        # Calculate the delayed buffer position (N samples before the current position)
        delayed_buffer_pos = (self.buffer_pos - self.delay_samples) % len(self.delay_buffer)
        
        # Calculate the range of indices for the delayed samples
        delayed_indices = np.arange(delayed_buffer_pos, delayed_buffer_pos + signal_len) % len(self.delay_buffer)
        
        # Apply the delay using the formula: y[n] = x[n] + gain * x[n - N]
        output_signal[:] = signal[:] + self.delay_gain * self.delay_buffer[delayed_indices]
        
        # Handle circular buffer wrap-around for writing new data
        end_pos = self.buffer_pos + signal_len
        if end_pos <= len(self.delay_buffer):
            # No wrap-around needed, can write in one go
            self.delay_buffer[self.buffer_pos:end_pos] = signal[:]
        else:
            # Wrap-around needed, split the writing process
            first_part_len = len(self.delay_buffer) - self.buffer_pos
            self.delay_buffer[self.buffer_pos:] = signal[:first_part_len]  # Write to the end of the buffer
            self.delay_buffer[:end_pos % len(self.delay_buffer)] = signal[first_part_len:]  # Write the rest at the start

        # Update buffer position and handle wrap-around
        self.buffer_pos = (self.buffer_pos + signal_len) % len(self.delay_buffer)

        return output_signal

    

def circular_buffer(self,signal, sample_rate, buffer_len = 2000):
    """circular buffer creates a circular buffer with a defualt length of 2 seconds of
    delay, but can be overrided, in ms
    """
    self.buffer_size = int((buffer_len/1000) * sample_rate)
    self.buffer = np.zeros(self.buffer_size)
    self.pos = 0
    signal_len = len(signal)
    def update_buffer(signal):
        """Used within buffer to update its value
        and updates its position"""
        if self.pos+signal_len > self.buffer_size:
            #we need to wrap
            self.buffer[self.pos:-1] = signal[0:self.buffer_size-self.pos]
            self.buffer[0:signal_len-self.buffer_size+self.pos] = signal[self.buffer_size-self.pos:-1]
        else:
            self.buffer[self.pos:self.pos+signal_len] = signal
        
        self.pos = (self.pos + signal_len) % self.buffer_size
    
    update_buffer(signal)
    
    return self.buffer

class CircularBuffer:
    def __init__(self, buffer_len_ms, sample_rate, signal_size):
        """
        Initialize a circular buffer for audio processing.

        Args:
        buffer_len_ms (float): The length of the buffer in milliseconds.
        sample_rate (int): The sample rate of the audio signal.
        signal_size (int): how long incoming signal is, assumes same size for
        every use
        """
        self.buffer_size = int((buffer_len_ms / 1000) * sample_rate)
        self.buffer = np.zeros(self.buffer_size)
        self.pos = 0
        self.signal_len = signal_size
    def update_buffer(self, signal):
        """
        Update the circular buffer with new incoming audio signal.

        Args:
        signal (numpy array): The input audio signal (current block).
        """
        # Handle circular buffer wrap-around
        if self.pos + self.signal_len > self.buffer_size:
            end_len = self.buffer_size - self.pos
            self.buffer[self.pos:] = signal[:end_len]
            self.buffer[:self.signal_len - end_len] = signal[end_len:]
        else:
            self.buffer[self.pos:self.pos + self.signal_len] = signal
        
        self.pos = (self.pos + self.signal_len) % self.buffer_size

    def get_buffer(self):
        """Retrieve the current buffer state."""
        return self.buffer






