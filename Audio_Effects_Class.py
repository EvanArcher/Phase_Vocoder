import numpy as np
import scipy.signal 

class AudioEffects:
    def __init__(self):
        None    
    def normal_signal(self,signal):
        return signal
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
        self.delay_buffer = CircularBuffer(buffer_len_ms = delay_ms * 2, sample_rate=samplerate, signal_size=0)
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
        
        # Initialize delay buffer if not done already (lazy initialization of signal size)
        if self.delay_buffer.signal_len == 0:
            self.delay_buffer.signal_len = signal_len
        
        # Retrieve the current buffer state (delayed signal)
        delayed_signal = self.delay_buffer.get_buffer()

        # Calculate delayed indices for reading
        delayed_buffer_pos = (self.delay_buffer.pos - self.delay_buffer.buffer_size // 2) % self.delay_buffer.buffer_size
        delayed_indices = np.arange(delayed_buffer_pos, delayed_buffer_pos + signal_len) % self.delay_buffer.buffer_size
        
        # Apply delay effect
        output_signal = signal + self.delay_gain * delayed_signal[delayed_indices]
        
        # Update buffer with the current signal
        self.delay_buffer.update_buffer(output_signal)

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






