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
    def basic_delay_with_feedback(self, signal):
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
    
    def basic_delay(self, signal):
        """
        Apply a simple delay effect to the audio signal with NO feedback.

        Args:
        signal (numpy array): The input audio signal (current block).
        delay_ms (float): The delay time in milliseconds.
        samplerate (int): The sample rate of the audio (used to calculate delay).

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
        
        # After signal played need to erase section in buffer since it only plays once
        self.delay_buffer.clear_buffer(delayed_indices)
        # Update buffer with the current signal ONLY
        self.delay_buffer.update_buffer(signal)

        return output_signal
    
    def basic_noise_filter_init(self,signal_len, NSR = 0.1):
        """
        Initialize the filter with an NSR value and finding noise based on that
        
        Args:
        NSR (Ratio): noise to signal ratio, default is 1:10 
        """
        
        self.NSR = NSR
        self.signal_len_noise_filter = signal_len
        self.signal_hamm_window = np.hamming(self.signal_len_noise_filter)
    
    def basic_noise_filter(self, signal, ):
        """
        Apply a noise filter using PSD and fft, filters out untouched frequencies,
        however we can leverage this with just reducing the magnitude of the fft
        to remove noise through the whole thing

        Args:
        signal (numpy array): The input audio signal (current block).
        filter strength

        Returns:
        numpy array: The reduced noise, signal.
        """
        
        # Ensure signal is 1D (flatten if it's a 2D array with shape (512, 1))
        signal = signal.flatten()
        
        signal_std = np.std(signal) #signal standard devitation
        desired_noise_ratio  = self.NSR #NSR lol = noise/signal
            
        noise = np.random.normal(0,1,signal.shape)
        scaled_noise = noise * (signal_std * desired_noise_ratio) #always new nosie signal based on devations
        # of real current signal

        #______________________________________________________________
        fft_noisy = np.fft.fft(scaled_noise) # reference for power of noisy signal
        PSD = np.abs(fft_noisy * np.conj(fft_noisy) / self.signal_len_noise_filter) # <- Power spectrum density per frame, why we divide by N
        max_noise_PSD = max(PSD)
        
        # apply window for cleaner fft
        windowed_signal = self.signal_hamm_window * signal
        fft_windowed_signal = np.fft.fft(windowed_signal)
        
        #5d. Now we are gonna use the PSD to find our signal in the noise
        signal_PSD = np.abs(fft_windowed_signal * np.conj(fft_windowed_signal)) / self.signal_len_noise_filter
        indicies = signal_PSD > max_noise_PSD #grab indicies greater than noise, should be signal
        #will filter out untouched frequencies
        filtered_signal = fft_windowed_signal * indicies

        #6. apply IFFT to noisy_signal
        ifft_denoised_signal = np.fft.ifft(filtered_signal)

        # Now return the cleaned up signal 
        return ifft_denoised_signal





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
    def clear_buffer(self, indices):
        """
        Clear the buffer in a given set of indicies, sets it back to zero
        
        Args:
        sindicies (nunpy arrange): this tells which indicies to clear and set to zero
        """
        self.buffer[indices] = 0
        

    def get_buffer(self):
        """Retrieve the current buffer state."""
        return self.buffer






