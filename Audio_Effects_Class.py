import numpy as np
import math
from scipy.interpolate import interp1d
import scipy.signal 
import matplotlib.pyplot as plt
import time as timeimport
from scipy.signal.windows import gaussian,hann
import soundfile as sf
import sounddevice as sd

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
    
    def basic_noise_filter(self, signal):
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
    
    def vibrato_init(self, amplitude=0.10, frequency=5.0, sample_rate=44100, waveform='sine'):
        """
        Initilize the vibrato function, essentially being used as wrapper to 
        the LFO class where you pass in what LFO you want and that determines 
        everything else about your vibrato function
        
        Args
        amplitude (float): between 0 and 1, amplitude of wave
        frequency (float): in hz freqency of the wave
        waveform (str): what type of waveform is it. examples square, triangle, sine
        sample_rate (int): sample rate of signal
        """
        self.vibrato_amplitude = float(amplitude)
        self.vibrato_frequency = float(frequency)        
        self.vibrato_sample_rate = int(sample_rate)
        self.vibrato_waveform = str(waveform)
        
        #%%Create LFO for vibrato filter
        self.lfo = LFO(frequency=self.vibrato_frequency, amplitude=self.vibrato_amplitude, 
                  sample_rate=self.vibrato_sample_rate, waveform = self.vibrato_waveform)
        
        # Generate 1 second of samples
        duration_seconds = 1.0
        self.lfo_vibrato_num_samples = int(self.lfo.sample_rate * duration_seconds)
        self.lfo_vibrato_samples = [self.lfo.next_sample() for _ in range(self.lfo_vibrato_num_samples)]
        
        self.vibrato_buffer = CircularBuffer(buffer_len_ms = 2000, sample_rate=self.vibrato_sample_rate, signal_size=0)#2000 ms buffer
        
    def vibrato(self, signal):
        """
        Parameters
        ----------
        signal : numpy array
            Applys vibrato to the function, vibrato is done by taking the LFO results
            and then applying interpolation based on the vibrato results, so it 
            will slow and speed up the signal

        Returns
        -------
        output_signal (vibratoed signal).
        """
        
        # Ensure signal is 1D (flatten if it's a 2D array with shape (512, 1))
        signal = signal.flatten()
        signal_len = len(signal)
        
        # Initialize delay buffer if not done already (lazy initialization of signal size)
        if self.vibrato_buffer.signal_len == 0:
            self.vibrato_buffer.signal_len = signal_len
        
        # Retrieve the current buffer state (delayed signal)
        vibrato_signal = self.vibrato_buffer.get_buffer()

        
        # Interpolate signal based on computed LFO freq
        # 1 + LFO value _____ = signal length scale factor
        LFO_decimal,_ = math.modf(timeimport.time())
        LFO_index = int( self.lfo_vibrato_num_samples * LFO_decimal )
        LFO_signal_len_scale_factor = 1 + self.lfo_vibrato_samples[ LFO_index ] #gives sample value based on time
        LFO_signal_scaled_len = int( signal_len * LFO_signal_len_scale_factor ) 
        
        # Calculate delayed indices for reading
        vibrato_buffer_pos = (self.vibrato_buffer.pos - self.vibrato_buffer.buffer_size // 2) % self.vibrato_buffer.buffer_size
        vibrato_indices = np.arange(vibrato_buffer_pos, vibrato_buffer_pos + LFO_signal_scaled_len) % self.vibrato_buffer.buffer_size
        
        
        current_vibrato_signal = np.interp(np.linspace(0,signal_len,num=LFO_signal_scaled_len), 
                                  np.linspace(0,signal_len,num=signal_len), 
                                  signal) 
        
        
        preupdate_pos = self.vibrato_buffer.pos
        # Update buffer with the current signal
        self.vibrato_buffer.update_buffer(current_vibrato_signal)
        
        #Now we need to return a signal that is same length as input
        # vibrato_signal = self.vibrato_buffer.get_buffer()
        # output_signal = vibrato_signal[ self.vibrato_buffer.pos:self.vibrato_buffer.pos + signal_len]
        output_signal = self.vibrato_buffer.get_specific_buffer_points(preupdate_pos, signal_len)
        
        return output_signal
        
        
    
    def vibrato_lfo_test(self):
        # Generate 1 second of samples
        duration_seconds = 1.0
        start_time = timeimport.time()
        num_samples = int(self.lfo.sample_rate * duration_seconds)

        samples = [self.lfo.next_sample() for _ in range(num_samples)]
        time = [t / self.lfo.sample_rate for t in range(num_samples)]
        
        current_time = timeimport.time()
        decimal,_ = math.modf(current_time)
        print(current_time)
        print(decimal)
        # Plot the result
        plt.figure(figsize=(10, 4))
        plt.plot(time, samples)
        plt.title("LFO Output - 5 Hz Sine Wave")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


        # Compute FFT
        fft_result = np.fft.fft(samples)
        frequencies = np.fft.fftfreq(num_samples, d=1/self.vibrato_sample_rate)
        magnitude = np.abs(fft_result)

        # Limit to positive frequencies and up to 20 Hz
        positive_freqs = frequencies[:num_samples // 2]
        positive_magnitude = magnitude[:num_samples // 2]

        # Mask for 0–20 Hz range
        mask = (positive_freqs >= 0) & (positive_freqs <= 20)

        # Plot FFT
        plt.figure(figsize=(8, 4))
        plt.plot(positive_freqs[mask], positive_magnitude[mask])
        plt.title("FFT of LFO Output (0–20 Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        
        #Now show the sine wave we will modulate as a test its 440hz
        # Parameters
        sample_rate = 44100  # samples per second
        duration = 1.0       # seconds
        frequency = 100      # Hz (A4 note)
        
        # Generate the time axis
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Generate the sine wave
        sine_wave = np.sin(2 * np.pi * frequency * t)
        
        # Plot the first 1000 samples for clarity
        plt.figure(figsize=(10, 4))
        plt.plot(t[:], sine_wave[:])
        plt.title("1-Second 100 Hz Sine Wave")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
        
        # Compute FFT
        fft_result = np.fft.fft(sine_wave)
        frequencies = np.fft.fftfreq(len(sine_wave), d=1/self.vibrato_sample_rate)
        magnitude = np.abs(fft_result)

        # Limit to positive frequencies and up to 20 Hz
        positive_freqs = frequencies[:len(sine_wave) // 2]
        positive_magnitude = magnitude[:len(sine_wave) // 2]

        # Mask for 0–20 Hz range
        mask = (positive_freqs >= 0) & (positive_freqs <= 200)

        # Plot FFT
        plt.figure(figsize=(8, 4))
        plt.plot(positive_freqs[mask], positive_magnitude[mask])
        plt.title("FFT of Signal (0–200 Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        #Now we will make a vibrato class and call it using this 100hz signal 
        #and addign the modulation on top of it
        self.vibrato_init(amplitude=0.10, frequency=50.0, sample_rate=44100, waveform='sine')
        output_sig = self.vibrato(sine_wave)
        
        duration = len(output_sig) / sample_rate  # Calculate duration in seconds

        # Generate the time axis
        time_axis = np.linspace(0, duration, len(output_sig), endpoint=False)
        
        #Now lets plot our signal as a test
        # Plot the first 1000 samples for clarity
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis[:], output_sig[:])
        plt.title("Modulated signal vibratoed")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
        
        # Compute FFT
        fft_result = np.fft.fft(output_sig)
        frequencies = np.fft.fftfreq(len(output_sig), d=1/self.vibrato_sample_rate)
        magnitude = np.abs(fft_result)

        # Limit to positive frequencies and up to 20 Hz
        positive_freqs = frequencies[:len(output_sig) // 2]
        positive_magnitude = magnitude[:len(output_sig) // 2]

        # Mask for 0–20 Hz range
        mask = (positive_freqs >= 0) & (positive_freqs <= 200)

        # Plot FFT
        plt.figure(figsize=(8, 4))
        plt.plot(positive_freqs[mask], positive_magnitude[mask])
        plt.title("FFT of Vibrato (0–200 Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def pitch_shift_test(self):
        
        # Step 1: Original signal (440 Hz sine wave)
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        original = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 869 * t)
        
        original,sample_rate = sf.read('My_vocals.wav')
        t = np.arange(len(original)) / sample_rate
        try:
            if original.shape[1] == 2: # check if impulse is dual signal
                original = np.mean(original, axis=1)
        except:
            None
            
        # sd.play(original, sample_rate)
        
        fft_len = 2**12
        #STFT TEST
        w = hann(int(fft_len), sym=True)
        SFT = scipy.signal.ShortTimeFFT(w, hop=2**11, fs=sample_rate, mfft=fft_len, scale_to='magnitude')
        Sx = SFT.stft(original)  # perform the STFT
        
        # Axes
        freqs = np.fft.rfftfreq(SFT.mfft, d=1 / sample_rate)
        times = np.arange(Sx.shape[1]) * (SFT.hop / sample_rate)
        
        # Plot
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(
            times,
            freqs,
            np.abs(Sx),
            shading='gouraud'
        )
        plt.colorbar(label='Magnitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('STFT Magnitude Spectrogram OG Signal')
        plt.ylim(0, 2000)  # zoom in to see the tone clearly
        plt.tight_layout()
        plt.show()
        
        # Step 2: Shrink signal by 1.5× to raise pitch
        # Resample to shorter array
        shrink_factor = 1 / 1.5
        x_old = np.arange(len(original))
        x_new_shrunk = np.linspace(0, len(original) - 1, int(len(original) * shrink_factor))
        shrunk = interp1d(x_old, original, kind='linear')(x_new_shrunk)
        
        # Step 3: Stretch it back to original length using interpolation
        # NEED TO FIX
        # NEED THIS TO PRESERVE THE SAME TIME AND PITCH EFFECT
        # PITCH EFFECT WORKS SINCE IT SPEEDS IT UP, BUT WE NEED TO RAISE PITCH AND KEEP LENGTH
        x_shrunk = np.arange(len(shrunk))
        x_stretch = np.linspace(0, len(shrunk) - 1, len(original))
        stretched = interp1d(x_shrunk, shrunk, kind='linear')(x_stretch)
        
        # Step 4: Plot comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(3, 1, 1)
        plt.plot(t, original)
        plt.title("Original Signal (440 Hz)")
        plt.ylabel("Amplitude")
        
        plt.subplot(3, 1, 2)
        shrunk_time = np.linspace(0, duration, len(shrunk), endpoint=False)
        plt.plot(shrunk_time, shrunk)
        plt.title("Shrunk Signal (Shorter, Pitch Up by 1.5×)")
        plt.ylabel("Amplitude")
        
        plt.subplot(3, 1, 3)
        plt.plot(t, stretched)
        plt.title("Pitch-Shifted Signal (1.5× Higher Pitch, Original Duration)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        plt.tight_layout()
        plt.show()
        
        
        ######NOW LETS ATTEMPT to shift everything up by 2x freq 
        ####Then plot it#####
        
        STFT_shifted_matrix = self.shift_STFT_freqs(Sx, shift_factor=2)
        
        # Plot
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(
            times,
            freqs,
            np.abs(STFT_shifted_matrix),
            shading='gouraud'
        )
        plt.colorbar(label='Magnitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('STFT Magnitude Spectrogram Shifted Signal')
        plt.ylim(0, 2000)  # zoom in to see the tone clearly
        plt.tight_layout()
        plt.show()
        
        ####Now bring it to real domain and lets play/ plot it
        # 1) Inverse STFT
        shifted_signal_real = SFT.istft(STFT_shifted_matrix)
        
        # 2) Match original length (ShortTimeFFT may return slightly different length)
        N = len(original)
        if len(shifted_signal_real) > N:
            shifted_signal_real = shifted_signal_real[:N]
        else:
            shifted_signal_real = np.pad(shifted_signal_real, (0, N - len(shifted_signal_real)))
        
        # 3) Plot waveform (first 0.05s for visibility)
        t_y = np.arange(len(shifted_signal_real)) / sample_rate
        
        plt.figure(figsize=(12, 4))
        plt.plot(t_y, shifted_signal_real, linewidth=0.8)
        plt.title("Reconstructed signal from shifted STFT")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.xlim(0, max(t_y))
        plt.tight_layout()
        plt.show()
        
        #Now lets play og then new
        sd.play(original, sample_rate)
        sd.wait()
        sd.play(shifted_signal_real, sample_rate)
        sd.wait()
    def shift_STFT_freqs(self, STFT_matrix, shift_factor = 1):
        """
        

        Parameters
        ----------
        shift_factor : TYPE, float
            DESCRIPTION. The default is 1.
        STFT_matrix : matrix
            STFT_matrix[row,col]

        Returns
        -------
        STFT shifted.

        """
        STFT_shifted_matrix = np.zeros_like(STFT_matrix)
        freqs_row, time_col = STFT_matrix.shape
        #Go through each column and shift the index value up to its index shifted 
        #Unless it goes over the dimensions of the matrix
        for col in range(time_col):
            for row in range(freqs_row):
                shifted_row = int(shift_factor * row)

                if shifted_row < freqs_row:
                    STFT_shifted_matrix[shifted_row,col] += STFT_matrix[row,col]
            
        return STFT_shifted_matrix
        
        
    def pitch_shift_init(self,shift_amount = 1):
        """
        

        Parameters
        ----------
        shift_amount : TYPE, float
            DESCRIPTION. The default is 1. Give a floating number from -x,x 
            This will shift the signal, a 2 will shift the signal by 2x and 
            and -2 will shift it down by 2. Min value needs to be abs(shift_amount) >=1 


        Returns
        -------
        None.

        """
        if abs(shift_amount) < 1:
            shift_amount = 1
        self.shift_amount = shift_amount
        
        
    def pitch_shift(self,signal):
        """

        Parameters
        ----------
        shift_amount : TYPE, float
            DESCRIPTION. The default is 1. Give a floating number from -x,x 
            This will shift the signal, a 2 will shift the signal by 2x and 
            and -2 will shift it down by 2. Min value needs to be abs(shift_amount) >=1 

        Raises
        ------
        pitch
            DESCRIPTION.

        Returns
        -------
        Pitch shifted signal.

        """
        
        # Step 1: Original signal (440 Hz sine wave)
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        original = np.sin(2 * np.pi * 440 * t)
        
        # Step 2: Shrink signal by 1.5× to raise pitch
        # Resample to shorter array
        
        shrink_factor = 1 / .2
        x_old = np.arange(len(original))
        x_new_shrunk = np.linspace(0, len(original) - 1, int(len(original) * shrink_factor))
        shrunk = interp1d(x_old, original, kind='linear')(x_new_shrunk)
        
        # Step 3: Stretch it back to original length using interpolation
        x_shrunk = np.arange(len(shrunk))
        x_stretch = np.linspace(0, len(shrunk) - 1, len(original))
        stretched = interp1d(x_shrunk, shrunk, kind='linear')(x_stretch)
        
        # Step 4: Plot comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(3, 1, 1)
        plt.plot(t[:1000], original[:1000])
        plt.title("Original Signal (440 Hz)")
        plt.ylabel("Amplitude")
        
        plt.subplot(3, 1, 2)
        shrunk_time = np.linspace(0, duration, len(shrunk), endpoint=False)
        plt.plot(shrunk_time[:1000], shrunk[:1000])
        plt.title("Shrunk Signal (Shorter, Pitch Up by 1.5×)")
        plt.ylabel("Amplitude")
        
        plt.subplot(3, 1, 3)
        plt.plot(t[:1000], stretched[:1000])
        plt.title("Pitch-Shifted Signal (1.5× Higher Pitch, Original Duration)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        plt.tight_layout()
        plt.show()


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
        self.signal_len = len(signal)
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
    def get_specific_buffer_points(self, starting_index, return_signal_len):
        """
        

        Parameters
        ----------
        starting_index : INT
            This is the starting point you want to get from the buffer
            I.E. feed 42069 and you will get buffer[42069:42069+return_signal_len].
        return_signal_len : INT
            How long you want the returned buffer split to be.
            I.E if you give it 1024 you will get a signal 1024 long back from where you
            told the buffer to start looking

        Returns
        -------
        buffer_signal_slice: array/list this returns the data you requested from
        the buffer.

        """
        # Handle circular buffer wrap-around
        if starting_index + return_signal_len > self.buffer_size:
            # print(f'starting index is {starting_index} and buffer size is {self.buffer_size}')
            # print(f'return sig size = {return_signal_len}')
            first_part_of_signal = self.buffer[starting_index:self.buffer_size]
            # print(f'len of first part of sig {len(first_part_of_signal)}')
            #grab the wrapped around section
            indicies_left_to_grab = return_signal_len - (self.buffer_size - starting_index)
            # print(f'indicies left to grab is {indicies_left_to_grab}')
            second_part_of_signal= self.buffer[0:indicies_left_to_grab]
            # print(f'len of second part of sig {len(second_part_of_signal)}')
            buffer_signal_slice = np.concatenate((first_part_of_signal,second_part_of_signal))
            # print(f'len of total sig {len(buffer_signal_slice)}')
        else:
            buffer_signal_slice = self.buffer[starting_index:starting_index+return_signal_len]
        return buffer_signal_slice
    def get_buffer(self):
        """Retrieve the current buffer state."""
        return self.buffer



class LFO:
    def __init__(self, amplitude=1.0, frequency=5.0, sample_rate=44100, waveform='sine'):
        """
        Initilize a Low Frequency Oscillator
        
        Args
        amplitude (float): between 0 and 1, amplitude of wave
        frequency (float): in hz freqency of the wave
        waveform (str): what type of waveform is it
        sample_rate (int): sample rate of signal
        """
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)        
        self.sample_rate = int(sample_rate)
        self.waveform = str(waveform)
        self.phase = 0.0
        # Calculate phase increment for each sample step
        self._phase_increment = 2 * np.pi * self.frequency / self.sample_rate
    
    def __iter__(self):
        """Return the iterator (the object itself is the iterator)."""
        return self

    def __next__(self):
        """Compute the next LFO sample and update internal state."""
        # Generate waveform value based on the current phase
        if self.waveform == 'sine':
            value = math.sin(self.phase)
        elif self.waveform == 'square':
            # Square wave: output is either +1 or -1 based on sine's sign
            value = 1.0 if math.sin(self.phase) >= 0 else -1.0
        elif self.waveform == 'triangle':
            # Triangle wave: using arcsin of sine to get a linear triangle shape 
            value = (2 / math.pi) * math.asin(math.sin(self.phase))
        else:
            # Default to sine if unknown waveform
            value = math.sin(self.phase)
        # Increment phase for the next sample
        self.phase += self._phase_increment
        if self.phase >= 2 * math.pi:
            # Wrap the phase to keep it in [0, 2π)
            self.phase -= 2 * math.pi
        # Scale by amplitude and return
        return value * self.amplitude

    def reset(self):
        """Reset the oscillator phase to 0 (optional utility method)."""
        self.phase = 0.0

    # (Optional) If we want a direct method to get the next sample without using next()
    def next_sample(self):
        return self.__next__()



