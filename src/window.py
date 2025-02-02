import librosa
import numpy as np
import librosa.display
import streamlit as st
import matplotlib.pyplot as plt


class Windowing_Techniques():    
    def hann_window_function(self, N):
        n = np.arange(N)
        hann_window = 0.5 * (1 - np.cos((2 * np.pi * n) / (N - 1)))  # Hann function
        return hann_window
    
    def hamming_window_function(self, N):
        n = np.arange(N)
        hamming_window = 0.54 - 0.46 * np.cos((2 * np.pi * n) / (N - 1))  # Hamming function
        return hamming_window
    
    
class Signal():
    def __init__(self, file_path):
        self.file_path = file_path
        self.signal, self.sr = self.load_audio()
        self.windowing_techniques = Windowing_Techniques()

    def load_audio(self):
        """Load the audio file using librosa."""
        signal, sr = librosa.load(self.file_path, sr=None)  # Keep original sampling rate
        return signal, sr
    
    def get_segment(self, start_time, end_time):
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)

        signal_segment = self.signal[start_sample:end_sample]

        return signal_segment, len(signal_segment)
    
    def apply_window(self, start_time: float, end_time: float, window_fn: str = "boxcar"):
        signal_segment, N = self.get_segment(start_time, end_time)
        
        if window_fn == "boxcar":
            return signal_segment, signal_segment
        
        elif window_fn == "hann":
            hann_window = self.windowing_techniques.hann_window_function(N)
            return signal_segment * hann_window, signal_segment
        
        elif window_fn == "hamming":
            hamming_window = self.windowing_techniques.hamming_window_function(N)
            return signal_segment * hamming_window, signal_segment
        
    def check_correctness(self, original_signal, windowed_signal):
        def compute_snr(original_signal, windowed_signal):
            # Compute the power of the signal and the error (difference between original and windowed signal)
            signal_power = np.sum(original_signal**2)
            error_power = np.sum((original_signal - windowed_signal)**2)
            
            # Calculate SNR in dB
            snr = 10 * np.log10(signal_power / error_power)
            return snr

        def compute_rmse(original_signal, windowed_signal):
            # Compute RMSE (Root Mean Square Error)
            rmse = np.sqrt(np.mean((original_signal - windowed_signal)**2))
            return rmse
        
        return compute_snr(original_signal, windowed_signal), compute_rmse(original_signal, windowed_signal)



st.title("Windowing Functions")
st.write("Select a portion of the audio and apply a window function.")


uploaded_file = st.file_uploader("Upload an Audio File", type=["mp3", "wav", "flac"])

if uploaded_file:

    signal_obj = Signal(uploaded_file)
    signal, sample_rate = signal_obj.signal, signal_obj.sr
        

    audio_duration = librosa.get_duration(y=signal, sr=sample_rate)
    st.write(f"Audio duration: {audio_duration:.2f} seconds")

    start_time = st.slider("Select Start Time", 0.0, audio_duration, 5.0, 0.1)
    end_time = st.slider("Select End Time", start_time, audio_duration, 10.0, 0.1)

    window_fn = st.selectbox("Select Window Function", ["boxcar", "hann", "hamming"],)
    signal_segment_windowed, signal_segment = signal_obj.apply_window(start_time, end_time, window_fn)
    snr, rmse = signal_obj.check_correctness(signal_segment, signal_segment_windowed)

    time_segment = np.linspace(start_time, end_time, len(signal_segment_windowed))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].set_title("Full Audio Signal")
    librosa.display.waveshow(signal, sr=sample_rate, ax=axes[0], alpha=0.6)
    axes[0].axvline(x=start_time, color='red', linestyle='--', label="Start")
    axes[0].axvline(x=end_time, color='red', linestyle='--', label="End")
    axes[0].legend()
    axes[0].set_ylabel("Amplitude")

    axes[1].set_title(f"Extracted Segment ({start_time}s - {end_time}s)")
    axes[1].plot(time_segment, signal[int(start_time * sample_rate):int(end_time * sample_rate)], color='blue')
    axes[1].set_ylabel("Amplitude")

    if window_fn == "hann":
        plot_fn = signal_obj.windowing_techniques.hann_window_function(len(signal_segment_windowed))
        plot_title = "Hann Window Applied"
    elif window_fn == "hamming":
        plot_fn = signal_obj.windowing_techniques.hamming_window_function(len(signal_segment_windowed))
        plot_title = "Hamming Window Applied"
    else:  
        plot_fn = np.ones(len(signal_segment_windowed)) 
        plot_title = "Boxcar Window Applied"


    axes[2].set_title(f"Extracted Segment with {plot_title}")
    axes[2].plot(time_segment, signal_segment_windowed, color='red', alpha=0.8, label="Windowed Signal")
    axes[2].plot(time_segment, plot_fn, 'k--', label=f"{plot_title} (Scaled)", linewidth=1.5)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()

    st.pyplot(fig)

    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"Signal to Noise Ratio: {snr:.2f}")
