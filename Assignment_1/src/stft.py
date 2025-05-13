import os
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

class SpectrogramGenerator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.signal, self.sr = self.load_audio()

    def load_audio(self):
        signal, sr = librosa.load(self.file_path, sr=None)  
        return signal, sr

    def plot_all_spectrogram(self, window_type: str, ax):
        # Compute the Short-Time Fourier Transform (STFT)
        S = np.abs(librosa.stft(self.signal, window=window_type))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=self.sr, x_axis="time", y_axis="log", ax=ax)
        ax.set_title(f"Window: {window_type}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.label_outer()  

    def plot_spectrogram(self, save_path=None, window_type="hann"):
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure
        S = np.abs(librosa.stft(self.signal, window=window_type))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=self.sr, x_axis="time", y_axis="log", ax=ax)
        
        ax.set_title(f"Spectrogram (Window: {window_type})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")  # Save image with high resolution
            print(f"Spectrogram saved at: {save_path}")

        plt.close(fig)


if __name__ == "__main__":
    file_path = "data/Project_108.mp3" 

    output_dir = "spectrograms"
    os.makedirs(output_dir, exist_ok=True)

    spectrogram = SpectrogramGenerator(file_path)

    windows = ["boxcar", "hamming", "hann"]
    songs = ["data/Eminem - Stan (Short Version) ft. Dido.mp3",
             "data/Hardwell - Oldskool Sound.mp3",
             "data/Harry Styles - As It Was (Official Video).mp3",
             "data/Swedish House Mafia ft. John Martin - Dont You Worry Child (Official Video).mp3"
             ]
    
    # for song in songs:
    #     song_spectrogram = SpectrogramGenerator(song)
    #     base_name = os.path.splitext(os.path.basename(song))[0]  # Remove path & extension
    #     output_path = os.path.join(output_dir, f"{base_name}.png")
    #     song_spectrogram.plot_spectrogram(save_path=output_path, window_type="hann")
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))  
    
    for i, window in enumerate(windows):
        spectrogram.plot_all_spectrogram(window_type=window, ax=axs[i])
    
    plt.savefig("spectrograms/project_108_all_windows_comparision.png", bbox_inches="tight")  # Save image with high resolution
    print(f"Spectrogram saved at: spectrograms/all_windows_comparision.png")


    plt.tight_layout()
    plt.show()

    