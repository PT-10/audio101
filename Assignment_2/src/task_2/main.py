import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import random

def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None, )
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc, sr

def plot_mfcc(mfcc, sr, title="MFCC Spectrogram"):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def save_mfcc(mfcc, sr, title, save_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_statistics(mfcc):
    mean = np.mean(mfcc, axis=1)
    var = np.var(mfcc, axis=1)
    return mean, var

def analyze_audio_files(audio_files):
    for path in audio_files:
        print(f"\nProcessing: {path}")
        mfcc, sr = extract_mfcc(path)
        
        # Visualization
        plot_mfcc(mfcc, sr, title=f"MFCC Spectrogram - {os.path.basename(path)}")
        
        # Statistical Analysis
        mean, var = compute_statistics(mfcc)
        print("Mean of MFCC coefficients:\n", mean)
        print("Variance of MFCC coefficients:\n", var)

if __name__ == '__main__':
    # Define paths
    data_dir = "data"
    features_dir = os.path.join(data_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    # List of language folders
    language_folders = [
        "Bengali","Gujarati","Hindi","Kannada","Malayalam","Marathi","Punjabi", "Tamil", "Telugu", "Urdu"
    ]

    # Base directory for language data
    language_data_dir = "data/Language Detection Dataset"

    # # Process each language folder
    # for language in tqdm(language_folders, desc="Processing Languages"):
    #     lang_folder = os.path.join(language_data_dir, language)
    #     csv_path = os.path.join(features_dir, f"{language}_mfcc.csv")
        
    #     # Collect MFCC features for all audio files in the language folder
    #     audio_files = os.listdir(lang_folder)
    #     mfcc_features = []
        
    #     for audio_file in tqdm(audio_files, desc=f"Processing {language}", leave=False):
    #         if audio_file.endswith(".mp3"):
    #             audio_path = os.path.join(lang_folder, audio_file)
    #             try:
    #                 # print(f"Processing: {audio_path}")
    #                 mfcc, _ = extract_mfcc(audio_path)
    #                 mean, var = compute_statistics(mfcc)
    #                 mfcc_features.append({
    #                     "file": os.path.basename(audio_file),
    #                     "mean": mean.tolist(),
    #                     "variance": var.tolist()
    #                 })
    #             except Exception as e:
    #                 print(f"Error processing {audio_path}: {e}")
        
    #     # Save features to CSV
    #     df = pd.DataFrame(mfcc_features)
    #     df.to_csv(csv_path, index=False)
    #     print(f"Saved MFCC features for {language} to {csv_path}")

    spectrograms_dir = os.path.join(data_dir, "output/spectrograms")
    os.makedirs(spectrograms_dir, exist_ok=True)

    selected_languages = ["Hindi"]

    for language in selected_languages:
        lang_folder = os.path.join(language_data_dir, language)
        audio_files = [f for f in os.listdir(lang_folder) if f.endswith(".mp3")]
        selected_audio = random.sample(audio_files, min(6, len(audio_files)))

        for audio_file in selected_audio:
            audio_path = os.path.join(lang_folder, audio_file)
            try:
                mfcc, sr = extract_mfcc(audio_path)
                title = f"{language} - {audio_file}"
                save_path = os.path.join(spectrograms_dir, f"{language}_{os.path.splitext(audio_file)[0]}.png")
                save_mfcc(mfcc, sr, title, save_path)
                print(f"Saved spectrogram: {save_path}")
            except Exception as e:
                print(f"Failed to process {audio_path}: {e}")