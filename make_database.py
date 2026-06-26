import os
import librosa
import numpy as np
from scipy.ndimage import maximum_filter
from collections import defaultdict
import pickle

# --- Configuration ---
FOLDER_PATH = 'song_database'
PICKLE_FILE = 'song_database.pkl'

# --- 1. Core Functions (Same as your app) ---
def get_peaks(y_signal, sr, hop_length=512, n_fft=2048):
    D_stft = librosa.stft(y_signal, n_fft=n_fft, hop_length=hop_length)
    S_db_stft = librosa.amplitude_to_db(np.abs(D_stft), ref=np.max)
    local_max = maximum_filter(S_db_stft, size=20) == S_db_stft
    peaks_mask = local_max & (S_db_stft > np.percentile(S_db_stft, 95))
    freq_bins, time_frames = np.where(peaks_mask)
    return sorted(list(zip(time_frames, freq_bins)), key=lambda x: x[0]), S_db_stft

def generate_hashes(peaks, fan_value=15):
    hashes = []
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if i + j < len(peaks):
                t1, f1 = peaks[i]
                t2, f2 = peaks[i + j]
                delta_t = t2 - t1
                if 0 < delta_t < 200:
                    hashes.append(((int(f1), int(f2), int(delta_t)), t1))
    return hashes

# --- 2. Build and Save the Database ---
def main():
    print(f"Looking for songs in '{FOLDER_PATH}'...")
    if not os.path.exists(FOLDER_PATH):
        print(f"Error: Could not find the folder '{FOLDER_PATH}'.")
        return

    song_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.mp3')]
    if not song_files:
        print(f"No MP3 files found in '{FOLDER_PATH}'.")
        return

    print(f"Found {len(song_files)} songs. Starting indexing...")
    
    database = defaultdict(list)
    master_sr = None
    
    for i, song_file in enumerate(song_files):
        print(f"[{i+1}/{len(song_files)}] Processing: {song_file}")
        song_path = os.path.join(FOLDER_PATH, song_file)
        
        # Load audio
        y_song, sr = librosa.load(song_path, sr=None)
        if master_sr is None:
            master_sr = sr
            
        # Get fingerprints
        song_peaks, _ = get_peaks(y_song, sr)
        song_hashes = generate_hashes(song_peaks)
        
        # Add to dictionary
        for h, t1 in song_hashes:
            database[h].append((song_file, t1))

    # Save to .pkl file
    print("\nSaving database to pickle file...")
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump((database, master_sr), f)

    print(f"Success! Your database is now saved as '{PICKLE_FILE}'.")

if __name__ == "__main__":
    main()