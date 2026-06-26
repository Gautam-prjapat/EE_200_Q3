import streamlit as st
import os
import tempfile
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from collections import defaultdict, Counter
import pandas as pd
import pickle

# ------------------------------------------------------------------------
# 1. Page Configuration & Constants
# ------------------------------------------------------------------------
st.set_page_config(page_title="Audio Fingerprinter", layout="wide")

# ------------------------------------------------------------------------
# 2. Core Audio Processing Functions
# ------------------------------------------------------------------------
def load_uploaded_audio(uploaded_file, target_sr=None):
    """
    Temporarily saves a Streamlit ByteStream upload to the physical hard drive 
    so librosa's underlying C-library can read it safely, then deletes it.
    """
    # Extract extension (e.g., '.mp3' or '.wav') safely
    _, ext = os.path.splitext(uploaded_file.name)
    if not ext:
        ext = ".wav" # Fallback if extension is missing

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
        
    try:
        y, sr = librosa.load(tmp_path, sr=target_sr)
    finally:
        os.remove(tmp_path) # Always clean up the temp file
        
    return y, sr

@st.cache_data(show_spinner=False)
def get_peaks(y_signal, sr, hop_length=512, n_fft=2048):
    """Extracts the strongest frequencies (constellation points) from the audio."""
    D_stft = librosa.stft(y_signal, n_fft=n_fft, hop_length=hop_length)
    S_db_stft = librosa.amplitude_to_db(np.abs(D_stft), ref=np.max)
    
    # Find local maxima using a 20x20 scanning window
    local_max = maximum_filter(S_db_stft, size=20) == S_db_stft
    
    # Only keep peaks that are in the top 5% of loudness
    peaks_mask = local_max & (S_db_stft > np.percentile(S_db_stft, 95))
    
    freq_bins, time_frames = np.where(peaks_mask)
    peak_points = sorted(list(zip(time_frames, freq_bins)), key=lambda x: x[0])
    return peak_points, S_db_stft

@st.cache_data(show_spinner=False)
def generate_hashes(peaks, fan_value=15):
    """Links single peaks into robust 3-tuple pairs (Freq1, Freq2, TimeDelta)."""
    hashes = []
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if i + j < len(peaks):
                t1, f1 = peaks[i]
                t2, f2 = peaks[i + j]
                delta_t = t2 - t1
                
                # Only pair peaks that happen relatively close to each other
                if 0 < delta_t < 200:
                    hashes.append(((int(f1), int(f2), int(delta_t)), t1))
    return hashes

@st.cache_resource(show_spinner=False)
def build_database():
    """Loads the pre-computed database instantly from the pickle file."""
    with open('song_database.pkl', 'rb') as f:
        database, master_sr = pickle.load(f)
    return database, master_sr

def predict_song(query_hashes, database):
    """Calculates offsets and finds the highest cluster of aligned hashes."""
    matches_per_song = defaultdict(list)
    for h, t_query in query_hashes:
        if h in database:
            for song_name, t_db in database[h]:
                offset = t_db - t_query
                matches_per_song[song_name].append(offset)
                
    best_song = None
    max_matches = 0
    best_offsets = []
    
    for song, offsets in matches_per_song.items():
        if offsets:
            most_common_offset, count = Counter(offsets).most_common(1)[0]
            if count > max_matches:
                max_matches = count
                best_song = song
                best_offsets = offsets
                
    return best_song, max_matches, best_offsets

# ------------------------------------------------------------------------
# 3. Streamlit User Interface
# ------------------------------------------------------------------------
st.title("Sonic Signatures: Audio Fingerprinter")

with st.spinner("Loading database into memory..."):
    if not os.path.exists('song_database.pkl'):
        st.error("Error: 'song_database.pkl' not found! Please run your script to generate it first.")
        st.stop()
    database, master_sr = build_database()

# Create App Tabs
tab1, tab2 = st.tabs(["Single-Clip Mode", "Batch Mode"])

# --- TAB 1: Single-Clip Visualizer ---
with tab1:
    st.header("Single-Clip Mode")
    st.markdown("Upload a noisy recording to see how the matching engine decides the winner.")
    
    uploaded_file = st.file_uploader("Upload a query clip (MP3/WAV)", type=["mp3", "wav"], key="single")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing clip..."):
            # Use the tempfile workaround to safely read the uploaded ByteStream
            y_query, sr = load_uploaded_audio(uploaded_file, target_sr=master_sr)
            
            query_peaks, S_db = get_peaks(y_query, sr)
            query_hashes = generate_hashes(query_peaks)
            
            best_song, score, target_offsets = predict_song(query_hashes, database)
            
            if best_song:
                st.success(f"**Identified Song:** {best_song} (Score: {score} aligned hashes)")
                
                st.subheader("Intermediate Visualizations")
                fig, ax = plt.subplots(1, 3, figsize=(18, 5))
                
                # Plot 1: Spectrogram
                librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz', ax=ax[0])
                ax[0].set_title('Spectrogram')
                ax[0].set_ylim([0, min(sr/2, 10000)])
                
                # Plot 2: Constellation Map
                times = librosa.frames_to_time([p[0] for p in query_peaks], sr=sr, hop_length=512)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)[[p[1] for p in query_peaks]]
                ax[1].scatter(times, freqs, c='black', s=5)
                ax[1].set_title('Constellation of Peaks')
                ax[1].set_xlabel('Time (s)')
                ax[1].set_ylabel('Frequency (Hz)')
                ax[1].set_ylim([0, min(sr/2, 10000)])
                ax[1].grid(True, alpha=0.3)
                
                # Plot 3: Offset Histogram
                ax[2].hist(target_offsets, bins=100, color='royalblue')
                ax[2].set_title('Offset Histogram (Matched Song)')
                ax[2].set_xlabel('Time Offset (frames)')
                ax[2].set_ylabel('Frequency of Match')
                
                st.pyplot(fig)
            else:
                st.error("No match found in the database. The audio might be too noisy or short.")

# --- TAB 2: Batch Mode CSV Generator ---
with tab2:
    st.header("Batch Mode")
    st.markdown("Upload multiple query clips at once to generate an automated evaluation CSV.")
    
    uploaded_files = st.file_uploader("Upload multiple query clips", type=["mp3", "wav"], accept_multiple_files=True, key="batch")
    
    if uploaded_files:
        if st.button("Run Batch Processing"):
            results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                # Use the tempfile workaround
                y_query, sr = load_uploaded_audio(file, target_sr=master_sr)
                
                query_peaks, _ = get_peaks(y_query, sr)
                query_hashes = generate_hashes(query_peaks)
                
                best_song, _, _ = predict_song(query_hashes, database)
                
                # Format exactly as requested: remove the .mp3 from the prediction
                filename = file.name
                prediction = os.path.splitext(best_song)[0] if best_song else "No Match"
                
                results.append({"filename": filename, "prediction": prediction})
                
                # Update visual progress bar
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            # Create Pandas DataFrame and Display it
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)
            
            # Generate Native CSV Download Button
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download results.csv",
                data=csv,
                file_name='results.csv',
                mime='text/csv',
            )