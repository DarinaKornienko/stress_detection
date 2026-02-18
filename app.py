import streamlit as st
from audiorecorder import audiorecorder
from pydub import AudioSegment
import io
import os
from pathlib import Path
from app import utils
import numpy as np

import os
import platform


def configure_ffmpeg():
# Check if we are running on Windows 
    if platform.system() == "Windows":
# Look for the .exe files we put in the project folder
        ffmpeg_exe = os.path.abspath("ffmpeg.exe")
        ffprobe_exe = os.path.abspath("ffprobe.exe")
        
        if os.path.exists(ffmpeg_exe):
            AudioSegment.converter = ffmpeg_exe
            AudioSegment.ffprobe = ffprobe_exe
    else:
        # For non-Windows, we assume ffmpeg is in the PATH
        pass

configure_ffmpeg()
# conversion function to standardize all audio inputs to 16kHz Mono WAV
def process_to_standard_wav(audio_input):
    """
    Converts any input (bytes or file) to 16kHz, Mono, WAV.
    This ensures the models get exactly what they were trained on.
    """
# Load audio from bytes
    audio = AudioSegment.from_file(io.BytesIO(audio_input))
# Convert to Mono and 16000Hz
    audio = audio.set_frame_rate(16000).set_channels(1)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()

# Streamlit App
st.set_page_config(page_title="Ensemble Stress AI", page_icon="🧠")

st.title("Universal Stress Detector")
st.markdown("---")

# Input Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎤 Record")
    recorded_audio = audiorecorder("Start Recording", "Stop & Save")

with col2:
    st.subheader("📁 Upload")
    uploaded_file = st.file_uploader("Upload any format", type=["wav", "mp3", "m4a", "ogg", "flac"])


final_wav_bytes = None

if len(recorded_audio) > 0:
# Handle recording
    st.info("Using recorded audio...")
    final_wav_bytes = process_to_standard_wav(recorded_audio.export().read())
elif uploaded_file is not None:

# Handle upload and conversion
    st.info(f"Processing {uploaded_file.name}...")
    final_wav_bytes = process_to_standard_wav(uploaded_file.read())

# Display audio player and analysis button if we have valid audio
if final_wav_bytes:
    st.audio(final_wav_bytes, format="audio/wav")
    
    if st.button("-Run Deep Analysis-", use_container_width=True):
        temp_file = Path("temp_converted.wav")
        temp_file.write_bytes(final_wav_bytes)
        
        with st.spinner("Calculating Ensemble Probability..."):
            try:
# Get scores
                s1 = utils.predict_cnn_lstm(str(temp_file))
                s2 = utils.predict_panns(str(temp_file))
                s3 = utils.predict_yamnet(str(temp_file))
                
                avg_score = (s1 + s2 + s3) / 3

# Visual output
                st.markdown("### Analysis Results")
                
# Dynamic color selection
                if avg_score < 0.35:
                    color, label = "#28a745", "CALM / STABLE"
                elif avg_score < 0.70:
                    color, label = "#fd7e14", "MODERATE TENSION"
                else:
                    color, label = "#dc3545", "HIGH STRESS"

                st.markdown(f"""
                    <div style="background-color: #1e1e1e; padding: 25px; border-radius: 15px; border-left: 10px solid {color};">
                        <h4 style="color: white; margin: 0;">Overall Stress Probability</h4>
                        <h1 style="color: {color}; font-size: 70px; margin: 10px 0;">{avg_score*100:.1f}%</h1>
                        <p style="color: #888; font-weight: bold;">{label}</p>
                    </div>
                """, unsafe_allow_html=True)

# Detailed breakdown
                with st.expander("See Model Breakdown"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("CNN-LSTM", f"{s1*100:.1f}%")
                    c2.metric("PANNs", f"{s2*100:.1f}%")
                    c3.metric("YAMNet", f"{s3*100:.1f}%")

            except Exception as e:
                st.error(f"Error analyzing audio: {e}")
            finally:
                if temp_file.exists():
                    temp_file.unlink()