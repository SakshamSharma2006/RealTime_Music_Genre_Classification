import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler
import os

# ==============================
# Paths
# ==============================
MODEL_PATH = r"F:\RealTime_Music_Genre_Classification\model\cnn_genre_model.h5"
SCALER_MEAN_PATH = r"F:\RealTime_Music_Genre_Classification\model\scaler.npy"
SCALER_SCALE_PATH = r"F:\RealTime_Music_Genre_Classification\model\scaler_scale.npy"

TEMP_AUDIO_FILE = "temp_audio.wav"
LIVE_AUDIO_FILE = "live_audio.wav"

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# ==============================
# Load Model & Scaler
# ==============================
@st.cache_resource
def load_model_and_scaler():
    model = load_model(MODEL_PATH)
    scaler = StandardScaler()
    scaler.mean_ = np.load(SCALER_MEAN_PATH)
    scaler.scale_ = np.load(SCALER_SCALE_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

# ==============================
# Feature Extraction
# ==============================
def extract_mfcc(audio_path):
    try:
        audio, sr = librosa.load(audio_path, duration=30)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        return mfcc
    except Exception as e:
        st.error(f"Error extracting MFCC: {e}")
        return None

# ==============================
# Live Audio Recording
# ==============================
def record_audio(duration=5, sr=22050):
    st.info("🎙️ Recording... Please play music now")
    try:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
        sd.wait()
        write(LIVE_AUDIO_FILE, sr, (audio * 32767).astype(np.int16))
        st.success("✅ Recording completed!")
        return LIVE_AUDIO_FILE
    except Exception as e:
        st.error(f"Recording error: {e}")
        return None

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(
    page_title="Real-Time Music Genre Classification",
    layout="centered"
)

st.title("🎧 Real-Time Music Genre Classification")
st.write(
    "This web application classifies music genres in real time using "
    "audio processing (MFCC) and a CNN-based deep learning model."
)

mode = st.radio(
    "Select Input Mode:",
    ["Upload Audio File", "Live Microphone Recording"]
)

audio_path = None

# ==============================
# Upload Audio Mode
# ==============================
if mode == "Upload Audio File":
    uploaded_file = st.file_uploader(
        "Upload an audio file (.wav or .mp3)",
        type=["wav", "mp3"]
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)
        with open(TEMP_AUDIO_FILE, "wb") as f:
            f.write(uploaded_file.read())
        audio_path = TEMP_AUDIO_FILE

# ==============================
# Live Microphone Mode
# ==============================
elif mode == "Live Microphone Recording":
    if st.button("🎙️ Record 5 Seconds"):
        audio_path = record_audio()

# ==============================
# Prediction Section
# ==============================
if audio_path is not None and os.path.exists(audio_path):
    st.write("🔍 Extracting MFCC features...")
    mfcc = extract_mfcc(audio_path)

    if mfcc is not None:
        mfcc_scaled = scaler.transform([mfcc])
        mfcc_scaled = mfcc_scaled[..., np.newaxis]

        st.write("🧠 Predicting genre...")
        prediction = model.predict(mfcc_scaled, verbose=0)[0]

        genre_index = np.argmax(prediction)
        confidence = prediction[genre_index] * 100

        st.success(
            f"🎼 Predicted Genre: **{GENRES[genre_index].upper()}**"
        )
        st.info(f"Confidence: **{confidence:.2f}%**")

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(GENRES, prediction, color='steelblue')
        ax.set_ylabel("Probability")
        ax.set_title("Genre Prediction Confidence")
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
# streamlit run app.py