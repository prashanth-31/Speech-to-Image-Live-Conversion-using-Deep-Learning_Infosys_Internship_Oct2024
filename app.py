import streamlit as st
import sounddevice as sd
import torch
import numpy as np
from scipy.io.wavfile import write
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from diffusers import StableDiffusionPipeline
import librosa
from time import sleep
from transformers import pipeline

# Paths to models
whisper_model_path = "C:/Users/prash/whisper-finetuned-v2"
sd_model_id = "runwayml/stable-diffusion-v1-5"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained(whisper_model_path)
model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path).to(device)

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)  # Move Stable Diffusion model to GPU

# Initialize HuggingFace BERT sentiment analysis pipeline (using GPU if available)
sentiment_analysis = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)

# Streamlit UI setup
st.set_page_config(page_title="Speech-to-Image Generator", layout="wide")

# Custom CSS for a more sophisticated design
st.markdown(
    """
    <style>
    /* App Background and Layout */
    .stApp {
        background-color: #f4f7fb;
        font-family: "Arial", sans-serif;
    }

    /* Title Styling */
    .title-container {
        text-align: center;
        margin-top: 40px;
        margin-bottom: 40px;
    }

    .title-container h1 {
        font-size: 42px;
        color: #333;
        font-weight: 600;
        letter-spacing: 1px;
    }

    .title-container p {
        font-size: 18px;
        color: #555;
        font-weight: 400;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #5C6BC0;
        color: white;
        border-radius: 50px;
        padding: 14px 28px;
        font-size: 18px;
        border: none;
        transition: background-color 0.3s, transform 0.2s ease-in-out;
        box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.15);
    }

    .stButton > button:hover {
        background-color: #3F51B5;
        transform: scale(1.05);
    }

    /* Slider Styling */
    .stSlider > div {
        margin-top: 15px;
        width: 60%;
        margin-left: auto;
        margin-right: auto;
    }

    /* Results Card Styling */
    .result-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0px 8px 24px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        margin-bottom: 30px;
        width: 80%;
        margin-left: auto;
        margin-right: auto;
    }

    .result-card p {
        font-size: 20px;
        color: #333;
        font-weight: 600;
        text-align: center;
    }

    .result-card .text {
        font-size: 18px;
        color: #555;
        text-align: center;
        line-height: 1.8;
    }

    /* Sentiment Styling */
    .sentiment-positive {
        color: #4caf50;
    }

    .sentiment-negative {
        color: #e53935;
    }

    /* Image Styling */
    .stImage {
        border-radius: 12px;
        box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.2);
        margin-top: 20px;
        width: 70%;
        margin-left: auto;
        margin-right: auto;
    }

    /* Footer Styling */
    .footer {
        font-size: 16px;
        color: #777;
        text-align: center;
        padding-top: 40px;
        padding-bottom: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Main title and subtitle
st.markdown(
    """
    <div class="title-container">
        <h1>Speech-to-Image Generator</h1>
        <p>Record your audio, transcribe it, perform sentiment analysis, and generate an image based on the transcription.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Audio recording duration slider (centered and wider)
duration = st.slider(
    "Select recording duration (seconds)",
    min_value=1,
    max_value=30,
    value=15,
    step=1,
    help="Drag the slider to set the duration of your audio recording.",
)

# Record audio button
if st.button("Start Recording 🎙️"):
    fs = 16000

    st.write("**Recording... Please speak clearly.**")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    st.success("Recording complete!")

    # Save recorded audio to a file
    audio_path = "audio_input.wav"
    write(audio_path, fs, (audio * 32767).astype(np.int16))

    # Transcribe audio with Whisper
    st.write("**Transcribing audio...**")
    audio_input, _ = librosa.load(audio_path, sr=16000)
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    predicted_ids = model.generate(input_features)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

    # Display transcribed text with enhanced styling
    st.markdown(
        f"""
        <div class="result-card">
            <p><strong>Transcription:</strong></p>
            <p class="text">{transcription}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("📊 **Analyzing sentiment...**")
    sentiment = sentiment_analysis(transcription)
    sentiment_label = sentiment[0]["label"]
    sentiment_score = sentiment[0]["score"]

    # Display sentiment analysis with enhanced styling
    sentiment_class = "sentiment-positive" if sentiment_label == "POSITIVE" else "sentiment-negative"
    st.markdown(
        f"""
        <div class="result-card">
            <p><strong>Sentiment Analysis:</strong></p>
            <p class="{sentiment_class}">{sentiment_label} (Confidence: {sentiment_score:.2f})</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if sentiment_label == "NEGATIVE":
        st.warning("The sentiment is negative. No image will be generated.")
    else:
        st.write("**Generating image from text...**")
        with st.spinner("This may take a few seconds..."):
            sleep(2)
            with torch.no_grad():
                image = pipe(transcription).images[0]
            st.image(image, caption="Generated Image", use_column_width=True)

# Footer section
st.markdown(
    """
    <div class="footer">
        **Instructions:**
        1. Adjust the recording duration using the slider.
        2. Click **Start Recording** to begin the audio capture.
        3. View the transcription, sentiment analysis, and image generation output below.
    </div>
    """,
    unsafe_allow_html=True,
)