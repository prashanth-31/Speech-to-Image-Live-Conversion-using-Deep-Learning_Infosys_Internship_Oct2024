import streamlit as st
import sounddevice as sd
import torch
import numpy as np
from scipy.io.wavfile import write
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from diffusers import StableDiffusionPipeline
import librosa
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from time import sleep
from transformers import pipeline

# Initialize the Snowball Stemmer and Sentiment Analysis pipeline from HuggingFace
nltk.download('punkt')
nltk.download('vader_lexicon')
stemmer = SnowballStemmer("english")

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
st.title("Audio-to-Image Generator with Sentiment Analysis")
st.write("Record your audio, transcribe it, apply stemming to the transcription, perform sentiment analysis, and generate an image from the processed text.")

# Slider for audio recording duration
duration = st.slider("Select recording duration (seconds)", min_value=1, max_value=30, value=15, step=1)

# Record audio
if st.button("Record"):
    fs = 16000

    # Inform user of recording
    st.write("Recording... Please speak clearly.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.write("Recording complete!")

    # Save recorded audio to a file
    audio_path = "audio_input.wav"
    write(audio_path, fs, (audio * 32767).astype(np.int16))

    # Transcribe audio with Whisper
    st.write("Transcribing audio...")
    audio_input, _ = librosa.load(audio_path, sr=16000)
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    predicted_ids = model.generate(input_features)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    st.write("Original Transcription:", transcription)

    # Tokenize and Apply Snowball Stemming to transcription
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(transcription)
    st.write("Tokenized Words:", words)

    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_transcription = " ".join(stemmed_words)
    st.write("Stemmed Transcription:", stemmed_transcription)

    # Sentiment Analysis using BERT
    sentiment = sentiment_analysis(stemmed_transcription)
    st.write("Sentiment Analysis:", sentiment)

    # Check if sentiment is negative
    if sentiment[0]['label'] == "NEGATIVE":
        st.write("The sentiment is negative, so no image will be generated.")
    else:
        # Generate image with Stable Diffusion
        st.write("Generating image from text...")
        with st.spinner('This may take a few seconds...'):
            sleep(2)
            with torch.no_grad():
                image = pipe(stemmed_transcription).images[0]
            st.image(image, caption="Generated Image", use_column_width=True)

# Instructions
st.write("Click 'Record' to transcribe audio, apply stemming, perform sentiment analysis, and generate an image.")