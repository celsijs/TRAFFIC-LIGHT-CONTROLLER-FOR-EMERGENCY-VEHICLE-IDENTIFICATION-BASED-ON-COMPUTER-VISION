import tensorflow as tf
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

model = tf.keras.models.load_model('model.h5')
print("Model loaded successfully.")
def load_wav_file(file_path, target_sr=16000):
    # Load audio and resample
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def compute_spectrogram(audio, sr, n_mels=64, n_fft=2048, hop_length=512):
    # Convert waveform to mel-spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)  # Convert to log scale
    return spectrogram_db
def predict_file(file_path):
    audio, sr = load_wav_file(file_path)
    spectrogram = compute_spectrogram(audio, sr)
    spectrogram_resized = tf.image.resize(spectrogram[..., np.newaxis], (64, 64)).numpy()
    spectrogram_resized = spectrogram_resized / 255.0
    prediction = model.predict(spectrogram_resized[np.newaxis, ...])
    predicted_class = np.argmax(prediction)
    return predicted_class

test_file = "Soundset/sounds/traffic/sound_403.wav"
print(f"Predicted class: {predict_file(test_file)}")