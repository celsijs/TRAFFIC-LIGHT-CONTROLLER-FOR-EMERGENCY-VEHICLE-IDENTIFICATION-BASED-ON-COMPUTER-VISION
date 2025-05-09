import tensorflow as tf
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Load and preprocess .wav files
def load_wav_file(file_path, target_sr=16000):
    # Load audio and resample
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def compute_spectrogram(audio, sr, n_mels=64, n_fft=2048, hop_length=512):
    # Convert waveform to mel-spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)  # Convert to log scale
    return spectrogram_db

def prepare_data(data_dir, target_sr=16000, img_size=(64, 64)):
    X, y = [], []
    classes = sorted(os.listdir(data_dir))  # Assuming subdirectories for each class
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_dir, file_name)
                audio, sr = load_wav_file(file_path, target_sr)
                spectrogram = compute_spectrogram(audio, sr)
                spectrogram_resized = tf.image.resize(spectrogram[..., np.newaxis], img_size).numpy()
                X.append(spectrogram_resized)
                y.append(label)
    return np.array(X), np.array(y), classes

# Step 2: Load dataset
data_dir = "Soundset/sounds"
X, y, class_names = prepare_data(data_dir)

# Normalize and split data
X = X / 255.0  # Normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model_save_path = "model.h5"
model.save(model_save_path)



# Step 5: Evaluate and classify
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# To predict
def predict_file(file_path):
    audio, sr = load_wav_file(file_path)
    spectrogram = compute_spectrogram(audio, sr)
    spectrogram_resized = tf.image.resize(spectrogram[..., np.newaxis], (64, 64)).numpy()
    spectrogram_resized = spectrogram_resized / 255.0
    prediction = model.predict(spectrogram_resized[np.newaxis, ...])
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

test_file = "Soundset/sounds/ambulance/sound_3.wav"
print(f"Predicted class: {predict_file(test_file)}")
