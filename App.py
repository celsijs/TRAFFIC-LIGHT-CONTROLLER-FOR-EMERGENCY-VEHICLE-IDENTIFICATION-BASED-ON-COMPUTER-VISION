from flask import Flask, render_template, flash, request, session, send_file
from flask import render_template, redirect, url_for, request
import warnings
import datetime
import tensorflow as tf
import numpy as np
import os
import librosa
model = tf.keras.models.load_model('model.h5')
print("Model loaded successfully.")
app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/Test")
def Test():
    return render_template('Test.html')


@app.route("/ImageTest")
def ImageTest():
    res = ''
    result1 = ''

    import cv2
    from ultralytics import YOLO

    dd1 = 0

    # Load the YOLOv8 model
    model = YOLO('runs/detect/Ambu/weights/best.pt')
    # Open the video file
    # video_path = "path/to/your/video/file.mp4"
    cap = cv2.VideoCapture(0)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=0.7)
            for result in results:
                if result.boxes:
                    box = result.boxes[0]
                    class_id = int(box.cls)
                    object_name = model.names[class_id]
                    print(object_name)

                    if object_name == 'Ambulance':
                        dd1 += 1

                    if dd1 == 20:
                        dd1 = 0
                        annotated_frame = results[0].plot()
                        cv2.imwrite("static/Out/alert.jpg", annotated_frame)
                        cap.release()
                        cv2.destroyAllWindows()

                        return render_template('Test.html', result="Ambulance", gry="static/Out/alert.jpg")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv11 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    return render_template('ImageTest.html')









@app.route("/testsound", methods=['GET', 'POST'])
def testsound():
    if request.method == 'POST':

        file = request.files['fileupload']
        file.save('static/Out/' + file.filename)

        live_prediction = predict_file('static/Out/' + file.filename)

        res = live_prediction
        print(res)
        gry = ''
        if res ==2:

            res = 'Traffic'
            result1 = 'Normal'
            gry = 'static/emoji/red.jpg'

        else:
            res = 'Ambulance'
            result1 = 'Emergency'
            gry = 'static/emoji/green.jpg'

        return render_template('Test.html', result=res, result1=result1, gry=gry)


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

def sendmsg(targetno, message):
    import requests
    requests.post(
        "http://smsserver9.creativepoint.in/api.php?username=fantasy&password=596692&to=" + targetno + "&from=FSSMSS&message=Dear user  your msg is " + message + " Sent By FSMSG FSSMSS&PEID=1501563800000030506&templateid=1507162882948811640")


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
