import os
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
import librosa
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp3', 'wav', 'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Rebuilding the Audio CNN model
def create_audio_model(input_shape):
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape_audio = (128, 128, 1)
audio_model = create_audio_model(input_shape_audio)
audio_model.load_weights('models/Audios_CNN_Model.h5')

# Rebuilding the Photo CNN model
def create_photo_model():
    model = Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0003), loss='binary_crossentropy', metrics=['accuracy'])
    return model

photo_model = create_photo_model()
photo_model.load_weights('models/Photos_CNN_Model.h5')

# Rebuilding the Video VGG16 model
def create_vgg16_model(img_height, img_width):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

img_height_video, img_width_video = 224, 224
video_model = create_vgg16_model(img_height_video, img_width_video)
video_model.load_weights('models/Videos_VGG16_Model.h5')  # Use your correct trained weights

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = photo_model.predict(img_array)[0][0]
    return 'Suspicious' if prediction > 0.5 else 'Not Suspicious'

def predict_audio(filepath):
    y, sr = librosa.load(filepath, sr=None, duration=5.0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = np.resize(mel, (128, 128))
    mel = np.expand_dims(mel, axis=-1)
    mel = np.expand_dims(mel, axis=0)
    prediction = audio_model.predict(mel)
    return 'Suspicious' if prediction[0] > 0.5 else 'Not Suspicious'

def predict_video(filepath):
    cap = cv2.VideoCapture(filepath)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    # Sample 10 evenly spaced frames
    sample_indices = np.linspace(0, frame_count - 1, 10).astype(int)

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        resized = cv2.resize(frame, (224, 224)) / 255.0
        frames.append(resized)

    cap.release()

    if not frames:
        return "Not enough frames"

    frames = np.array(frames)  # shape: (10, 224, 224, 3)
    predictions = video_model.predict(frames)
    avg_prediction = np.mean(predictions)

    return 'Suspicious' if avg_prediction > 0.5 else 'Not Suspicious'

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    filename = ''
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            ext = filename.rsplit('.', 1)[1].lower()
            if ext in ['jpg', 'jpeg', 'png']:
                result = predict_image(filepath)
            elif ext in ['mp3', 'wav']:
                result = predict_audio(filepath)
            elif ext == 'mp4':
                result = predict_video(filepath)
            else:
                result = 'Unsupported file format'
    return render_template('index.html', result=result, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)