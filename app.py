import os
import numpy as np
import cv2
import librosa
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp3', 'wav', 'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Ensure lazy loading of models only when needed
audio_model = None
photo_model = None
video_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === LAZY LOADING OF MODELS (on CPU) ===

def load_audio_model():
    global audio_model
    if audio_model is None:
        print("Loading audio model on CPU...")
        with tf.device('/CPU:0'):  # Force the model to load on CPU
            input_shape_audio = (128, 128, 1)
            audio_model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape_audio),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            audio_model.load_weights('models/Audios_CNN_Model.h5')
    return audio_model

def load_photo_model():
    global photo_model
    if photo_model is None:
        print("Loading photo model on CPU...")
        with tf.device('/CPU:0'):  # Force the model to load on CPU
            photo_model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            photo_model.load_weights('models/Photos_CNN_Model.h5')
    return photo_model

def load_video_model():
    global video_model
    if video_model is None:
        print("Loading video model on CPU...")
        with tf.device('/CPU:0'):  # Force the model to load on CPU
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            base_model.trainable = False
            x = tf.keras.layers.Flatten()(base_model.output)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            video_model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
            video_model.load_weights('models/Videos_VGG16_Model.h5')
    return video_model

# === PREDICT FUNCTIONS ===

def predict_image(filepath):
    model = load_photo_model()
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    K.clear_session()  # Clear session after prediction to free up memory
    return 'Suspicious' if prediction > 0.5 else 'Not Suspicious'

def predict_audio(filepath):
    model = load_audio_model()
    y, sr = librosa.load(filepath, sr=None, duration=5.0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = np.resize(mel, (128, 128))
    mel = np.expand_dims(mel, axis=-1)
    mel = np.expand_dims(mel, axis=0)
    prediction = model.predict(mel)
    K.clear_session()  # Clear session after prediction to free up memory
    return 'Suspicious' if prediction[0] > 0.5 else 'Not Suspicious'

def predict_video(filepath):
    model = load_video_model()
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
        K.clear_session()  # Clear session after prediction to free up memory
        return "Not enough frames"

    frames = np.array(frames)  # shape: (10, 224, 224, 3)
    predictions = model.predict(frames)
    avg_prediction = np.mean(predictions)
    
    K.clear_session()  # Clear session after prediction to free up memory
    return 'Suspicious' if avg_prediction > 0.5 else 'Not Suspicious'

# === FLASK ROUTES ===

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
