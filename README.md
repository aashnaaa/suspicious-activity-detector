# 🔍 Suspicious Content Detection (Flask + Deep Learning)

This is a Flask web application that detects **suspicious content** in:
- 🖼️ Images
- 🎧 Audio files
- 🎥 Videos

It uses pre-trained deep learning models (CNN + VGG16) to classify uploaded content as either **Suspicious** or **Not Suspicious**.

---

## 📁 Project Structure

```
Project/
├── models/               # Pre-trained .h5 model files
├── static/
│   ├── uploads/          # Uploaded media by users
│   └── images/           # Static assets (e.g., logo)
├── templates/
│   └── index.html        # Frontend HTML page
├── venc/                 # Virtual environment (ignored)
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore rules
```

---

## 🧠 Models Used

| Type   | Architecture  | Model File                  |
|--------|----------------|-----------------------------|
| Image  | Custom CNN     | `Photos_CNN_Model.h5`       |
| Audio  | CNN on MelSpec | `Audios_CNN_Model.h5`       |
| Video  | VGG16 (frozen) | `Videos_VGG16_Model.h5`     |

---

## 🚀 Features

- Upload and analyze **images, audio, or video**
- Deep learning-based classification
- Real-time predictions through a web interface
- Modular code with separate models

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/aashnaaa/suspicious-content-detector.git
cd suspicious-content-detector
```

### 2. Create and activate virtual environment

```bash
python -m venv venc
source venc/bin/activate       # On macOS/Linux
venc\Scripts\activate        # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask app

```bash
python app.py
```

Then open your browser and go to:  
`http://127.0.0.1:5000/`

---

## 📦 Requirements

Install from `requirements.txt`, or ensure these major libraries:

- Flask
- TensorFlow / Keras
- NumPy
- OpenCV
- librosa

---

## 🖼️ Screenshot

![App Screenshot](https://github.com/aashnaaa/suspicious-activity-detector/blob/master/static/images/app_screenshot.png)

---

## 📜 License

This project is for educational purposes.

---

## 🙋‍♀️ Author

**Aashna**  
GitHub: [@aashnaaa](https://github.com/aashnaaa)

---
