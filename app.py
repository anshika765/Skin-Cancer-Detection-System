import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, url_for
import tensorflow as tf

# Suppress background logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Upload folder config
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 1. Trained model load karo
MODEL_PATH = 'skin_cancer_best_model.keras'
print("Loading trained weights...", flush=True)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!", flush=True)

# 2. 9 Classes in exact flow order
CLASS_NAMES = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='Please select an image.')
        
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No file selected.')
        
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Model prediction
            processed_img = preprocess_image(file_path)
            preds = model.predict(processed_img)[0]

            top_idx = int(np.argmax(preds))
            predicted_class = CLASS_NAMES[top_idx].title()
            confidence = float(preds[top_idx] * 100)

            # Format sorted probability scores for new UI bars
            all_scores = [
                {"name": CLASS_NAMES[i].title(), "prob": f"{preds[i] * 100:.2f}"}
                for i in range(len(CLASS_NAMES))
            ]
            all_scores.sort(key=lambda x: float(x["prob"]), reverse=True)

            # Frontend ke liye clean URL path
            web_image_url = url_for('static', filename=f'uploads/{filename}')

            return render_template(
                'index.html',
                image_url=web_image_url,
                prediction=predicted_class,
                confidence=f"{confidence:.2f}%",
                scores=all_scores
            )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)