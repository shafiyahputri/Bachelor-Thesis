import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from skimage.feature import local_binary_pattern
import joblib
import rembg
import time

# Inisialisasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model & scaler
svm_model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label = ['arummanis', 'gadung', 'manalagi']

# Parameter Gabor
ksize = (15, 15)
sigma = 5
theta_list = np.linspace(0, np.pi, 8, endpoint=False)
lambda_list = [np.pi/8, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
gamma = 0.5
psi = 0

def create_gabor_filter_bank():
    filters = []
    for theta in theta_list:
        for lambd in lambda_list:
            kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
            filters.append((kernel, theta, lambd))
    return filters

def apply_gabor_filters(image, filters):
    filtered_images = []
    for kernel, _, _ in filters:
        filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
        filtered_images.append(filtered)
    return filtered_images

def predict_image(image_path):
    # Preprocess
    img_array = cv2.imread(image_path)
    resized_img = cv2.resize(img_array, (500, 375))
    output_rgba = rembg.remove(resized_img)
    output_rgb = output_rgba[:, :, :3]
    gray_image = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2GRAY)

    # LBP
    lbp = local_binary_pattern(gray_image, P=16, R=2, method='uniform')
    mask = (gray_image > 0)
    lbp_masked = lbp[mask]
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins), density=True)
    featureLbp = pd.DataFrame([hist])

    # Gabor
    filters = create_gabor_filter_bank()
    filtered_images = apply_gabor_filters(gray_image, filters)
    _, binary_mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
    mask = binary_mask.astype(bool)

    gabor_features = []
    for fimg in filtered_images:
        masked_values = fimg[mask]
        gabor_features.extend([masked_values.mean(), masked_values.var()])
    featureGbr = pd.DataFrame([gabor_features])

    # Gabungkan dan prediksi
    combined_features = pd.concat([featureLbp, featureGbr], axis=1)
    X_scaled = scaler.transform(combined_features)
    prediction = svm_model.predict(X_scaled)[0]
    return label[int(prediction)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            start_time = time.time()
            prediction = predict_image(save_path)
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            return render_template('index.html', prediction=prediction, image_path=save_path, processing_time=processing_time)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
