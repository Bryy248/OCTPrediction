import cv2
import pickle
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# Constant Variable
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
GABOR_THETAS = np.arange(0, np.pi, np.pi/4)

# Disease Classes
CLASSES = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']

# Preprocessing
def preprocess(image):
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced =clahe.apply(denoised)
    norm = (enhanced - np.mean(enhanced)) / (np.std(enhanced) + 1e-8)
    return enhanced, norm

# Feature Extraction
def feat_extr(image):
    enhanced, norm = preprocess(image)
    features = []
    enhanced_uint8 = enhanced.astype(np.uint8)

    # Statistic
    flat = norm.ravel()
    features.extend([
        np.mean(flat),
        np.std(flat),
        skew(flat),
        kurtosis(flat),
        np.percentile(flat, 10),
        np.percentile(flat, 50),
        np.percentile(flat, 90)
    ])

    # Sobel
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    features.extend([np.mean(grad_mag), np.std(grad_mag)])

    # Canny
    blurred = cv2.GaussianBlur(enhanced_uint8, (5, 5), 1)
    edges = cv2.Canny(blurred, 80, 200)
    edge_density = np.mean(edges > 0)
    features.append(edge_density)

    # LBP
    lbp = local_binary_pattern(enhanced_uint8, LBP_POINTS, LBP_RADIUS, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_POINTS+2, range=(0, LBP_POINTS+2))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-8)
    features.extend(hist)

    # GLCM
    glcm = graycomatrix(enhanced_uint8, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    props = ['contrast', 'correlation', 'energy', 'homogeneity']
    for p in props:
        features.extend(graycoprops(glcm, p).flatten())

    # GABOR Filter Banks
    for theta in GABOR_THETAS:
        kernel = cv2.getGaborKernel((15, 15), 4.0, theta, 10.0, 0.5, 0)
        fimg = cv2.filter2D(enhanced_uint8, cv2.CV_32F, kernel)
        features.extend([np.mean(fimg), np.std(fimg)])
    
    return np.array(features, dtype=np.float32), enhanced_uint8

# Load Model
def load_model_and_scaler():
    with open('best_svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_oct(image, model, scaler):
    features, enhanced = feat_extr(image)
    features_scaled = scaler.transform(features.reshape(1, -1))
    pred_idx = model.predict(features_scaled)[0]
    pred_label = CLASSES[pred_idx]
    return pred_label, enhanced
