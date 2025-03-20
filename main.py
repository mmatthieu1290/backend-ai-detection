from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from io import BytesIO
from pydantic import BaseModel 
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# uvicorn main:app --host 127.0.0.1 --port 8000 --reload


# Charger le modèle
model = tf.keras.models.load_model("model.keras")

# Initialiser FastAPI
description='Hello world'

app = FastAPI(title="Detection of IA image generated",description=description)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pour autoriser tout le monde, sinon mets ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fonctions d'extraction de features
def calculate_glcm_features(image_gray):
    glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    return {'glcm_contrast': contrast, 'glcm_dissimilarity': dissimilarity}



def calculate_edge_features(image_gray):
    laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    edges = cv2.Canny(image_gray, 100, 200)
    edge_density = np.mean(edges)
    return {'laplacian_var': laplacian_var, 'edge_density': edge_density}

def calculate_fft_features(image_gray):
    fft = np.fft.fftshift(np.fft.fft2(image_gray))
    magnitude = np.log(np.abs(fft) + 1e-6)
    h, w = magnitude.shape
    crop_size = 50
    high_freq = magnitude[h//2 - crop_size:h//2 + crop_size, w//2 - crop_size:w//2 + crop_size]
    return {'fft_high_freq_mean': np.mean(high_freq)}

def extract_all_features(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = {}
    features.update(calculate_glcm_features(gray))
    features.update(calculate_edge_features(gray))
    features.update(calculate_fft_features(gray))
    return features

def batch_feature_extraction(images, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        features = list(executor.map(extract_all_features, images))
    return pd.DataFrame(features)    

def load_model():
    global model
    model = tf.keras.models.load_model("model.keras")

@app.get('/')
async def index():
    message = "This API detects if an image has been built by an IA.\
          If you want to learn more, check out documentation of \
            the api at `/docs`"
    return message  

@app.post("/predict")
async def predict(file: UploadFile):

    feature_columns =  ["glcm_contrast", "glcm_dissimilarity", "laplacian_var", "edge_density", "fft_high_freq_mean"]
    contents = await file.read()
    file_name = file.filename.replace('·','')
    save_path = f"./{file_name}"
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{save_path}**********************************") 
    with open(save_path, "wb") as f:
       f.write(contents)

    image = cv2.imread(f'./{file_name}')
 #   print("ok charger image")
 #   image = np.array(Image.open(BytesIO(contents))).astype(np.uint8)
    features = batch_feature_extraction([image])
    features['label'] = 0
    feature_array = features[feature_columns].values.astype(np.float32)
    print(feature_array.shape)
    labels = features["label"].values.astype(np.float32)

    def load_and_preprocess(path):
       img = tf.io.read_file(path)
    #   img = tf.image.decode_jpeg(tf.io.encode_jpeg(image), channels=3)
       img = tf.image.decode_jpeg(img, channels=3)
       img = tf.image.resize(img, (224, 224))
       img = tf.keras.applications.efficientnet.preprocess_input(img)
       return img

    image_ds = tf.data.Dataset.from_tensor_slices([save_path]).map(load_and_preprocess)
    # features manuais
    feature_ds = tf.data.Dataset.from_tensor_slices(feature_array)
    # labels
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    # combinar todos os componentes
    combined_ds = tf.data.Dataset.zip(
        ((image_ds, feature_ds), label_ds)
    ).batch(32).prefetch(tf.data.AUTOTUNE)
    
    model = tf.keras.models.load_model("model.keras")
    model1 = tf.keras.models.load_model("model_1.keras")
    model2 = tf.keras.models.load_model("model_2.keras")
    model3 = tf.keras.models.load_model("model_3.keras")
    model4 = tf.keras.models.load_model("model_4.keras")
    model5 = tf.keras.models.load_model("model_5.keras")
    result = model.predict(combined_ds)
    result1 = model1.predict(combined_ds)
    result2 = model2.predict(combined_ds)
    result3 = model3.predict(combined_ds)
    result4 = model4.predict(combined_ds)
    result5 = model5.predict(combined_ds)
    proba = max([float(result),float(result1),float(result2),\
                 float(result3),float(result4),float(result5)])
    if proba < 0.7610199:
        proba_adjusted = proba / (2*0.7610199)
    else:
        proba_adjusted = (0.5 / (1-0.7610199)) * (proba - 0.7610199) + 0.5    
    return {"probability_AI": proba_adjusted}

if __name__ == "__main__":
    import uvicorn
#    load_model()
    uvicorn.run(app, host="127.0.0.1", port=8000)