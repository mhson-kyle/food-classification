import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

LABELS = pd.read_csv('food_classification/dataset/meta/meta/labels.txt', header=None)
MODEL_WEIGHT = 'food_classification/drive/mobilenetv2_tuned_1.h5'
model = load_model(MODEL_WEIGHT)

def train_img(img_path):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        shear_range=0.3,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.25,
    )
    train_data = train_datagen.flow_from_directory(
        img_path,
        target_size=(224, 224),
        batch_size=128,
        shuffle=False,
        classes=['apple_pie']
    )
    return train_data

def test_img(img_path):
    test_datagen = ImageDataGenerator(
        rescale=1./255,
    )
    test_data = test_datagen.flow_from_directory(
        img_path,
        target_size=(224, 224),
        batch_size=128,
        shuffle=False,
        classes=['apple_pie']
    )
    return test_data
    
def img_preprocssing(img_path):
    img = Image.open(img_path)
    img = img.resize((224,224), Image.ANTIALIAS)
    img = np.reshape(img, [1, 224, 224, 3])
    return img

def predict_class(pred_value):
    pred_df = pd.DataFrame(pred_value[0])
    max = pred_df.max()[0]
    pred_class = pred_df[pred_df[0] == max].index.values
    return pred_class[0]

def predict_img(img_path, model=model):
    img = img_preprocssing(img_path)
    pred_value = model.predict(img)
    pred_class = predict_class(pred_value)
    return LABELS.iloc[pred_class].values[0]