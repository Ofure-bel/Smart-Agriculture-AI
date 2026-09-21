import json
import os
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "tomato_disease_stage2.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "reports", "results", "class_names.json")
IMAGE_SIZE = (224, 224)

class TomatoDiseasePredictor:
    def __init__(self):
        print("Loading tomato disease model...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)
        print("Model loaded successfully.")
        print(f"Number of classes: {len(self.class_names)}")

    def preprocess_image(self, image):
        #Convert an uploaded PIL image into the format expected by the EfficientNetB0 model
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def predict(self, image):
        #Predict tomato leaf disease and return the predicted_class, confidence, top_predictions
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image, verbose=0)[0]

        #Get indices of highest probabilities
        top_indices = np.argsort(predictions)[::-1][:3]
        top_predictions = []
        for index in top_indices:
            top_predictions.append({"class_name": self.class_names[index], "confidence": float(predictions[index])})

        predicted_index = top_indices[0]
        predicted_class = self.class_names[predicted_index]
        confidence = float(predictions[predicted_index])

        return {
            "predicted_class": predicted_class,
            "predicted_index": predicted_index,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "processed_image": processed_image
        }
    
predictor = TomatoDiseasePredictor()  #Load the model once when the application starts