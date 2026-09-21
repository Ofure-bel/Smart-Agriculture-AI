import os
import json
import pandas as pd
import tensorflow as tf
from data_pipeline import load_datasets
from build_model import build_model

#Configuration
EPOCHS = 15
CLASS_WEIGHTS_FILE = ("reports/results/class_weights.json")
MODEL_OUTPUT = ("models/saved_models/" "tomato_disease_stage1.keras")
CHECKPOINT_PATH = ("models/checkpoints/" "best_stage1.keras")
HISTORY_OUTPUT = ("reports/results/" "training_history_stage1.csv")

#Create required directories
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("models/saved_models", exist_ok=True)
os.makedirs("reports/results", exist_ok=True)

#Start
print("=" * 60)
print("SMART AGRICULTURE AI - MODEL TRAINING")
print("=" * 60)

#Load class weights
print("\nLoading class weights...")
with open(CLASS_WEIGHTS_FILE, "r") as file:
    class_weights = json.load(file)

#TensorFlow/Keras expects integer keys
class_weights = {int(key): value for key, value in class_weights.items()}
print("Class weights loaded successfully.")

#Load datasets
print("\nLoading TensorFlow datasets...")
(train_dataset, validation_dataset, test_dataset) = load_datasets()
print("Datasets loaded successfully.")

#Build model
print("\nBuilding EfficientNetB0 model...")
model = build_model()

#Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1)
csv_logger = tf.keras.callbacks.CSVLogger(HISTORY_OUTPUT, append=False)

#Train
print("\n" + "=" * 60)
print("STARTING STAGE 1 TRAINING")
print("=" * 60)
print("\nEpochs:", EPOCHS)
print("EfficientNetB0: FROZEN")
print("Classification head: TRAINABLE")
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, class_weight=class_weights, callbacks=[checkpoint, early_stopping, reduce_lr, csv_logger])
model.save(MODEL_OUTPUT)  #Save final stage1 model

#Display results
print("\n" + "=" * 60)
print("STAGE 1 TRAINING COMPLETE!")
print("=" * 60)
print("\nFinal model saved to:")
print(MODEL_OUTPUT)
print("\nBest model checkpoint saved to:")
print(CHECKPOINT_PATH)
print("\nTraining history saved to:")
print(HISTORY_OUTPUT)

print("\nBest validation accuracy:")
best_val_accuracy = max(history.history["val_accuracy"])
print(f"{best_val_accuracy:.4f}")
print("\nBest validation loss:")
best_val_loss = min(history.history["val_loss"])
print(f"{best_val_loss:.4f}")