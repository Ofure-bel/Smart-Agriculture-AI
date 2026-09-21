import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score)
from data_pipeline import create_dataset

#Configuration
MODEL_PATH = ("models/checkpoints/" "best_stage1.keras")
TEST_FILE = ("reports/results/test.csv")
CLASS_NAMES_FILE = ("reports/results/class_names.json")
RESULTS_PATH = ("reports/results/")
FIGURES_PATH = ("reports/figures/")

#Directories
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

#Start
print("=" * 60)
print("SMART AGRICULTURE AI - MODEL EVALUATION")
print("=" * 60)

#Load class names
print("\nLoading class names...")
with open(CLASS_NAMES_FILE, "r") as file:
    class_names = json.load(file)
print(f"Number of classes: {len(class_names)}")

#Load test data
print("\nLoading test dataset...")
test_data = pd.read_csv(TEST_FILE)
print(f"Test images: {len(test_data)}")
test_dataset = create_dataset(test_data, training=False)

#Load model
print("\nLoading best Stage 1 model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

#Model Evaluation
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

#Predictions
print("\nGenerating predictions...")
predictions = model.predict(test_dataset, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_data["label"].values
accuracy = accuracy_score(true_classes, predicted_classes)  #Accuracy
report = classification_report(true_classes, predicted_classes, target_names=class_names, digits=4)  #Classification report
cm = confusion_matrix(true_classes, predicted_classes)  #Confusion matrix

#Save classification report
report_file = (f"{RESULTS_PATH}" "classification_report_stage1.txt")
with open(report_file, "w") as file:
    file.write("SMART AGRICULTURE AI\n" )
    file.write("STAGE 1 MODEL EVALUATION\n\n")
    file.write(f"Test Accuracy: " f"{accuracy:.4f}\n")
    file.write(f"Test Accuracy (%): " f"{accuracy * 100:.2f}%\n\n")
    file.write("Classification Report\n")
    file.write("=====================\n\n")
    file.write(report)

np.savetxt(f"{RESULTS_PATH}confusion_matrix_stage1.csv", cm, delimiter=",", fmt="%d")  #Sace confusion matrix

#Results summary
print("\n" + "=" * 60)
print("STAGE 1 TEST RESULTS")
print("=" * 60)
print(f"\nTest Loss: " f"{test_loss:.4f}")
print(f"Test Accuracy: " f"{test_accuracy:.4f}")
print(f"Test Accuracy: " f"{test_accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)
print("\nFiles saved:")
print(f"  {report_file}")
print(f"  {RESULTS_PATH}" "confusion_matrix_stage1.csv")
print("\nStage 1 evaluation complete!")