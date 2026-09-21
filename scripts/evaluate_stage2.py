import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from data_pipeline import load_datasets

#Paths
MODEL_PATH = "models/checkpoints/best_stage2.keras"
CLASS_NAMES_PATH = "reports/results/class_names.json"
REPORT_PATH = "reports/results/classification_report_stage2.txt"
CONFUSION_MATRIX_PATH = "reports/results/confusion_matrix_stage2.csv"
METRICS_PATH = "reports/results/test_metrics_stage2.json"

#Load class names
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)
print("Classes:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

#Load test dataset
print("\nLoading test dataset...")
_, _, test_ds = load_datasets()
print("Test dataset loaded.")

#Load stage 2 model
print("\nLoading Stage 2 model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded from: {MODEL_PATH}")

#Evaluate model
print("\nEvaluating Stage 2 model...")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print("\nStage 2 Test Results")
print("-------------------")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#Generate predictions
print("\nGenerating predictions...")
y_true = []
y_pred = []
for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)
y_true = np.array(y_true)
y_pred = np.array(y_pred)

#Classification report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\nClassification Report")
print("=====================")
print(report)

#Save classification report
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("Stage 2 EfficientNetB0 Classification Report\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n")
    f.write(report)
print(f"Classification report saved to: {REPORT_PATH}")

#Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(CONFUSION_MATRIX_PATH)
print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

#Save summary metrics
metrics = {
    "model": "EfficientNetB0",
    "stage": "Stage 2 - Fine Tuning",
    "test_samples": int(len(y_true)),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "test_accuracy_percent": float(test_accuracy * 100)
}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)
print(f"Test metrics saved to: {METRICS_PATH}")
print("\nEvaluation complete.")
print("Stage 2 evaluation files are ready for comparison with Stage 1.")  #Final message