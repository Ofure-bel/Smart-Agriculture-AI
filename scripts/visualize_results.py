import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Paths
RESULTS_DIR = "reports/results"
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

#Load training histories
stage1_history = pd.read_csv(os.path.join(RESULTS_DIR, "training_history_stage1.csv"))
stage2_history = pd.read_csv(os.path.join(RESULTS_DIR, "training_history_stage2.csv"))

#Training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(stage1_history["accuracy"], label="Stage 1 Training Accuracy")
plt.plot(stage1_history["val_accuracy"], label="Stage 1 Validation Accuracy")
plt.plot(stage2_history["accuracy"], label="Stage 2 Training Accuracy")
plt.plot(stage2_history["val_accuracy"], label="Stage 2 Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "training_validation_accuracy.png"), dpi=300)
plt.close()

#Training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(stage1_history["loss"], label="Stage 1 Training Loss")
plt.plot(stage1_history["val_loss"], label="Stage 1 Validation Loss")
plt.plot(stage2_history["loss"], label="Stage 2 Training Loss")
plt.plot(stage2_history["val_loss"], label="Stage 2 Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "training_validation_loss.png"), dpi=300)
plt.close()

#Stage1 vs Stage2 test accuracy
stage1_accuracy = 0.8870
stage2_accuracy = 0.9292
models = ["Stage 1", "Stage 2"]
accuracies = [stage1_accuracy * 100, stage2_accuracy * 100]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies)
plt.ylabel("Test Accuracy (%)")
plt.title("Stage 1 vs Stage 2 Test Accuracy")
plt.ylim(0, 100)
for bar, value in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "stage1_vs_stage2_accuracy.png"), dpi=300)
plt.close()

#Stage2 confusion matrix
cm = pd.read_csv(os.path.join(RESULTS_DIR, "confusion_matrix_stage2.csv"), index_col=0)
plt.figure(figsize=(11, 9))
plt.imshow(cm.values)
plt.colorbar(label="Number of Images")
plt.xticks(np.arange(len(cm.columns)), cm.columns, rotation=90)
plt.yticks(np.arange(len(cm.index)), cm.index)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Stage 2 Confusion Matrix")

#Add values inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm.iloc[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix_stage2.png"), dpi=300)
plt.close()

#Per-class F1 scores
with open(os.path.join(RESULTS_DIR, "classification_report_stage1.txt"), "r", encoding="utf-8") as f:
    stage1_report_text = f.read()
with open(os.path.join(RESULTS_DIR, "classification_report_stage2.txt"), "r", encoding="utf-8") as f:
    stage2_report_text = f.read()


#Create values directly from the saved reports' known metrics.
classes = ["Bacterial spot", "Early blight", "Late blight", "Leaf Mold", "Septoria leaf spot", "Spider mites", "Target Spot", "Yellow Leaf Curl Virus", "Mosaic virus", "Healthy"]
stage1_f1 = [0.8689, 0.7829, 0.9134, 0.8857, 0.8601, 0.8471, 0.7676, 0.9560, 0.9060, 0.9025]
stage2_f1 = [0.9216, 0.8705, 0.9580, 0.9441, 0.9418, 0.8920, 0.8195, 0.9693, 0.9725, 0.9222]

x = np.arange(len(classes))
width = 0.38

plt.figure(figsize=(13, 7))
plt.bar(x - width / 2, [value * 100 for value in stage1_f1], width, label="Stage 1")
plt.bar(x + width / 2, [value * 100 for value in stage2_f1], width, label="Stage 2")
plt.ylabel("F1 Score (%)")
plt.xlabel("Disease Class")
plt.title("Per-Class F1 Score: Stage 1 vs Stage 2")
plt.xticks(x, classes, rotation=45, ha="right")
plt.ylim(0, 100)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "per_class_f1_comparison.png"), dpi=300)
plt.close()

#Save comparison summary
comparison = {
    "Stage 1": {
        "test_accuracy": 0.8870,
        "test_accuracy_percent": 88.70,
        "test_loss": 0.3709,
        "macro_f1": 0.8690,
        "weighted_f1": 0.8884
    },
    "Stage 2": {
        "test_accuracy": 0.9292,
        "test_accuracy_percent": 92.92,
        "test_loss": 0.2210,
        "macro_f1": 0.9211,
        "weighted_f1": 0.9303
    },
    "improvement": {
        "accuracy_percentage_points": 4.22,
        "macro_f1_improvement": 0.0521,
        "weighted_f1_improvement": 0.0419,
        "test_loss_reduction": 0.1499
    }
}
with open(os.path.join(RESULTS_DIR, "model_comparison.json"), "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=4)

#Finished
print("\nVisualization complete.")
print("\nGenerated figures:")
for filename in [
    "training_validation_accuracy.png",
    "training_validation_loss.png",
    "stage1_vs_stage2_accuracy.png",
    "confusion_matrix_stage2.png",
    "per_class_f1_comparison.png"
]:
    print(os.path.join(FIGURES_DIR, filename))
print("\nComparison summary:")
print(os.path.join(RESULTS_DIR, "model_comparison.json"))