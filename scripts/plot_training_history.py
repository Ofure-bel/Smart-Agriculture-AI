import os
import pandas as pd
import matplotlib.pyplot as plt

#Configuration
HISTORY_FILE = ("reports/results/" "training_history_stage1.csv")
SAVE_PATH = "reports/figures"
os.makedirs(SAVE_PATH, exist_ok=True)

#Load training history
print("=" * 60)
print("SMART AGRICULTURE AI - TRAINING HISTORY")
print("=" * 60)
print("\nLoading training history...")
history = pd.read_csv(HISTORY_FILE)
print("Training history loaded successfully.")
print("\nAvailable metrics:")
print(list(history.columns))

#Accuracy graph
plt.figure(figsize=(10, 6))
plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("EfficientNetB0 Stage 1 - Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
accuracy_path = (f"{SAVE_PATH}/stage1_accuracy.png")
plt.savefig(accuracy_path,dpi=300)
plt.show()

#Loss graph
plt.figure(figsize=(10, 6))
plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EfficientNetB0 Stage 1 - Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_path = (f"{SAVE_PATH}/stage1_loss.png")
plt.savefig(loss_path, dpi=300)
plt.show()

#Results
best_accuracy = history["val_accuracy"].max()
best_accuracy_epoch = (history["val_accuracy"].idxmax() + 1)
best_loss = history["val_loss"].min()
best_loss_epoch = (history["val_loss"].idxmin() + 1)

print("\n" + "=" * 60)
print("STAGE 1 RESULTS")
print("=" * 60)
print(f"\nBest validation accuracy: " f"{best_accuracy:.4f} " f"({best_accuracy * 100:.2f}%)")
print(f"Best validation accuracy epoch: " f"{best_accuracy_epoch}")
print(f"\nLowest validation loss: " f"{best_loss:.4f}")
print(f"Lowest validation loss epoch: " f"{best_loss_epoch}")
print("\nGraphs saved to:")
print(f"  {accuracy_path}")
print(f"  {loss_path}")
print("\nTraining history analysis complete!")