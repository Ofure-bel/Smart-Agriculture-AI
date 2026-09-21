import os
import json
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

#Configuration
TRAIN_CSV = "reports/results/train.csv"
CLASS_NAMES_FILE = "reports/results/class_names.json"
OUTPUT_FILE = "reports/results/class_weights.json"

#Load data
print("=" * 60)
print("SMART AGRICULTURE AI - CLASS WEIGHT CALCULATION")
print("=" * 60)
print("\nLoading training dataset...")

train_data = pd.read_csv(TRAIN_CSV)

with open(CLASS_NAMES_FILE, "r") as file:
    class_names = json.load(file)

#Calculate class weights
classes = np.array(sorted(train_data["label"].unique()))
weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_data["label"])
class_weights = {str(int(class_index)): float(weight) for class_index, weight in zip(classes, weights)}  #Convert NumPy values to normal Python values

#Display results
print("\nClass weights:")
print("-" * 60)

for class_index in classes:
    class_name = class_names[int(class_index)]
    weight = class_weights[str(int(class_index))]
    print(f"{class_index}: " f"{class_name:<55} " f"Weight: {weight:.4f}")

with open(OUTPUT_FILE, "w") as file:
    json.dump(class_weights, file, indent=4)  #Save class weights

print("\n" + "=" * 60)
print("CLASS WEIGHT CALCULATION COMPLETE!")
print("=" * 60)

print(f"\nSaved to:")
print(OUTPUT_FILE)