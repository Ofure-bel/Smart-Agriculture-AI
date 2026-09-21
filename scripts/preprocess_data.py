import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

#Configuration
DATASET_PATH = "dataset/tomato"
RESULTS_PATH = "reports/results"
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

os.makedirs(RESULTS_PATH, exist_ok=True)  #Create output directory

#Find all images
print("=" * 60)
print("SMART AGRICULTURE AI - DATA PREPROCESSING")
print("=" * 60)
print("\nSearching for images...")

image_paths = []
labels = []

classes = sorted([folder for folder in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, folder))])
print(f"\nNumber of classes found: {len(classes)}")

#Create numerical labels
class_to_index = {class_name: index for index, class_name in enumerate(classes)}
for class_name in classes:
    class_path = os.path.join(DATASET_PATH, class_name)
    for image_name in os.listdir(class_path):
        if image_name.endswith(IMAGE_EXTENSIONS):
            image_path = os.path.join(class_path, image_name)
            image_paths.append(image_path)
            labels.append(class_to_index[class_name])

#Create dataframe
data = pd.DataFrame({"image_path": image_paths, "label": labels})
print(f"Total images found: {len(data)}")

train_data, temporary_data = train_test_split(data, test_size=(1 - TRAIN_SIZE), stratify=data["label"], random_state=RANDOM_SEED)  #First split= 70% TRAIN and 30% TEMPORARY
validation_data, test_data = train_test_split(temporary_data, test_size=0.5, stratify=temporary_data["label"], random_state=RANDOM_SEED)  #Second split= half validation/half test: 30% temporary becomes: 15% validation and 15% test

#Shuffle each dataset
train_data = train_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
validation_data = validation_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

#Save splits
train_data.to_csv(os.path.join(RESULTS_PATH, "train.csv"), index=False)
validation_data.to_csv(os.path.join(RESULTS_PATH, "validation.csv"), index=False)
test_data.to_csv(os.path.join(RESULTS_PATH, "test.csv"),index=False)

#Save class names
with open(os.path.join(RESULTS_PATH, "class_names.json"), "w") as file:
    json.dump(classes, file, indent=4)

#Display results
print("\n" + "=" * 60)
print("DATASET SPLIT RESULTS")
print("=" * 60)
print(f"\nTraining images:   {len(train_data)}")
print(f"Validation images: {len(validation_data)}")
print(f"Testing images:    {len(test_data)}")
print(f"Total images:      {len(data)}")
print("\n" + "=" * 60)   


#Class distribution
print("TRAINING CLASS DISTRIBUTION")
print("=" * 60)
for class_name, class_index in class_to_index.items():
    count = len(train_data[train_data["label"] == class_index])
    print(f"{class_name}: {count}")
print("\n" + "=" * 60)
print("VALIDATION CLASS DISTRIBUTION")
print("=" * 60)

for class_name, class_index in class_to_index.items():
    count = len(validation_data[validation_data["label"] == class_index])
    print(f"{class_name}: {count}")
print("\n" + "=" * 60)
print("TEST CLASS DISTRIBUTION")
print("=" * 60)

for class_name, class_index in class_to_index.items():
    count = len(test_data[test_data["label"] == class_index])
    print(f"{class_name}: {count}")

#Finished
print("\n" + "=" * 60)
print("PREPROCESSING STEP COMPLETE!")
print("=" * 60)
print("\nFiles created:")
print("  reports/results/train.csv")
print("  reports/results/validation.csv")
print("  reports/results/test.csv")
print("  reports/results/class_names.json")