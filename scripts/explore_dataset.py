import os
from PIL import Image

DATASET_PATH = "dataset/tomato"  #Location of our tomato dataset
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")  #Image file extensions we will accept

def explore_dataset():
    print("=" * 60)
    print("SMART AGRICULTURE AI - DATASET EXPLORATION")
    print("=" * 60)

    #Check that the dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\nERROR: Dataset not found at: {DATASET_PATH}")
        return

    classes = sorted([folder for folder in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, folder))])  #Get class folders
    print(f"\nNumber of classes: {len(classes)}")

    total_images = 0
    corrupted_images = 0
    print("\nImages per class:")
    print("-" * 60)

    for class_name in classes:
        class_path = os.path.join(DATASET_PATH, class_name)
        images = [file for file in os.listdir(class_path) if file.endswith(IMAGE_EXTENSIONS)]

        print(f"{class_name}: {len(images)} images")
        total_images += len(images)

        #Check images for corruption
        for image_name in images:
            image_path = os.path.join(class_path, image_name)
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception:
                corrupted_images += 1
                print(f"  WARNING: Corrupted image -> {image_name}")
                
    print("-" * 60)
    print(f"Total images: {total_images}")
    print(f"Corrupted images: {corrupted_images}")

    #Check image dimensions from the first image in each class
    print("\nSample image dimensions:")
    print("-" * 60)

    for class_name in classes:
        class_path = os.path.join(DATASET_PATH, class_name)
        images = [file for file in os.listdir(class_path) if file.endswith(IMAGE_EXTENSIONS)]
        if images:
            image_path = os.path.join(class_path, images[0])
            try:
                with Image.open(image_path) as img:
                    print(f"{class_name}: {img.size}")
            except Exception:
                print(f"{class_name}: Could not read image")
    print("\nDataset exploration complete!")

if __name__ == "__main__":
    explore_dataset()