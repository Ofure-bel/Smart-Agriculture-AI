import os
import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = "dataset/tomato"
SAVE_PATH = "reports/figures"
os.makedirs(SAVE_PATH, exist_ok=True)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

classes = sorted([folder for folder in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, folder))])

image_counts = []
for class_name in classes:
    class_path = os.path.join(DATASET_PATH, class_name)
    images = [img for img in os.listdir(class_path) if img.endswith(IMAGE_EXTENSIONS)]
    image_counts.append(len(images))

#Bar chart
plt.figure(figsize=(12,6))
bars = plt.bar(classes, image_counts, color="forestgreen", edgecolor="black")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 40, str(height), ha='center', fontsize=8)

plt.xticks(rotation=45, ha="right", fontsize=9)
plt.ylabel("Number of Images")
plt.title("Tomato Disease Dataset Distribution")
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/dataset_distribution.png")
plt.show()

#Sample images
fig, axes = plt.subplots(2, 5, figsize=(18,8))
axes = axes.flatten()
for i, class_name in enumerate(classes):
    class_path = os.path.join(DATASET_PATH, class_name)
    image_name = os.listdir(class_path)[0]
    image_path = os.path.join(class_path, image_name)
    image = Image.open(image_path)
    axes[i].imshow(image)
    axes[i].set_title(class_name.replace("Tomato___",""), fontsize=8)
    axes[i].axis("off")
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/sample_images.png")
plt.show()

print("\nVisualizations saved successfully!")
print(f"\nLocation: {SAVE_PATH}")