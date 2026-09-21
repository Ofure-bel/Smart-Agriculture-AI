import os
import sys
from PIL import Image

#Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.predictor import predictor

#Find a test image
TEST_FOLDER = os.path.join(PROJECT_ROOT, "dataset", "tomato", "Tomato___healthy")

def find_first_image(folder):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    for filename in os.listdir(folder):
        if filename.lower().endswith(valid_extensions):
            return os.path.join(folder, filename)
    return None

#Find image
image_path = find_first_image(TEST_FOLDER)
if image_path is None:
    raise FileNotFoundError(f"No image found in: {TEST_FOLDER}")

#Test predictor
print("\nTesting deployment predictor")
print("============================")
print(f"Image: {image_path}")
image = Image.open(image_path)
result = predictor.predict(image)

#Display result
print("\nPrediction:")
print(result["predicted_class"])
print(f"Confidence: " f"{result['confidence'] * 100:.2f}%")
print("\nTop 3 predictions:")
for item in result["top_predictions"]:
    print(f"- {item['class_name']}: " f"{item['confidence'] * 100:.2f}%")
print("\nPredictor test completed successfully.")