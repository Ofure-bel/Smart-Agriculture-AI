import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))  #Base directory
SECRET_KEY = os.environ.get("SECRET_KEY", "smart-agriculture-ai-development-key")  #Flask configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")  #Upload configuration
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  #10 MB

#Application information
APP_NAME = "Smart Agriculture AI"
MODEL_NAME = "EfficientNetB0"
MODEL_ACCURACY = 92.92
NUM_CLASSES = 10