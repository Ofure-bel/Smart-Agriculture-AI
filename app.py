import os
import sys
import uuid
import json
import time
from flask import (Flask, render_template, request, redirect, url_for, jsonify, flash)
from werkzeug.utils import secure_filename
from model.predictor import predictor
from model.gradcam import generate_gradcam, save_gradcam

#Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

#Disease information
DISEASE_INFO_PATH = os.path.join(PROJECT_ROOT, "data", "disease_info.json")
with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f: DISEASE_INFO = json.load(f)

from config import (SECRET_KEY, UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH, APP_NAME, MODEL_NAME, MODEL_ACCURACY, NUM_CLASSES)  #Application imports
from database.database import (initialize_database, save_prediction, get_predictions, get_prediction, delete_prediction)


app = Flask(__name__)  #Flask application

app.config["SECRET_KEY"] = SECRET_KEY
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

#Create upload directory
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
GRADCAM_FOLDER = os.path.join(PROJECT_ROOT, "static", "gradcam")
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

initialize_database()

def allowed_file(filename):
    #Check whether the uploaded file has an allowed extension
    if not filename:
        return False
    if "." not in filename:
        return False
    extension = (filename.rsplit(".", 1)[1] .lower())
    return extension in ALLOWED_EXTENSIONS


def format_class_name(class_name):
    #Convert the internal PlantVillage class name into a user-friendly disease name
    name = class_name
    if name.startswith("Tomato___"):
        name = name.replace("Tomato___", "")
    name = name.replace("_", " ")
    name = name.replace("Two-spotted spider mite", "Two-Spotted Spider Mite")
    return name


def get_disease_info(class_name):
    #Get additional information for a predicted disease.
    return DISEASE_INFO.get(class_name,
        {
            "name": format_class_name(class_name),
            "category": "Unknown",
            "description": "No additional information is available for this class.",
            "symptoms": [],
            "recommendation": "Consult an agricultural specialist for further assessment."
        }
    )


def confidence_message(confidence):
    #Convert a confidence score into a user-friendly explanation.
    if confidence >= 0.95:
        return {
            "level": "Very High",
            "stars": "★★★★★",
            "message": (
                "The AI is highly confident in this prediction. "
                "The uploaded image closely matches patterns learned during training."
            )
        }
    elif confidence >= 0.85:
        return {
            "level": "High",
            "stars": "★★★★☆",
            "message": (
                "The AI is confident in this prediction. "
                "The detected disease strongly matches the uploaded image."
            )
        }
    elif confidence >= 0.70:
        return {
            "level": "Moderate",
            "stars": "★★★☆☆",
            "message": (
                "The uploaded image shares characteristics with this disease, "
                "although another disease may also be possible."
            )
        }
    else:
        return {
            "level": "Low",
            "stars": "★★☆☆☆",
            "message": (
                "The image was difficult to classify. "
                "A clearer image taken under good lighting may improve the prediction."
            )
        }


def prepare_history_prediction(prediction):
    #Convert a database prediction into the format expected by the History and Prediction Details pages.
    prediction = dict(prediction)
    prediction["original_filename"] = prediction.get("filename")  #Original uploaded filename
    prediction["display_class"] = format_class_name(prediction["predicted_class"])  #User-friendly prediction name
    prediction["disease_info"] = get_disease_info(prediction["predicted_class"])  #Disease information

    #Rebuild the top 3 predictions from the database fields
    top_predictions = []

    if prediction.get("predicted_class"):
        top_predictions.append({
            "class_name": prediction["predicted_class"], "display_class": format_class_name(prediction["predicted_class"]),
            "confidence": prediction["confidence"], "confidence_percent": (prediction["confidence"] * 100)
        })

    if prediction.get("second_class"):
        top_predictions.append({
            "class_name": prediction["second_class"], "display_class": format_class_name(prediction["second_class"]),
            "confidence": prediction["second_confidence"], "confidence_percent": (prediction["second_confidence"] * 100)
        })

    if prediction.get("third_class"):
        top_predictions.append({
            "class_name": prediction["third_class"], "display_class": format_class_name(prediction["third_class"]),
            "confidence": prediction["third_confidence"], "confidence_percent": (prediction["third_confidence"] * 100)
        })

    prediction["top_predictions"] = top_predictions
    return prediction


#Home
@app.route("/")
def index():
    return render_template(
        "index.html",
        app_name=APP_NAME,
        model_name=MODEL_NAME,
        model_accuracy=MODEL_ACCURACY,
        num_classes=NUM_CLASSES
    )


#Detection page
@app.route("/detect")
def detect():
    return render_template(
        "detect.html",
        app_name=APP_NAME
    )


#Prediction
@app.route("/predict", methods=["POST"])
def predict():
    #Check upload
    if "image" not in request.files:
        flash("Please select an image.", "error")
        return redirect(url_for("detect"))

    file = request.files["image"]
    if file.filename == "":
        flash("No image was selected.", "error")
        return redirect(url_for("detect"))

    #Validate extension
    if not allowed_file(file.filename):
        flash("Invalid image format. " "Please upload JPG, JPEG, PNG or WEBP.", "error")
        return redirect(url_for("detect"))

    #Secure filename
    original_filename = secure_filename(file.filename)
    extension = (original_filename.rsplit(".", 1)[1] .lower())
    unique_filename = (f"{uuid.uuid4().hex}.{extension}")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

    #Save image
    try:
        file.save(filepath)
    except Exception:
        flash("Unable to save the uploaded image.", "error")
        return redirect(url_for("detect"))

    #Open image
    try:
        from PIL import Image
        image = Image.open(filepath)
        image.verify()
        image = Image.open(filepath)  #Reopen after verify()
    except Exception:
        if os.path.exists(filepath):
            os.remove(filepath)
        flash("The uploaded file is not a valid image.", "error")
        return redirect(url_for("detect"))

    #AI prediction
    try:
        start_time = time.perf_counter()
        result = predictor.predict(image)

        #Format prediction
        result["display_class"] = format_class_name(result["predicted_class"])
        for item in result["top_predictions"]:
            item["display_class"] = format_class_name(item["class_name"])
        result["confidence_percent"] = (result["confidence"] * 100)

        for item in result["top_predictions"]:
            item["confidence_percent"] = (item["confidence"] * 100)
        analysis_time = time.perf_counter() - start_time
        disease_info = get_disease_info(result["predicted_class"])
        result["disease_info"] = disease_info

        #Generate Grad-CAM visualization
        gradcam_filename = None
        try:
            gradcam_filename = f"{uuid.uuid4().hex}.jpg"
            gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)
            heatmap = generate_gradcam(predictor.model,
                result["processed_image"],
                result["predicted_index"])
            save_gradcam(filepath, heatmap, gradcam_path)
        except Exception as error:
            print(f"Grad-CAM error: {error}")
            gradcam_filename = None
        result["gradcam_image"] = gradcam_filename

        result["confidence_info"] = confidence_message(result["confidence"])
        result["analysis_time"] = round(analysis_time, 2)
        result["observation"] = (
            f"The uploaded tomato leaf most closely matches "
            f"{result['display_class']} with "
            f"{result['confidence_percent']:.2f}% confidence. "
            f"This prediction is based on visual characteristics detected by the AI model "
            f"and should be used alongside field inspection."
        )
    except Exception as error:
        print(f"Prediction error: {error}")
        if os.path.exists(filepath):
            os.remove(filepath)
        flash("An error occurred while analyzing " "the image.", "error")
        return redirect(url_for("detect"))

    prediction_id = save_prediction(original_filename, unique_filename, result)  #Save to database

    #Result page
    return render_template(
        "result.html",
        app_name=APP_NAME,
        prediction=result,
        prediction_id=prediction_id,
        image_filename=unique_filename,
        gradcam_image=gradcam_filename,
        original_filename=original_filename
    )

#Prediction history
@app.route("/history")
def history():
    predictions = get_predictions(limit=100)
    predictions = [prepare_history_prediction(prediction) for prediction in predictions]
    return render_template("history.html", app_name=APP_NAME, predictions=predictions)


#Individual prediction
@app.route("/history/<int:prediction_id>")
def prediction_detail(prediction_id):
    prediction = get_prediction(prediction_id)
    if prediction is None:
        return render_template("error.html", app_name=APP_NAME, message="Prediction not found."), 404
    prediction = prepare_history_prediction(prediction)
    return render_template("prediction_detail.html", app_name=APP_NAME, prediction=prediction, model_accuracy=MODEL_ACCURACY)


#Delete individual prediction
@app.route("/history/<int:prediction_id>/delete", methods=["POST"])
def delete_history_prediction(prediction_id):
    prediction = get_prediction(prediction_id)
    if prediction is None:
        flash("Prediction not found.", "error")
        return redirect(url_for("history"))

    deleted = delete_prediction(prediction_id)  #Delete the database record

    if deleted:
        #Delete the associated uploaded image.
        image_filename = prediction["image_filename"]
        if image_filename:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            if os.path.isfile(image_path):
                try:
                    os.remove(image_path)
                except OSError:
                    #The database record has already been deleted,
                    #so don't fail the request if the file is gone.
                    pass
        flash("Prediction deleted successfully.", "success")
    else:
        flash("Unable to delete prediction.", "error")

    return redirect(url_for("history"))


#About page
@app.route("/about")
def about():
    return render_template(
        "about.html",
        app_name=APP_NAME,
        model_name=MODEL_NAME,
        model_accuracy=MODEL_ACCURACY,
        num_classes=NUM_CLASSES
    )


#Health check
@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "application": APP_NAME,
            "model": MODEL_NAME,
            "model_accuracy": MODEL_ACCURACY,
            "classes": NUM_CLASSES
        }
    )


#API prediction endpoint
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image provided."}), 400
    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No image selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": ("Invalid image format.")}), 400

    try:
        from PIL import Image
        image = Image.open(file)
        image.verify()
        file.seek(0)
        image = Image.open(file)
        result = predictor.predict(image)
        result["disease_info"] = get_disease_info(result["predicted_class"])
        result["display_class"] = format_class_name(result["predicted_class"])
        result["confidence_percent"] = (result["confidence"] * 100)
        result["confidence_info"] = confidence_message(result["confidence"])

        for item in result["top_predictions"]:
            item["display_class"] = format_class_name(item["class_name"])
            item["confidence_percent"] = (item["confidence"] * 100)

        result["observation"] = (f"The uploaded tomato leaf most closely matches " f"{result['display_class']} with " f"{result['confidence_percent']:.2f}% confidence.")
        return jsonify({"success": True, "prediction": result})

    except Exception as error:
        print(f"API prediction error: {error}")
        return jsonify({"success": False, "error": ("Unable to analyze image.")}), 500


#Error page 413- File too large
@app.errorhandler(413)
def file_too_large(error):
    return render_template("error.html", app_name=APP_NAME, message=("The uploaded image is too large. " "Maximum size is 10 MB.")), 413


#Error 404
@app.errorhandler(404)
def page_not_found(error):
    return render_template("error.html", app_name=APP_NAME, message="The requested page was not found."), 404


#Error 500
@app.errorhandler(500)
def internal_server_error(error):
    return render_template("error.html", app_name=APP_NAME, message=("An unexpected error occurred. " "Please try again.")), 500


#Run application
if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )