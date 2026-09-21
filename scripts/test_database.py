import os
import sys

#Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.database import (initialize_database,save_prediction,get_predictions)

print("\nTesting database")
print("================")
initialize_database()  #Initialize database
print("Database initialized successfully.")

#Test prediction
test_result = {
    "predicted_class":
        "Tomato___healthy",
    "confidence":
        0.9998,
    "top_predictions": [
        {"class_name": "Tomato___healthy", "confidence": 0.9998},
        {"class_name": "Tomato___Septoria_leaf_spot", "confidence": 0.0001},
        {"class_name": "Tomato___Target_Spot", "confidence": 0.0001}
    ]
}

#Save
prediction_id = save_prediction("test_leaf.jpg", test_result)
print(f"Test prediction saved with ID: " f"{prediction_id}")

#Retrieve
predictions = get_predictions()
print(f"Predictions in database: " f"{len(predictions)}")
for prediction in predictions:
    print(f"\nID: {prediction['id']}")
    print(f"Disease: " f"{prediction['predicted_class']}")
    print(f"Confidence: " f"{prediction['confidence'] * 100:.2f}%")
    print(f"Date: " f"{prediction['created_at']}")
print("\nDatabase test completed successfully.")