import sqlite3
import os
from datetime import datetime

DATABASE_DIR = "database"
DATABASE_PATH = os.path.join(DATABASE_DIR, "predictions.db")

def get_connection():
    #Create a connection to the SQLite database
    os.makedirs(DATABASE_DIR, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    #Create the predictions table and add missing columns if needed
    connection = get_connection()
    cursor = connection.cursor()

    #Create the original table if it does not exist.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            image_filename TEXT,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            second_class TEXT,
            second_confidence REAL,
            third_class TEXT,
            third_confidence REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    #Safe migration for an existing database
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [row["name"] for row in cursor.fetchall()]
    if "image_filename" not in columns:
        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN image_filename TEXT
            """
        )
    connection.commit()
    connection.close()


def save_prediction(filename, image_filename, prediction_result):
    #Save a prediction result and its stored image filename
    top_predictions = prediction_result["top_predictions"]
    second = (top_predictions[1] if len(top_predictions) > 1 else None)
    third = (top_predictions[2] if len(top_predictions) > 2 else None)

    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (filename, image_filename, predicted_class, confidence, second_class, second_confidence, third_class, third_confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (filename, image_filename, prediction_result["predicted_class"], prediction_result["confidence"],
         second["class_name"] if second else None, second["confidence"] if second else None, third["class_name"] if third else None,
         third["confidence"] if third else None, datetime.now().isoformat(timespec="seconds"))
        )
    prediction_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return prediction_id


def get_predictions(limit=50):
    #Return the most recent predictions
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(""" SELECT * FROM predictions ORDER BY id DESC LIMIT ? """, (limit,))
    predictions = cursor.fetchall()
    connection.close()
    return predictions


def get_prediction(prediction_id):
    #Return a single prediction by ID
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(""" SELECT * FROM predictions WHERE id = ? """, (prediction_id,))
    prediction = cursor.fetchone()
    connection.close()
    return prediction


def delete_prediction(prediction_id):
    #Delete a prediction from history
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(""" DELETE FROM predictions WHERE id = ? """, (prediction_id,))
    connection.commit()
    deleted = cursor.rowcount > 0
    connection.close()
    return deleted