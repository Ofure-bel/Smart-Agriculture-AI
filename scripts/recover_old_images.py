import os
import sqlite3
from datetime import datetime

DATABASE_PATH = os.path.join("database", "predictions.db")
UPLOAD_FOLDER = os.path.join("static", "uploads")

def get_database_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection

def main():
    print()
    print("=" * 60)
    print("Recovering old prediction images")
    print("=" * 60)
    print()

    #Get uploaded image files
    files = []
    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.isfile(filepath):
            continue

        extension = os.path.splitext(filename)[1].lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue

        modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        files.append({"filename": filename, "filepath": filepath, "time": modified_time})

    print(f"Found {len(files)} uploaded image files.")
    print()

    #Get predictions without image filenames
    connection = get_database_connection()
    predictions = connection.execute("""SELECT id, filename, image_filename, created_at, predicted_class FROM predictions WHERE image_filename IS NULL ORDER BY id """).fetchall()
    print(f"Found {len(predictions)} predictions without images.")
    print()

    #Match predictions to files
    used_files = set()
    for prediction in predictions:
        prediction_time = datetime.fromisoformat(prediction["created_at"])
        candidates = []
        for file in files:
            if file["filename"] in used_files:
                continue
            difference = abs((file["time"] - prediction_time).total_seconds())
            candidates.append((difference, file))

        if not candidates:
            print(f"ID {prediction['id']}: " "No available image files.")
            continue
        candidates.sort(key=lambda item: item[0])
        difference, best_file = candidates[0]

        #Only accept a match within 10 seconds.
        if difference <= 10:
            connection.execute("""UPDATE predictions SET image_filename = ? WHERE id = ? """, (best_file["filename"], prediction["id"]))
            used_files.add(best_file["filename"])

            print(f"ID {prediction['id']}: " f"matched {best_file['filename']} " f"({difference:.1f}s difference)")
        else:
            print(f"ID {prediction['id']}: " f"no safe match " f"(closest was {difference:.1f}s away)")
            
    connection.commit()
    connection.close()

    print()
    print("=" * 60)
    print("Recovery complete.")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()