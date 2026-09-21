import os
import json
import tensorflow as tf
from data_pipeline import load_datasets

#Configuration
NUM_CLASSES = 10
EPOCHS = 10
FINE_TUNE_LAYERS = 30
INPUT_MODEL = "models/checkpoints/best_stage1.keras"
BEST_MODEL_PATH = "models/checkpoints/best_stage2.keras"
FINAL_MODEL_PATH = "models/saved_models/tomato_disease_stage2.keras"
HISTORY_PATH = "reports/results/training_history_stage2.csv"
CLASS_WEIGHTS_PATH = "reports/results/class_weights.json"

#Load data
print("=" * 70)
print("SMART AGRICULTURE AI - STAGE 2 FINE-TUNING")
print("=" * 70)
print("\nLoading datasets...")
train_dataset, validation_dataset, test_dataset = load_datasets()
print("Datasets loaded successfully.")

#Load class weights
with open(CLASS_WEIGHTS_PATH, "r") as file:
    class_weights = json.load(file)
class_weights = {int(key): float(value) for key, value in class_weights.items()}
print("\nClass weights loaded.")

#Load stage1 model
print("\nLoading best Stage 1 model...")
model = tf.keras.models.load_model(INPUT_MODEL)
print("Stage 1 model loaded successfully.")

#Find efficientnet base model
base_model = model.get_layer("efficientnetb0")
print("\nEfficientNetB0 found.")
print("Total EfficientNet layers:", len(base_model.layers))

#Fine-tuning setup
base_model.trainable = True  #Freeze the entire base model first
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False   #Freeze all layers except the last fine-tune layers

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False   #Keep batch normalization layers frozen to maintain the pretrained statistics during fine-tuning

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])  #Recompile model


#Display trainable parameters
trainable_params = sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights)
non_trainable_params = sum(tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights)

print("\n" + "=" * 70)
print("FINE-TUNING CONFIGURATION")
print("=" * 70)
print(f"Fine-tuning layers: {FINE_TUNE_LAYERS}")
print("Learning rate: 0.00001")
print("Batch Normalization layers: Frozen")
print(f"\nTrainable parameters:     {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")
print(f"Total parameters:         {model.count_params():,}")

#Callbacks
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("models/saved_models", exist_ok=True)
os.makedirs("reports/results", exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2,min_lr=1e-7, verbose=1)
csv_logger = tf.keras.callbacks.CSVLogger(HISTORY_PATH, append=False)

#Train
print("\n" + "=" * 70)
print("STARTING STAGE 2 FINE-TUNING")
print("=" * 70)
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, class_weight=class_weights, callbacks=[checkpoint, early_stopping, reduce_lr, csv_logger], verbose=1)
model.save(FINAL_MODEL_PATH)  #Save final mode

#Training summary
best_val_accuracy = max(history.history["val_accuracy"])
best_val_loss = min(history.history["val_loss"])
print("\n" + "=" * 70)
print("STAGE 2 FINE-TUNING COMPLETED")
print("=" * 70)

print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
print(f"Best validation loss:     {best_val_loss:.4f}")

print("\nSaved files:")

print(f"Best checkpoint:")
print(BEST_MODEL_PATH)

print(f"\nFinal model:")
print(FINAL_MODEL_PATH)

print(f"\nTraining history:")
print(HISTORY_PATH)

print("\nNext step:")
print("Evaluate Stage 2 on the test dataset.")