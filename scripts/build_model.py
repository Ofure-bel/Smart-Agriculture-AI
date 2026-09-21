import os
import tensorflow as tf

#Configuration
NUM_CLASSES = 10
MODEL_SAVE_PATH = ("models/saved_models/" "efficientnetb0_base.keras")

#Build model function
def build_model():
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

    base_model.trainable = False  #Freeze pretrained layers
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tomato_disease_efficientnetb0")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

#Test model when ran directly
if __name__ == "__main__":

    print("=" * 60)
    print("SMART AGRICULTURE AI - MODEL DEVELOPMENT")
    print("=" * 60)
    print("\nBuilding EfficientNetB0...")
    model = build_model()
    model.summary()
    os.makedirs("models/saved_models", exist_ok=True)
    model.save(MODEL_SAVE_PATH)

    print("\n" + "=" * 60)
    print("MODEL CREATED SUCCESSFULLY!")
    print("=" * 60)

    print("\nSaved to:")
    print(MODEL_SAVE_PATH)

    print("\nModel input shape:")
    print(model.input_shape)

    print("\nModel output shape:")
    print(model.output_shape)

    trainable_params = sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights)

    print("\nTrainable parameters:", trainable_params)
    print("Non-trainable parameters:", non_trainable_params)
    print("Total parameters:", trainable_params + non_trainable_params)