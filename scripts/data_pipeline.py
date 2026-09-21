import pandas as pd
import tensorflow as tf

#Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
RANDOM_SEED = 42

#Image loading
def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label

#Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05)], 
    name="data_augmentation")

#Create dataset
def create_dataset(dataframe, training=False):
    paths = dataframe["image_path"].values
    labels = dataframe["label"].values
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=len(dataframe), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)

    if training:
        dataset = dataset.map(lambda images, labels: (data_augmentation(images, training=True), labels), num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

#Load all datasets
def load_datasets():
    train_data = pd.read_csv("reports/results/train.csv")
    validation_data = pd.read_csv("reports/results/validation.csv")
    test_data = pd.read_csv("reports/results/test.csv")
    train_dataset = create_dataset(train_data, training=True)
    validation_dataset = create_dataset(validation_data, training=False)
    test_dataset = create_dataset(test_data, training=False)
    return (train_dataset, validation_dataset, test_dataset)