import cv2
import numpy as np
import tensorflow as tf

LAST_CONV_LAYER = "top_conv"

def generate_gradcam(model, image_array, class_index):
    #Locate EfficientNet backbone first for the Grad-CAM heatmap for predictions
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if "efficientnet" in layer.name.lower():
                base_model = layer
                break
    if base_model is None:
        raise ValueError("EfficientNet backbone not found.")
    last_conv_layer = base_model.get_layer(LAST_CONV_LAYER)
    conv_model = tf.keras.Model(inputs=base_model.input, outputs=last_conv_layer.output)  #Model that outputs feature maps

    #Classifier after the backbone
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    found = False
    for layer in model.layers:
        if layer.name == base_model.name:
            found = True
            continue
        if found:
            x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        feature_maps = conv_model(image_array)
        tape.watch(feature_maps)
        predictions = classifier_model(feature_maps)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, feature_maps)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    feature_maps = feature_maps[0]
    heatmap = tf.reduce_sum(feature_maps * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap /= max_value
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224, 224))
    return heatmap


def save_gradcam(image_path, heatmap, output_path):
    #Overlay a Grad-CAM heatmap onto the original image.
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, overlay)
    return output_path