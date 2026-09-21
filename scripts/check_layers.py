from model.predictor import predictor
import tensorflow as tf

base_model = predictor.model.get_layer("efficientnetb0")
print("\nEfficientNetB0 Layers:\n")
for layer in base_model.layers:
    print(layer.name, "-", layer.__class__.__name__)