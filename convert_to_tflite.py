import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# Custom DepthwiseConv2D to ignore 'groups'
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove 'groups' if present
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

# Load the model with custom objects
model = load_model('Model/keras_model.h5', custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the new TFLite model
with open('Model/model_new.tflite', 'wb') as f:
    f.write(tflite_model)

print("Converted keras_model.h5 to model_new.tflite successfully!")