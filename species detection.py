import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Load pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

def load_image(img_path):
    """Load and preprocess image for MobileNetV2."""
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img  # Return both for display

def detect_species(img_path):
    """Predict the top 3 labels for a single image and display it."""
    if not os.path.exists(img_path):
        print(f"Error: Image '{img_path}' does not exist.")
        return

    img_array, img = load_image(img_path)
    preds = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predictions for {os.path.basename(img_path)}")
    plt.show()

    print(f"\nPredictions for {img_path}:")
    for (_, label, score) in decoded:
        print(f"{label} --> {score*100:.2f}%")

# -------------------------
# Ask user for image path
# -------------------------
image_path = input("Enter the full path of the image: ").strip()
detect_species(image_path)
