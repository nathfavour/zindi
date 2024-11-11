# model.py
import tensorflow as tf
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split

class RunwayDetector:
    def __init__(self):
        self.optical_model = self._build_optical_model()
        self.sar_model = self._build_sar_model()
        
    def _build_optical_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
    def _build_sar_model(self):
        # Similar architecture but with single channel input
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def preprocess_image(self, optical_path, sar_path):
        # Read and normalize satellite imagery
        with rasterio.open(optical_path) as src:
            optical = src.read()
        with rasterio.open(sar_path) as src:
            sar = src.read(1)
        
        # Normalize
        optical = optical / 255.0
        sar = (sar - sar.mean()) / sar.std()
        
        return optical.transpose(1,2,0), sar

    def train(self, optical_images, sar_images, labels):
        # Train both models separately
        self.optical_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.sar_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        self.optical_model.fit(optical_images, labels, epochs=10, validation_split=0.2)
        self.sar_model.fit(sar_images, labels, epochs=10, validation_split=0.2)

    def predict(self, optical_image, sar_image):
        # Ensemble prediction
        optical_pred = self.optical_model.predict(optical_image[np.newaxis, ...])
        sar_pred = self.sar_model.predict(sar_image[np.newaxis, ...])
        
        # Average predictions
        return (optical_pred + sar_pred) / 2