import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Verzeichnispfad zu Ihren Datensätzen
train_dir = "Desktop/Projekt_ML/traffic_light_data/train"
val_dir = "Desktop/Projekt_ML/traffic_light_data/val"

# Bildgröße
img_height, img_width = 64, 64

# Erstellen der Trainingsdatensätze und Anwendungen der Transformationen (Beispielhaft)
train_ds = keras.preprocessing.image_dataset_from_directory(
  train_dir,
  image_size=(img_height, img_width),
  batch_size=64)

val_ds = keras.preprocessing.image_dataset_from_directory(
  val_dir,
  image_size=(img_height, img_width),
  batch_size=64)

class_names = train_ds.class_names

# CNN-Modelldefinition
model = keras.Sequential([
  layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
  layers.BatchNormalization(),
  layers.MaxPooling2D(),
  layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
  layers.BatchNormalization(),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(64, activation='relu'),
  layers.Dense(len(class_names), activation='softmax')
])

# Konfiguration des Modells zum Trainieren
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Trainieren des Modells
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=50
)

# Anzeigen der Leistungskurve
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Beispielsbildvergleich

test_image_path = "Desktop/Projekt_ML/traffic_light_data/val/red/red_71.jpg"
test_image_original = Image.open(test_image_path)
test_image = test_image_original.resize((img_height, img_width))
test_image = np.array(test_image)/255.0
test_image = np.expand_dims(test_image, axis=0)

predictions = model.predict(test_image)
predicted_class = class_names[np.argmax(predictions[0])]

print(f"Predicted Class: {predicted_class}")
print(f"Class Probabilities: {predictions[0]}")