import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report
import pathlib

dataset_path = r"E:\ss\dataset"
dataset_dir = pathlib.Path(dataset_path)

img_height = 180
img_width = 180
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))  
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

epochs = 5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

val_images_all = []
val_labels_all = []

for images, labels in val_ds:
    val_images_all.append(images)
    val_labels_all.append(labels)

val_images_all = tf.concat(val_images_all, axis=0)
val_labels_all = tf.concat(val_labels_all, axis=0).numpy()

predictions = model.predict(val_images_all)
predicted_labels = np.argmax(predictions, axis=1)

for i in range(5):
    plt.imshow(val_images_all[i].numpy().astype("uint8"))
    actual = class_names[val_labels_all[i]]
    predicted = class_names[predicted_labels[i]]
    plt.title(f"Actual: {actual}, Predicted: {predicted}")
    plt.axis("off")
    plt.show()

print("\nClassification Report:\n")
print(classification_report(val_labels_all, predicted_labels, target_names=class_names))
