import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import Input
import matplotlib.pyplot as plt

# Parameters
img_size = (128, 128)
batch_size = 32

# Load CSV
csv_path = "labels.csv"
df = pd.read_csv(csv_path)

file_paths = df["plot_path"].values
labels = df["label"].values

# Split data
train_paths, test_paths, train_labels, test_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42
)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.25, random_state=42
)

# Preprocess function
def preprocess_image(filepath, label):
    img = load_img(filepath.numpy().decode('utf-8'), target_size=img_size)
    img_array = img_to_array(img) / 255.0
    return img_array, label

def preprocess_image_wrapper(filepath, label):
    img, lbl = tf.py_function(
        func=preprocess_image,
        inp=[filepath, label],
        Tout=[tf.float32, tf.int64],  # Keep dtype as int64 here for compatibility
    )
    img.set_shape((img_size[0], img_size[1], 3))
    lbl.set_shape(())  # Label is scalar
    lbl = tf.cast(lbl, tf.int32)  # Explicitly cast to int32
    return img, lbl

def create_dataset(file_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(preprocess_image_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_paths, train_labels)
val_dataset = create_dataset(val_paths, val_labels)
test_dataset = create_dataset(test_paths, test_labels)

# # Verify dataset shapes
# for img, lbl in train_dataset.take(1):
#     print(f"Image batch shape: {img.shape}")
#     print(f"Label batch shape: {lbl.shape}")

#build 
model = models.Sequential([
    Input(shape=(img_size[0], img_size[1], 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

#compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10, #adjust
    verbose=1
)
model.save_weights("model.weights.h5")
model.save("model.h5")