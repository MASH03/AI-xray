# =============================================================
# PNEUMONIA X-RAY DETECTION - TRAINING SCRIPT (Google Colab)
# =============================================================
# 
# HOW TO USE:
# 1. Go to https://colab.research.google.com/
# 2. Create a new notebook
# 3. Upload your dataset to Google Drive OR use the Kaggle API
# 4. Copy-paste each section into separate cells and run them
#
# =============================================================

# ---- CELL 1: Install Dependencies ----
# !pip install tensorflowjs

# ---- CELL 2: Mount Google Drive (if your dataset is on Drive) ----
# from google.colab import drive
# drive.mount('/content/drive')

# ---- CELL 3: Download Dataset from Kaggle (Alternative) ----
# !pip install kaggle
# !mkdir -p ~/.kaggle
# Upload your kaggle.json API key, then:
# !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
# !unzip chest-xray-pneumonia.zip -d /content/dataset

# ---- CELL 4: Import Libraries ----
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# ---- CELL 5: Configuration ----
# CHANGE THIS to your actual dataset path
DATASET_PATH = "/content/dataset/pneumonia_dataset"  # Update this!
TRAIN_IMAGES_DIR = os.path.join(DATASET_PATH, "train_images", "train_images")
TEST_IMAGES_DIR = os.path.join(DATASET_PATH, "test_images", "test_images")
LABELS_CSV = os.path.join(DATASET_PATH, "labels_train.csv")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Class mapping
CLASS_NAMES = {0: "Normal", 1: "Bacterial Pneumonia", 2: "Viral Pneumonia"}

# ---- CELL 6: Load and Explore Labels ----
labels_df = pd.read_csv(LABELS_CSV)
print(f"Total training samples: {len(labels_df)}")
print(f"\nClass distribution:")
print(labels_df['class_id'].value_counts().rename(CLASS_NAMES))

# Map class_id to string labels for the generator
labels_df['label'] = labels_df['class_id'].map(CLASS_NAMES)

# Verify images exist
sample_file = labels_df.iloc[0]['file_name']
sample_path = os.path.join(TRAIN_IMAGES_DIR, sample_file)
print(f"\nSample image path: {sample_path}")
print(f"File exists: {os.path.exists(sample_path)}")

# ---- CELL 7: Visualize Sample X-Rays ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, class_id in enumerate([0, 1, 2]):
    sample = labels_df[labels_df['class_id'] == class_id].iloc[0]
    img_path = os.path.join(TRAIN_IMAGES_DIR, sample['file_name'])
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    axes[idx].imshow(img, cmap='gray')
    axes[idx].set_title(CLASS_NAMES[class_id], fontsize=14, fontweight='bold')
    axes[idx].axis('off')
plt.suptitle("Sample X-Ray Images", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ---- CELL 8: Split into Train/Validation ----
train_df, val_df = train_test_split(
    labels_df, test_size=0.2, random_state=42, stratify=labels_df['class_id']
)
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# ---- CELL 9: Create Data Generators ----
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_IMAGES_DIR,
    x_col='file_name',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=list(CLASS_NAMES.values()),
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=TRAIN_IMAGES_DIR,
    x_col='file_name',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=list(CLASS_NAMES.values()),
    shuffle=False
)

# ---- CELL 10: Build the Model (Transfer Learning with MobileNetV2) ----
# MobileNetV2 is lightweight and perfect for web deployment
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model initially
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')  # 3 classes
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---- CELL 11: Train the Model ----
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# ---- CELL 12: Fine-tune (Unfreeze top layers) ----
base_model.trainable = True
# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks
)

# ---- CELL 13: Plot Training Results ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history_fine.history['accuracy'], label='Train Accuracy')
ax1.plot(history_fine.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history_fine.history['loss'], label='Train Loss')
ax2.plot(history_fine.history['val_loss'], label='Val Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.show()

# ---- CELL 14: Evaluate on Validation Set ----
val_loss, val_acc = model.evaluate(val_generator)
print(f"\nValidation Accuracy: {val_acc*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# ---- CELL 15: Save Model ----
# Save as Keras model
model.save('/content/pneumonia_model.h5')
print("Saved as pneumonia_model.h5")

# Save as TensorFlow SavedModel format
model.save('/content/pneumonia_saved_model')
print("Saved as SavedModel format")

# ---- CELL 16: Convert to TensorFlow.js ----
import tensorflowjs as tfjs

tfjs.converters.save_keras_model(model, '/content/tfjs_model')
print("\nTensorFlow.js model saved to /content/tfjs_model/")
print("Download the 'tfjs_model' folder and place it in your web app!")

# ---- CELL 17: Download the model files ----
# Run this to zip and download
import shutil
shutil.make_archive('/content/tfjs_model_download', 'zip', '/content/tfjs_model')

from google.colab import files  # type: ignore  # Only works in Google Colab
files.download('/content/tfjs_model_download.zip')
print("Download started! Extract into your web app's 'model/' folder.")
