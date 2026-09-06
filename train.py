import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 9
SEED = 42

BASE_DIR = os.path.join('dataset', 'Skin cancer ISIC The International Skin Imaging Collaboration')
TRAIN_DIR = os.path.join(BASE_DIR, 'Train')
TEST_DIR = os.path.join(BASE_DIR, 'Test')

# 1. Generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.15,
    validation_split=0.2
)

val_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_gen = val_test_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(train_gen.class_indices.keys())

# Class Weights
train_labels = train_gen.classes
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(weights))

# 2. Model Architecture
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Crucial: Batch Normalization freeze karo taaki statistics corrupt na hon
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ==========================================
# STABLE PHASE 1 (Feature Extraction with Checkpoint)
# ==========================================
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p1 = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    ModelCheckpoint('skin_cancer_best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

print("\n--- Training Frozen Base Model (12 Epochs) ---", flush=True)
model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen,
    validation_steps=len(val_gen),
    epochs=12,
    class_weight=class_weights,
    callbacks=callbacks_p1,
    verbose=1
)

# ==========================================
# CONTROLLED FINE-TUNING (Keeping BatchNorm Frozen)
# ==========================================
print("\n--- Fine-Tuning with BatchNorm Frozen ---", flush=True)
base_model.trainable = True

# Sirf aakhri 20 layers unfreeze karo aur BatchNorm layers ko STRICTLY frozen rakho
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    elif layer not in base_model.layers[-20:]:
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p2 = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint('skin_cancer_best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=1e-7, verbose=1)
]

model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen,
    validation_steps=len(val_gen),
    epochs=6,
    class_weight=class_weights,
    callbacks=callbacks_p2,
    verbose=1
)

# ==========================================
# EVALUATION USING BEST SAVED WEIGHTS
# ==========================================
print("\n--- Loading Best Model for Evaluation ---", flush=True)
best_model = tf.keras.models.load_model('skin_cancer_best_model.keras')

test_gen.reset()
Y_pred = best_model.predict(test_gen, steps=len(test_gen), verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_gen.classes

print("\n" + "="*55)
print("              CLASSIFICATION REPORT")
print("="*55)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Final Stable Confusion Matrix', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

print("\nConfusion Matrix image saved as 'confusion_matrix.png'", flush=True)
print("Final model saved as 'skin_cancer_best_model.keras'", flush=True)