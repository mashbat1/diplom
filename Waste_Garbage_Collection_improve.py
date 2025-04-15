# 📦 All-in-One EfficientNetB0 Fine-Tuning Pipeline (Improved)
import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns

# ✅ Seed

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything()

# ✅ Config
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
EPOCHS_TL = 10
EPOCHS_FT = 10
LR_TL = 1e-3
LR_FT = 1e-5
TRAIN_DIR = "./datasets/Train"
TEST_DIR = "./datasets/Test"
AUTOTUNE = tf.data.AUTOTUNE

# ✅ Balance dataset by copying

def balance_dataset_by_copying(data_dir, target_count=400):
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        images = os.listdir(class_path)
        current_count = len(images)
        if current_count >= target_count:
            continue
        for i in range(target_count - current_count):
            img_to_copy = random.choice(images)
            shutil.copy(os.path.join(class_path, img_to_copy), os.path.join(class_path, f"copy_{i}_{img_to_copy}"))

balance_dataset_by_copying(TRAIN_DIR, 400)

# ✅ Load Dataset

def load_dataset(directory, validation_split=0.2):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset="training",
        label_mode="categorical",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        seed=123
    ), tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset="validation",
        label_mode="categorical",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        seed=123
    )

train_ds, val_ds = load_dataset(TRAIN_DIR)
class_names = train_ds.class_names
n_classes = len(class_names)

train_ds = train_ds.map(lambda x, y: (tf.keras.applications.efficientnet.preprocess_input(x), y)).cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (tf.keras.applications.efficientnet.preprocess_input(x), y)).cache().prefetch(AUTOTUNE)

# ✅ Augmentation (only for training)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
])

# ✅ Model Builder

def build_model(trainable=False):
    base_model = EfficientNetB0(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet")
    base_model.trainable = trainable
    model = Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax")
    ])
    return model

# ✅ Phase 1: Transfer Learning
model = build_model(trainable=False)
model.compile(optimizer=Adam(learning_rate=LR_TL), loss="categorical_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint("efficientnet_best.weights.h5", save_best_only=True, save_weights_only=True, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

history_tl = model.fit(train_ds, epochs=EPOCHS_TL, validation_data=val_ds, callbacks=[checkpoint, early_stop, reduce_lr])

# ✅ Phase 2: Fine-Tuning
model = build_model(trainable=True)
model.load_weights("efficientnet_best.weights.h5")
model.compile(optimizer=Adam(learning_rate=LR_FT), loss="categorical_crossentropy", metrics=["accuracy"])

all_labels = [np.argmax(label.numpy()) for _, label in train_ds.unbatch()]
class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(all_labels), y=all_labels)
class_weights_dict = dict(enumerate(class_weights))

history_ft = model.fit(train_ds, epochs=EPOCHS_FT, validation_data=val_ds, callbacks=[checkpoint, early_stop, reduce_lr], class_weight=class_weights_dict)

# ✅ Save final model
model.load_weights("efficientnet_best.weights.h5")
model.save("efficientnet_balanced_finetuned.h5")

# ✅ Evaluate

test_ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR, label_mode="categorical", image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=32, seed=123)
test_ds = test_ds.map(lambda x, y: (tf.keras.applications.efficientnet.preprocess_input(x), y)).cache().prefetch(AUTOTUNE)

test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\n🎯 Test Accuracy: {test_acc:.2%}")

# ✅ Plot Training History

def plot_combined_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_combined_history(history_tl, history_ft)

# ✅ Confusion Matrix

def evaluate_confusion_matrix(model, dataset, class_names):
    y_true, y_pred = [], []
    for images, labels in dataset:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    print("\n📋 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

evaluate_confusion_matrix(model, test_ds, class_names)

# ✅ Class Distribution

def plot_class_distribution(data_dir):
    class_counts = {class_name: len(os.listdir(os.path.join(data_dir, class_name))) for class_name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, class_name))}
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Training Image Count per Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_class_distribution(TRAIN_DIR)
