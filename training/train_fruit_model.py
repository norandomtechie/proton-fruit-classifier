#!/usr/bin/env python3
"""
Train a small fruit classification CNN and export as quantized TFLite model + C array.

Target: RP2350B (520 KB SRAM, 16 MB flash) with OV7670 camera (160x120 YUY2).
Model input: 100x100 RGB INT8 (3 channels).
Classes: apple, banana, lime, blueberry (4 classes).

Usage:
    pip install tensorflow numpy Pillow kagglehub scikit-learn matplotlib
    python train_fruit_model.py

The script will:
1. Download Fruits-360 dataset from Kaggle via kagglehub
2. Train a CNN with data augmentation
3. Quantize to INT8
4. Export as C header file (include/fruit_model.h)
"""

import os
import sys
import json
import numpy as np

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Configuration ---
IMG_SIZE = 100          # Model input: 100x100
NUM_CLASSES = 4        # apple, banana, lime, blueberry
BATCH_SIZE = 32
EPOCHS = 50
MODEL_PATH = "fruit_model.tflite"
HEADER_PATH = os.path.join(os.path.dirname(__file__), "include", "fruit_model.h")
ROC_PLOT_PATH = os.path.join(os.path.dirname(__file__), "roc_curves.png")
ROC_METRICS_PATH = os.path.join(os.path.dirname(__file__), "roc_auc_metrics.json")

CLASS_NAMES = ["apple", "banana", "lime", "blueberry"]

# Fruits-360 class folder name prefixes (case-insensitive startswith match)
FRUIT360_PREFIXES = {
    # Red apples only (exclude green/golden/pink/yellow variants)
    "apple": [
        "apple red 1",
        "apple red 2",
        "apple red 3",
        "apple red delicious 1",
        "apple_red_1",
        "apple_red_2",
        "apple_red_3",
        "apple_red_delicious_1",
        "apple_red_delicios_1",
    ],
    "banana": ["banana"],
    "lime": ["lime"],
    "blueberry": ["blueberry"],
}


def build_model():
    """Build a CNN suitable for MCU deployment with more capacity."""
    model = keras.Sequential([
        # Input: 100x100x3 RGB
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Block 1: 24 filters
        layers.Conv2D(24, 3, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),  # -> 32x32

        # Block 2: 48 filters
        layers.Conv2D(48, 3, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),  # -> 16x16

        # Block 3: 64 filters
        layers.Conv2D(64, 3, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),  # -> 8x8

        # Block 4: 96 filters
        layers.Conv2D(96, 3, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),  # -> 96

        # Classifier — no softmax; use logits for better INT8 quantization
        layers.Dense(NUM_CLASSES),
    ])
    return model


def download_fruits360():
    """Download Fruits-360 dataset via kagglehub."""
    import kagglehub
    print("Downloading Fruits-360 dataset via kagglehub...")
    path = kagglehub.dataset_download("moltean/fruits")
    print(f"  Dataset downloaded to: {path}")
    return path


def find_training_dir(base_dir):
    """Find the Training directory within the Fruits-360 dataset."""
    # Prefer the 100x100 split from Fruits-360. It has richer class coverage
    # (including Limes/Blueberry) and already matches our configured input size.
    preferred = os.path.join(base_dir, "fruits-360_100x100", "fruits-360", "Training")
    if os.path.isdir(preferred):
        return preferred

    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == "Training":
            return root
    # Fallback: look for fruit class folders directly
    return base_dir


def load_fruits360(data_dir):
    """Load Fruits-360 dataset, mapping to our 4 classes."""
    images = []
    labels = []

    train_dir = find_training_dir(data_dir)
    print(f"  Using training dir: {train_dir}")

    # List available folders for debugging
    if os.path.isdir(train_dir):
        available = sorted(os.listdir(train_dir))
        print(f"  Available folders ({len(available)}): {available[:10]}...")

    for class_idx, (class_name, prefixes) in enumerate(FRUIT360_PREFIXES.items()):
        # Find all folders matching any prefix for this class
        matching_folders = []
        for folder_name in sorted(os.listdir(train_dir)):
            folder_lower = folder_name.lower().replace('_', ' ')
            if any(folder_lower.startswith(p) for p in prefixes):
                matching_folders.append(folder_name)
        print(f"  {class_name}: found {len(matching_folders)} folders: {matching_folders}")

        for folder in matching_folders:
            folder_path = os.path.join(train_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                try:
                    img = tf.io.read_file(fpath)
                    img = tf.image.decode_image(img, channels=3)
                    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
                    images.append(img.numpy())
                    labels.append(class_idx)
                except Exception:
                    continue
        print(f"  Loaded {class_name}: {sum(1 for l in labels if l == class_idx)} images")

    if len(images) == 0:
        print("ERROR: No fruit images loaded! Check dataset path.")
        sys.exit(1)

    # Balance classes by capping the largest class
    max_per_class = 2000
    images_arr = images
    labels_arr = labels
    balanced_images = []
    balanced_labels = []
    for cls_idx in range(len(FRUIT360_PREFIXES)):
        cls_imgs = [img for img, lbl in zip(images_arr, labels_arr) if lbl == cls_idx]
        if len(cls_imgs) > max_per_class:
            indices = np.random.choice(len(cls_imgs), max_per_class, replace=False)
            cls_imgs = [cls_imgs[i] for i in indices]
        balanced_images.extend(cls_imgs)
        balanced_labels.extend([cls_idx] * len(cls_imgs))
    images = balanced_images
    labels = balanced_labels
    print(f"  After balancing (max {max_per_class}/class): {len(images)} fruit images")

    # Validate all configured classes are represented.
    class_counts = {name: sum(1 for l in labels if l == idx) for idx, name in enumerate(CLASS_NAMES)}
    missing = [name for name, count in class_counts.items() if count == 0]
    if missing:
        print(f"ERROR: Missing images for classes: {', '.join(missing)}")
        print("Check FRUIT360_PREFIXES and dataset contents.")
        sys.exit(1)

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def augment(image, label):
    """Apply aggressive augmentations to bridge Fruits-360 → real camera domain gap."""
    # --- Background replacement ---
    # Fruits-360 has white backgrounds (~1.0 after /255); replace with random values
    # so the model can't rely on background color.
    # For RGB: white = all channels bright
    bg_mask = tf.cast(tf.reduce_min(image, axis=-1, keepdims=True) > 0.92, tf.float32)
    bg_val = tf.random.uniform([], 0.0, 0.7)
    bg_noise = tf.random.normal(tf.shape(image), stddev=0.08)
    random_bg = tf.clip_by_value(bg_val + bg_noise, 0.0, 1.0)
    image = image * (1.0 - bg_mask) + random_bg * bg_mask

    # --- Random scale/position ---
    # Pad the image then resize back to simulate fruit at different distances.
    # This teaches the model that fruit can be small in the frame.
    pad = tf.random.uniform([], 0, 20, dtype=tf.int32)
    bg_pad_val = tf.random.uniform([], 0.0, 0.5)
    padded = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]],
                    constant_values=bg_pad_val)
    image = tf.image.resize(padded, [IMG_SIZE, IMG_SIZE])

    # --- Standard augmentations ---
    image = tf.image.random_brightness(image, 0.4)
    image = tf.image.random_contrast(image, 0.5, 1.5)
    image = tf.image.random_saturation(image, 0.5, 1.5)
    image = tf.image.random_hue(image, 0.08)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # --- Gaussian noise (simulate OV7670 sensor noise) ---
    noise = tf.random.normal(tf.shape(image), stddev=0.06)
    image = image + noise

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def quantize_model(model, representative_data):
    """Convert to fully INT8 quantized TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(min(200, len(representative_data))):
            sample = representative_data[i:i+1].astype(np.float32)
            yield [sample]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    return tflite_model


def compute_and_save_roc(model, val_x, val_y):
    """Compute one-vs-rest ROC/AUC and save plot + JSON metrics."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt

    # Model outputs logits; softmax converts to per-class probabilities for ROC.
    logits = model.predict(val_x, batch_size=BATCH_SIZE, verbose=0)
    probs = tf.nn.softmax(logits, axis=1).numpy()

    y_bin = label_binarize(val_y, classes=list(range(NUM_CLASSES)))
    auc_by_class = {}

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        class_auc = auc(fpr, tpr)
        auc_by_class[class_name] = float(class_auc)
        plt.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC={class_auc:.3f})")

    # Random-chance reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH, dpi=160)
    plt.close()

    with open(ROC_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(auc_by_class, f, indent=2)

    print(f"ROC plot saved to {ROC_PLOT_PATH}")
    print(f"ROC AUC metrics saved to {ROC_METRICS_PATH}")
    for class_name in CLASS_NAMES:
        print(f"  AUC[{class_name}]: {auc_by_class[class_name]:.4f}")

    return auc_by_class


def export_c_header(tflite_model, header_path):
    """Export TFLite model as a C header file."""
    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    with open(header_path, 'w') as f:
        f.write("#ifndef FRUIT_MODEL_H_\n")
        f.write("#define FRUIT_MODEL_H_\n\n")
        f.write(f"// Auto-generated by train_fruit_model.py\n")
        f.write(f"// Model input: {IMG_SIZE}x{IMG_SIZE} RGB INT8 (3 channels)\n")
        f.write(f"// Classes: {', '.join(CLASS_NAMES)}\n")
        f.write(f"// Model size: {len(tflite_model)} bytes\n\n")
        f.write(f"#define FRUIT_MODEL_INPUT_SIZE {IMG_SIZE}\n")
        f.write(f"#define FRUIT_NUM_CLASSES {NUM_CLASSES}\n\n")

        # Class names
        f.write("static const char* const fruit_class_names[] = {\n")
        for name in CLASS_NAMES:
            f.write(f'    "{name}",\n')
        f.write("};\n\n")

        # Model data
        f.write(f"alignas(16) static const unsigned char fruit_model_data[] = {{\n")
        for i, byte in enumerate(tflite_model):
            if i % 16 == 0:
                f.write("    ")
            f.write(f"0x{byte:02x},")
            if i % 16 == 15 or i == len(tflite_model) - 1:
                f.write("\n")
            else:
                f.write(" ")
        f.write("};\n\n")

        f.write(f"static const unsigned int fruit_model_data_len = {len(tflite_model)};\n\n")
        f.write("#endif  // FRUIT_MODEL_H_\n")

    print(f"C header written to {header_path} ({len(tflite_model)} bytes)")


def main():
    print(f"=== Fruit Classification Model Training ===")
    print(f"Input: {IMG_SIZE}x{IMG_SIZE} RGB, {NUM_CLASSES} classes")
    print()

    # Download Fruits-360 via kagglehub or use local path
    fruits360_dir = os.environ.get("FRUITS360_DIR", "")
    if fruits360_dir and os.path.isdir(fruits360_dir):
        print(f"Using local Fruits-360 at {fruits360_dir}")
    else:
        fruits360_dir = download_fruits360()

    print(f"Loading Fruits-360...")
    images, labels = load_fruits360(fruits360_dir)

    print(f"\nDataset: {len(images)} images")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(labels == i)
        print(f"  {name}: {count}")

    # Shuffle and split
    indices = np.random.permutation(len(images))
    images, labels = images[indices], labels[indices]
    split = int(0.8 * len(images))
    train_x, val_x = images[:split], images[split:]
    train_y, val_y = labels[:split], labels[split:]

    print(f"Train: {len(train_x)}, Val: {len(val_x)}")
    print()

    # Build augmented training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_ds = train_ds.shuffle(len(train_x)).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Build and train
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    model.summary()
    print()

    # Compute class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_arr = compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
    class_weight = {i: w for i, w in enumerate(class_weights_arr)}
    print(f"Class weights: {class_weight}")

    # LR schedule: reduce on plateau
    lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[lr_callback],
        class_weight=class_weight,
        verbose=1,
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {val_acc:.3f}")

    # ROC/AUC on validation split
    print("\nComputing ROC/AUC curves...")
    compute_and_save_roc(model, val_x, val_y)

    # Quantize
    print("\nQuantizing model to INT8...")
    tflite_model = quantize_model(model, train_x)

    # Save .tflite
    with open(MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {MODEL_PATH} ({len(tflite_model)} bytes)")

    # Export C header
    export_c_header(tflite_model, HEADER_PATH)

    # Verify quantized model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"\nQuantized model details:")
    print(f"  Input: {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"  Output: {output_details[0]['shape']} dtype={output_details[0]['dtype']}")
    print(f"  Input quant: {input_details[0]['quantization_parameters']}")

    print("\nDone! Include 'fruit_model.h' in your firmware.")


if __name__ == "__main__":
    main()
