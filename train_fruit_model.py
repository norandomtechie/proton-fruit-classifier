#!/usr/bin/env python3
"""
Train a small fruit classification CNN and export as quantized TFLite model + C array.

Target: RP2350B (520 KB SRAM) with OV7670 camera (160x120 YUY2).
Model input: 48x48 grayscale INT8 (small enough for ~70-100 KB arena).
Classes: apple, banana, orange, background (4 classes).

Usage:
    pip install tensorflow numpy Pillow
    python train_fruit_model.py

The script will:
1. Download Fruits-360 dataset from Kaggle (or use synthetic data for testing)
2. Train a small CNN
3. Quantize to INT8
4. Export as C header file (include/fruit_model.h)
"""

import os
import sys
import numpy as np

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Configuration ---
IMG_SIZE = 48          # Model input: 48x48
NUM_CLASSES = 4        # apple, banana, orange, background
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "fruit_model.tflite"
HEADER_PATH = os.path.join(os.path.dirname(__file__), "include", "fruit_model.h")

CLASS_NAMES = ["apple", "banana", "orange", "background"]

# Fruits-360 class folder names (subset)
FRUIT360_CLASSES = {
    "apple":  ["Apple Braeburn", "Apple Golden 1", "Apple Golden 2", "Apple Golden 3",
               "Apple Granny Smith", "Apple Red 1", "Apple Red 2", "Apple Red 3",
               "Apple Red Delicious", "Apple Red Yellow 1"],
    "banana": ["Banana", "Banana Lady Finger", "Banana Red"],
    "orange": ["Orange"],
}


def build_model():
    """Build a tiny CNN suitable for MCU deployment."""
    model = keras.Sequential([
        # Input: 48x48x1 grayscale
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        # Block 1: 16 filters
        layers.Conv2D(16, 3, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),  # -> 24x24

        # Block 2: 32 filters
        layers.Conv2D(32, 3, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),  # -> 12x12

        # Block 3: 48 filters
        layers.Conv2D(48, 3, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),  # -> 6x6

        # Block 4: 64 filters
        layers.Conv2D(64, 3, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),  # -> 64

        # Classifier
        layers.Dense(NUM_CLASSES),
    ])
    return model


def load_fruits360(data_dir):
    """Load Fruits-360 dataset, mapping to our 4 classes."""
    images = []
    labels = []

    train_dir = os.path.join(data_dir, "Training")
    if not os.path.isdir(train_dir):
        # Try flat structure
        train_dir = data_dir

    for class_idx, (class_name, folder_names) in enumerate(FRUIT360_CLASSES.items()):
        for folder in folder_names:
            folder_path = os.path.join(train_dir, folder)
            if not os.path.isdir(folder_path):
                print(f"  Warning: {folder_path} not found, skipping")
                continue
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                try:
                    img = tf.io.read_file(fpath)
                    img = tf.image.decode_image(img, channels=1)
                    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
                    images.append(img.numpy())
                    labels.append(class_idx)
                except Exception:
                    continue
        print(f"  Loaded {class_name}: {sum(1 for l in labels if l == class_idx)} images")

    # Generate "background" class from random crops / noise
    n_bg = min(len(images) // 3, 500)
    print(f"  Generating {n_bg} background samples")
    for _ in range(n_bg):
        # Random noise images as background
        bg = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8).astype(np.float32)
        images.append(bg)
        labels.append(3)  # background

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def generate_synthetic_data(n_per_class=200):
    """Generate synthetic data for testing when Fruits-360 isn't available."""
    print("Generating synthetic training data (no Fruits-360 found)...")
    images = []
    labels = []

    for class_idx in range(NUM_CLASSES):
        for _ in range(n_per_class):
            img = np.random.randn(IMG_SIZE, IMG_SIZE, 1).astype(np.float32)
            # Add class-specific patterns so the model can learn something
            if class_idx == 0:  # apple - circular bright region
                y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
                mask = ((x - IMG_SIZE//2)**2 + (y - IMG_SIZE//2)**2) < (IMG_SIZE//3)**2
                img[mask] += 2.0
            elif class_idx == 1:  # banana - elongated region
                img[IMG_SIZE//3:2*IMG_SIZE//3, IMG_SIZE//6:5*IMG_SIZE//6] += 2.0
            elif class_idx == 2:  # orange - circular, different texture
                y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
                mask = ((x - IMG_SIZE//2)**2 + (y - IMG_SIZE//2)**2) < (IMG_SIZE//4)**2
                img[mask] += 1.5
                img += np.random.randn(IMG_SIZE, IMG_SIZE, 1).astype(np.float32) * 0.3
            # class 3 = background (just noise)

            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            images.append(img)
            labels.append(class_idx)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


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


def export_c_header(tflite_model, header_path):
    """Export TFLite model as a C header file."""
    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    with open(header_path, 'w') as f:
        f.write("#ifndef FRUIT_MODEL_H_\n")
        f.write("#define FRUIT_MODEL_H_\n\n")
        f.write(f"// Auto-generated by train_fruit_model.py\n")
        f.write(f"// Model input: {IMG_SIZE}x{IMG_SIZE} grayscale INT8\n")
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
    print(f"Input: {IMG_SIZE}x{IMG_SIZE} grayscale, {NUM_CLASSES} classes")
    print()

    # Try to load Fruits-360, fall back to synthetic
    fruits360_dir = os.environ.get("FRUITS360_DIR", "fruits-360")
    if os.path.isdir(fruits360_dir):
        print(f"Loading Fruits-360 from {fruits360_dir}...")
        images, labels = load_fruits360(fruits360_dir)
    else:
        print(f"Fruits-360 not found at '{fruits360_dir}'.")
        print("Set FRUITS360_DIR env var or download from:")
        print("  https://www.kaggle.com/datasets/moltean/fruits")
        print()
        images, labels = generate_synthetic_data(n_per_class=300)

    print(f"\nDataset: {len(images)} images")

    # Shuffle and split
    indices = np.random.permutation(len(images))
    images, labels = images[indices], labels[indices]
    split = int(0.8 * len(images))
    train_x, val_x = images[:split], images[split:]
    train_y, val_y = labels[:split], labels[split:]

    print(f"Train: {len(train_x)}, Val: {len(val_x)}")
    print()

    # Build and train
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    model.summary()
    print()

    model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(val_x, val_y, verbose=0)
    print(f"\nValidation accuracy: {val_acc:.3f}")

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
