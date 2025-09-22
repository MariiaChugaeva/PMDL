# code/models/train_digits.py
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

MODEL_FILENAME = "digit_recognizer_cnn.h5"

def get_data():
    """Load MNIST, normalize to [0,1], and add channel dimension."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # add channel axis for Conv2D: (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

def build_cnn_model(input_shape=(28, 28, 1), num_classes=10, dropout_rate=0.4):
    """
    Build a small but effective CNN.
    Returns a compiled Keras model.
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="mnist_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main(epochs=30, batch_size=64):
    # Prepare directories
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILENAME

    # Load data
    (x_train, y_train), (x_test, y_test) = get_data()

    # Data augmentation generator (small rotations, shifts, zoom)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        validation_split=0.1  # reserve 10% of training data for validation
    )

    train_gen = datagen.flow(x_train, y_train, batch_size=batch_size, subset="training")
    val_gen = datagen.flow(x_train, y_train, batch_size=batch_size, subset="validation")

    # Build model
    model = build_cnn_model(input_shape=(28,28,1), num_classes=10, dropout_rate=0.4)
    model.summary()

    # Callbacks
    cb_early = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    cb_reduce = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    cb_checkpoint = callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True)

    # Train
    steps_per_epoch = max(1, train_gen.n // train_gen.batch_size)
    validation_steps = max(1, val_gen.n // val_gen.batch_size)

    model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[cb_early, cb_reduce, cb_checkpoint]
    )

    # Load best weights (guarantee)
    model = tf.keras.models.load_model(model_path)

    # Evaluate on test
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {loss:.4f}  Test accuracy: {acc:.4f}")
    print(f"Saved CNN model to {model_path}")

if __name__ == "__main__":
    main()
