# code/models/train_digits.py
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.keras
import argparse

MODEL_FILENAME = "digit_recognizer_cnn.h5"
PROCESSED_TRAIN = "train.npz"
PROCESSED_TEST = "test.npz"

def get_data(repo_root: Path):
    proc_train = repo_root / "data" / "processed" / PROCESSED_TRAIN
    proc_test = repo_root / "data" / "processed" / PROCESSED_TEST
    if not proc_train.exists() or not proc_test.exists():
        raise FileNotFoundError("Processed data not found. Run code/datasets/split_data.py first and dvc-add the outputs.")
    with np.load(proc_train, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
    with np.load(proc_test, allow_pickle=True) as f:
        x_test, y_test = f["x_test"], f["y_test"]

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

def build_cnn_model(input_shape=(28,28,1), num_classes=10, dropout_rate=0.4):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="mnist_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main(epochs=20, batch_size=64):
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILENAME

    (x_train, y_train), (x_test, y_test) = get_data(repo_root)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        validation_split=0.1
    )

    train_gen = datagen.flow(x_train, y_train, batch_size=batch_size, subset="training")
    val_gen = datagen.flow(x_train, y_train, batch_size=batch_size, subset="validation")

    model = build_cnn_model()
    cb_early = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    cb_reduce = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    cb_checkpoint = callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True)

    # MLflow autolog
    mlflow.set_experiment("mnist_digit_recognizer")
    mlflow.keras.autolog()

    with mlflow.start_run(run_name="cnn_augmented"):
        # log some parameters explicitly
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

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

        # ensure best model saved
        mlflow.log_artifact(str(model_path), artifact_path="models")
        print(f"Saved model to {model_path}")

    # Evaluate
    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {loss:.4f}  Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
