# code/datasets/split_data.py

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_FILENAME = "mnist.npz"
PROCESSED_TRAIN = "train.npz"
PROCESSED_TEST = "test.npz"

def ensure_dirs(repo_root: Path):
    (repo_root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "processed").mkdir(parents=True, exist_ok=True)

def download_and_save_raw(repo_root: Path):
    # fallback: download via keras (only if user didn't place mnist.npz manually)
    from tensorflow.keras.datasets import mnist
    print("Downloading MNIST via keras (fallback)...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    raw_path = repo_root / "data" / "raw" / RAW_FILENAME
    np.savez_compressed(raw_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(f"Saved raw mnist to {raw_path}")
    return raw_path

def load_raw(repo_root: Path):
    raw_path = repo_root / "data" / "raw" / RAW_FILENAME
    if not raw_path.exists():
        raw_path = download_and_save_raw(repo_root)
    with np.load(raw_path, allow_pickle=True) as f:
        x_train = f["x_train"]
        y_train = f["y_train"]
        x_test = f["x_test"]
        y_test = f["y_test"]
    return x_train, y_train, x_test, y_test

def save_processed(repo_root: Path, x_train, y_train, x_test, y_test):
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / PROCESSED_TRAIN
    test_path = processed_dir / PROCESSED_TEST
    # Save compressed
    np.savez_compressed(train_path, x_train=x_train, y_train=y_train)
    np.savez_compressed(test_path, x_test=x_test, y_test=y_test)
    print(f"Saved processed train to {train_path}, test to {test_path}")

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    ensure_dirs(repo_root)
    x_tr, y_tr, x_te, y_te = load_raw(repo_root)

    # Keras splits: x_train/y_train and x_test/y_test.
    save_processed(repo_root, x_tr, y_tr, x_te, y_te)
    print("Done. Now run `dvc add data/processed/train.npz data/processed/test.npz` to track processed data.")

if __name__ == "__main__":
    main()
