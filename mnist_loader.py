import gzip
from pathlib import Path
import numpy as np


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError(f"Invalid magic number for IDX images: {magic}")
        num = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        buffer = f.read()
    data = np.frombuffer(buffer, dtype=np.uint8)
    return data.reshape(num, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError(f"Invalid magic number for IDX labels: {magic}")
        num = int.from_bytes(f.read(4), "big")
        buffer = f.read()
    data = np.frombuffer(buffer, dtype=np.uint8)
    return data.reshape(num)


def load_mnist(data_dir: Path, normalize: bool = True, flatten: bool = False):
    data_dir = Path(data_dir)
    train_images = _read_idx_images(data_dir / "train-images-idx3-ubyte.gz")
    train_labels = _read_idx_labels(data_dir / "train-labels-idx1-ubyte.gz")
    test_images = _read_idx_images(data_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = _read_idx_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

    if normalize:
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0

    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)

    return train_images, train_labels, test_images, test_labels
