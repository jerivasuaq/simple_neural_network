

import argparse
import gzip
import os
import urllib.request
from pathlib import Path

MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_file(url: str, dest_path: Path) -> None:
    def progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        percent = min(100, int(downloaded * 100 / total_size)) if total_size else 0
        print(f"Downloading {dest_path.name}: {percent}%\r", end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=progress)
    print(f"Downloaded {dest_path.name}")


def download_mnist(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MNIST_URLS.items():
        dest_path = data_dir / filename
        if dest_path.exists():
            print(f"Already exists: {dest_path}")
            continue
        download_file(url, dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple neural network MNIST setup")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download MNIST dataset files into the data directory",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent / "data"),
        help="Directory where MNIST files are stored",
    )
    args = parser.parse_args()

    if args.download:
        data_path = Path(args.data_dir)
        download_mnist(data_path)
        print(f"MNIST files saved to: {data_path}")
    else:
        print("Simple neural network")
        print("Run 'python main.py --download' to download the MNIST dataset.")
