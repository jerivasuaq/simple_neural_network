import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from mnist_loader import load_mnist


def plot_grid(images: np.ndarray, labels: np.ndarray, n: int = 25, cols: int = 5, cmap: str = "gray",
              figsize=(6, 6), save_path: Path | None = None):
    if images.ndim == 2:
        # flattened images
        img_h = int(np.sqrt(images.shape[1]))
        images = images.reshape(images.shape[0], img_h, img_h)

    n = min(n, images.shape[0])
    cols = max(1, cols)
    rows = (n + cols - 1) // cols

    indices = random.sample(range(images.shape[0]), n)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis('off')

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.imshow(images[idx], cmap=cmap)
        ax.set_title(str(int(labels[idx])))
        ax.axis('off')

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved sample grid to: {save_path}")
    else:
        try:
            plt.show()
        except Exception:
            out = Path('mnist_sample.png')
            fig.savefig(out, dpi=150)
            print(f"Could not show figure — saved sample grid to: {out}")


if __name__ == "__main__":
    data_dir = Path("./mnist_data")
    if not data_dir.exists():
        print("MNIST data not found. Run `python main.py` to download first.")
        raise SystemExit(1)

    train_images, train_labels, test_images, test_labels = load_mnist(data_dir)
    plot_grid(train_images, train_labels, n=25, cols=5, save_path=None)
