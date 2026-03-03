"""
MNIST × Stiefel多様体 学習スクリプト

モデル: y = softmax(Wx + b), W ∈ St(10, 784)
最適化: Riemannian SGD on Stiefel manifold
出力: public/data/mnist_stiefel.json (PCA射影済み3D座標)
"""

import gzip
import json
import pathlib
import struct
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

# ============================================================
# MNIST data loading (no torchvision dependency)
# ============================================================

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_mnist(data_dir: pathlib.Path):
    """Download MNIST dataset if not already cached."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MNIST_URLS.items():
        filepath = data_dir / f"{name}.gz"
        if not filepath.exists():
            print(f"  Downloading {name}...")
            urllib.request.urlretrieve(url, filepath)


def load_mnist_images(filepath: pathlib.Path) -> torch.Tensor:
    with gzip.open(filepath, "rb") as f:
        _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    # Normalize: same as transforms.Normalize((0.1307,), (0.3081,))
    images = torch.from_numpy(data.copy()).float() / 255.0
    images = (images - 0.1307) / 0.3081
    return images


def load_mnist_labels(filepath: pathlib.Path) -> torch.Tensor:
    with gzip.open(filepath, "rb") as f:
        _magic, _n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return torch.from_numpy(data.copy()).long()


# ============================================================
# Stiefel manifold utilities
# ============================================================


def sample_stiefel(p: int, n: int) -> torch.Tensor:
    """Sample uniformly from St(p, n) via QR decomposition of Gaussian matrix."""
    Z = torch.randn(p, n)
    Q, R = torch.linalg.qr(Z.T)  # Q: (n, p)
    # Fix sign ambiguity
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    return Q.T  # (p, n)


def stiefel_project_tangent(W: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Project Z onto the tangent space of St(p, n) at W.

    proj_W(Z) = Z - W @ sym(W^T Z)
    where sym(A) = (A + A^T) / 2
    """
    WtZ = W @ Z.T
    sym = 0.5 * (WtZ + WtZ.T)
    return Z - sym @ W


def stiefel_retract_qr(W: torch.Tensor) -> torch.Tensor:
    """Retract onto St(p, n) via QR decomposition."""
    Q, R = torch.linalg.qr(W.T)
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    return Q.T


def sample_tangent_perturbation(W: torch.Tensor, scale: float) -> torch.Tensor:
    """Sample a random tangent vector at W and retract to get a nearby point on St(p,n)."""
    Z = torch.randn_like(W) * scale
    Z_tan = stiefel_project_tangent(W, Z)
    return stiefel_retract_qr(W + Z_tan)


# ============================================================
# Training
# ============================================================


def train():
    device = torch.device("cpu")

    # Download and load MNIST
    data_dir = pathlib.Path("data/mnist")
    print("Loading MNIST...")
    download_mnist(data_dir)

    train_images = load_mnist_images(data_dir / "train_images.gz").to(device)
    train_labels = load_mnist_labels(data_dir / "train_labels.gz").to(device)
    test_images = load_mnist_images(data_dir / "test_images.gz").to(device)
    test_labels = load_mnist_labels(data_dir / "test_labels.gz").to(device)

    print(f"  Train: {train_images.shape[0]} samples, Test: {test_images.shape[0]} samples")

    # Subset for landscape evaluation (5000 samples)
    landscape_indices = torch.randperm(train_images.shape[0])[:5000]
    landscape_images = train_images[landscape_indices]
    landscape_labels = train_labels[landscape_indices]

    # Initialize W ∈ St(10, 784) and bias b
    lr = 0.01
    epochs = 10
    batch_size = 128
    W = sample_stiefel(10, 784).to(device).requires_grad_(False)
    b = torch.zeros(10, device=device)

    # Record optimization path
    path_weights: list[np.ndarray] = []
    path_losses: list[float] = []
    epoch_losses: list[float] = []
    epoch_accuracies: list[float] = []

    step_count = 0
    record_every = 5
    n_train = train_images.shape[0]

    print("Starting training...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Shuffle
        perm = torch.randperm(n_train)

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            images = train_images[idx]
            labels = train_labels[idx]

            # Forward pass: y = softmax(Wx + b)
            W_param = W.detach().requires_grad_(True)
            b_param = b.detach().requires_grad_(True)
            logits = images @ W_param.T + b_param
            loss = F.cross_entropy(logits, labels)

            # Compute Euclidean gradients
            loss.backward()
            eucl_grad_W = W_param.grad.detach()
            eucl_grad_b = b_param.grad.detach()

            # Riemannian gradient: project to tangent space of St(10, 784)
            riem_grad = stiefel_project_tangent(W, eucl_grad_W)

            # Update W via Riemannian SGD + QR retraction
            W = W - lr * riem_grad
            W = stiefel_retract_qr(W)

            # Update b via standard SGD
            b = b - lr * eucl_grad_b

            # Track
            running_loss += loss.item() * images.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += images.size(0)

            # Record path
            if step_count % record_every == 0:
                path_weights.append(W.detach().cpu().numpy().flatten())
                path_losses.append(loss.item())
            step_count += 1

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)
        print(f"  Epoch {epoch + 1}/{epochs}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

        # Verify Stiefel constraint
        WWT = W @ W.T
        identity_error = torch.norm(WWT - torch.eye(10, device=device)).item()
        print(f"    ||WW^T - I||_F = {identity_error:.6f}")

    # Final evaluation
    W_eval = W.detach()
    b_eval = b.detach()
    with torch.no_grad():
        test_logits = test_images @ W_eval.T + b_eval
        test_pred = test_logits.argmax(dim=1)
        final_accuracy = (test_pred == test_labels).float().mean().item()
    final_loss = epoch_losses[-1]
    print(f"\nFinal test accuracy: {final_accuracy:.4f}")

    # ============================================================
    # Helper: PCA projection (manual matmul to avoid sklearn overflow)
    # ============================================================
    def pca_project_3d(weights_list: list[np.ndarray]):
        """Fit PCA on given weights, return (projected, components, mean, variance)."""
        arr = np.array(weights_list, dtype=np.float64)
        mean = arr.mean(axis=0)
        centered = arr - mean
        pca = PCA(n_components=3, svd_solver="full")
        pca.fit(centered)
        projected = (centered @ pca.components_.T).tolist()
        return projected, pca.components_, mean, pca.explained_variance_ratio_.tolist()

    def project_with(weights_list: list[np.ndarray], components, mean):
        arr = np.array(weights_list, dtype=np.float64)
        return ((arr - mean) @ components.T).tolist()

    # ============================================================
    # LOCAL: PCA on path → directed perturbations along PC directions
    # ============================================================
    print("\n[Local] Computing PCA on optimization path...")
    path_array = np.array(path_weights, dtype=np.float64)
    local_path_3d, local_pcs, local_mean, local_variance = pca_project_3d(path_weights)
    print(f"  PCA explained variance: {local_variance}")

    print("[Local] Sampling 2000 perturbation points...")
    n_landscape = 2000
    local_weights: list[np.ndarray] = []
    local_losses: list[float] = []

    path_tensors = [torch.from_numpy(w.reshape(10, 784)).float().to(device) for w in path_weights]
    n_path = len(path_tensors)
    pc_tensors = [torch.from_numpy(pc.reshape(10, 784)).float().to(device) for pc in local_pcs]

    rng = np.random.RandomState(42)
    n_anchors = min(100, n_path)
    anchor_step = max(1, n_path // n_anchors)
    anchor_indices = list(range(0, n_path, anchor_step))
    samples_per_anchor = n_landscape // len(anchor_indices) + 1

    with torch.no_grad():
        count = 0
        for anchor_idx in anchor_indices:
            W_anchor = path_tensors[anchor_idx]
            for _j in range(samples_per_anchor):
                if count >= n_landscape:
                    break
                coeffs = rng.randn(3) * 0.3
                delta = sum(c * pc for c, pc in zip(coeffs, pc_tensors))
                delta_tan = stiefel_project_tangent(W_anchor, delta)
                W_perturbed = stiefel_retract_qr(W_anchor + delta_tan)
                logits = landscape_images @ W_perturbed.T
                loss_val = F.cross_entropy(logits, landscape_labels).item()
                local_weights.append(W_perturbed.cpu().numpy().flatten())
                local_losses.append(loss_val)
                count += 1
            if count >= n_landscape:
                break

    local_landscape_3d = project_with(local_weights, local_pcs, local_mean)
    print(f"  {len(local_weights)} points sampled")

    # ============================================================
    # GLOBAL: wider perturbations around path (scale=2.0 vs local 0.3)
    # Same PCA space as local, but explores further from the path
    # ============================================================
    print("\n[Global] Sampling 8000 wide perturbation points around path...")
    n_global = 8000
    global_weights: list[np.ndarray] = []
    global_losses: list[float] = []

    rng_global = np.random.RandomState(123)
    n_anchors_g = min(100, n_path)
    anchor_step_g = max(1, n_path // n_anchors_g)
    anchor_indices_g = list(range(0, n_path, anchor_step_g))
    samples_per_anchor_g = n_global // len(anchor_indices_g) + 1

    with torch.no_grad():
        count_g = 0
        for anchor_idx in anchor_indices_g:
            W_anchor = path_tensors[anchor_idx]
            for _j in range(samples_per_anchor_g):
                if count_g >= n_global:
                    break
                coeffs = rng_global.randn(3) * 2.0  # wider than local (0.3)
                delta = sum(c * pc for c, pc in zip(coeffs, pc_tensors))
                delta_tan = stiefel_project_tangent(W_anchor, delta)
                W_perturbed = stiefel_retract_qr(W_anchor + delta_tan)
                logits = landscape_images @ W_perturbed.T
                loss_val = F.cross_entropy(logits, landscape_labels).item()
                global_weights.append(W_perturbed.cpu().numpy().flatten())
                global_losses.append(loss_val)
                count_g += 1
            if count_g >= n_global:
                break

    # Use the same PCA as local (fitted on path) so both views share coordinates
    global_landscape_3d = project_with(global_weights, local_pcs, local_mean)
    global_path_3d = project_with(path_weights, local_pcs, local_mean)
    global_variance = local_variance  # same PCA space

    # ============================================================
    # JSON output — each mode has its own PCA coordinate system
    # ============================================================
    output = {
        "metadata": {
            "model": "y = softmax(Wx + b), W ∈ St(10, 784)",
            "epochs": epochs,
            "learning_rate": lr,
            "final_accuracy": round(final_accuracy, 4),
            "final_loss": round(final_loss, 4),
        },
        "local": {
            "pca_explained_variance": [round(v, 6) for v in local_variance],
            "landscape": {
                "points": [[round(c, 6) for c in p] for p in local_landscape_3d],
                "losses": [round(l, 6) for l in local_losses],
            },
            "path": {
                "points": [[round(c, 6) for c in p] for p in local_path_3d],
                "losses": [round(l, 4) for l in path_losses],
            },
        },
        "global": {
            "pca_explained_variance": [round(v, 6) for v in global_variance],
            "landscape": {
                "points": [[round(c, 6) for c in p] for p in global_landscape_3d],
                "losses": [round(l, 6) for l in global_losses],
            },
            "path": {
                "points": [[round(c, 6) for c in p] for p in global_path_3d],
                "losses": [round(l, 4) for l in path_losses],
            },
        },
        "training_history": {
            "epoch_losses": [round(l, 4) for l in epoch_losses],
            "epoch_accuracies": [round(a, 4) for a in epoch_accuracies],
        },
    }

    out_path = pathlib.Path("public/data/mnist_stiefel.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f)

    file_size = out_path.stat().st_size / 1024
    print(f"\nSaved to {out_path} ({file_size:.1f} KB)")
    print(f"  Local: {len(local_landscape_3d)} landscape + {len(local_path_3d)} path")
    print(f"  Global: {len(global_landscape_3d)} landscape + {len(global_path_3d)} path")


if __name__ == "__main__":
    train()
