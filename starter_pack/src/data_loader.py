import numpy as np
from typing import Tuple, Dict, List
import os


class DataLoader:
    def __init__(self, data_dir: str = "starter_pack/data"):
        self.data_dir = data_dir
    
    def load_synthetic(self, name: str) -> Dict[str, np.ndarray]:
        """
        Load synthetic dataset (linear_gaussian or moons).
        
        Returns:
            Dictionary with keys: X_train, y_train, X_val, y_val, X_test, y_test
        """
        # Data faylının yolu
        filepath = os.path.join(self.data_dir, f"{name}.npz")
        
        # Faylı yüklə
        data = np.load(filepath)
        
        # Nəticəni qaytar
        return {
            'X_train': data['X_train'],
            'y_train': data['y_train'],
            'X_val': data['X_val'],
            'y_val': data['y_val'],
            'X_test': data['X_test'],
            'y_test': data['y_test']
        }
    
    def load_digits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load digits dataset with fixed train/val/test split.
        
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        
        Data shapes:
            X: (1797, 64) - 64-dimensional flattened digits
            y: (1797,) - labels 0-9
            train_idx: (1074,) - training indices
            val_idx: (355,) - validation indices
            test_idx: (368,) - test indices
        """
        # Digits data yüklə
        digits_data = np.load(os.path.join(self.data_dir, "digits_data.npz"))
        X = digits_data['X']
        y = digits_data['y']
        
        # Split indices yüklə
        split_data = np.load(os.path.join(self.data_dir, "digits_split_indices.npz"))
        train_idx = split_data['train_idx']
        val_idx = split_data['val_idx']
        test_idx = split_data['test_idx']
        
        # Datant split et
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def create_minibatches(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        batch_size: int,
        shuffle: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create minibatches for training.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Labels (n_samples,)
            batch_size: Mini-batch size
            shuffle: Whether to shuffle data
        
        Returns:
            List of (X_batch, y_batch) tuples
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            batches.append((X[batch_idx], y[batch_idx]))
        
        return batches
    
    def one_hot_encode(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Convert labels to one-hot encoding.
        
        Args:
            y: Labels (n_samples,)
            num_classes: Number of classes
        
        Returns:
            One-hot encoded matrix (n_samples, num_classes)
        """
        n = len(y)
        Y = np.zeros((n, num_classes))
        Y[np.arange(n), y] = 1
        return Y


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    """
    return np.mean(y_true == y_pred)


def compute_cross_entropy(y_true_onehot: np.ndarray, P: np.ndarray, eps: float = 1e-9) -> float:
    """
    Compute mean cross-entropy loss.
    """
    log_probs = np.log(P + eps)
    cross_ent = -np.mean(np.sum(y_true_onehot * log_probs, axis=1))
    return cross_ent


def prepare_binary_data(X: np.ndarray, y: np.ndarray, classes: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract binary classification subset for synthetic tasks.
    """
    mask = (y == classes[0]) | (y == classes[1])
    return X[mask], y[mask]


# Test: DataLoader işləyirmi?
if __name__ == "__main__":
    print("Testing DataLoader...")
    
    loader = DataLoader()
    
    # Linear Gaussian
    print("\n[1] Linear Gaussian:")
    linear = loader.load_synthetic('linear_gaussian')
    print(f"    Train: X={linear['X_train'].shape}, y={linear['y_train'].shape}")
    print(f"    Val:   X={linear['X_val'].shape}, y={linear['y_val'].shape}")
    print(f"    Test:  X={linear['X_test'].shape}, y={linear['y_test'].shape}")
    
    # Moons
    print("\n[2] Moons:")
    moons = loader.load_synthetic('moons')
    print(f"    Train: X={moons['X_train'].shape}, y={moons['y_train'].shape}")
    print(f"    Val:   X={moons['X_val'].shape}, y={moons['y_val'].shape}")
    print(f"    Test:  X={moons['X_test'].shape}, y={moons['y_test'].shape}")
    
    # Digits
    print("\n[3] Digits:")
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_digits()
    print(f"    Train: X={X_train.shape}, y={y_train.shape}")
    print(f"    Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"    Test:  X={X_test.shape}, y={y_test.shape}")
    
    print("\n[OK] DataLoader working correctly!")
