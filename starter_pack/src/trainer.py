import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

from .models import SoftmaxRegression, OneHiddenLayerNN
from .optimizers import Optimizer, SGD, Adam, Momentum


@dataclass
class TrainingHistory:
    """
    Container for tracking training metrics over time.

    Attributes:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch
    """
    train_losses: List[float]
    val_losses: List[float]
    train_accuracies: List[float]
    val_accuracies: List[float]

    def to_dict(self) -> Dict[str, List]:
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accuracies,
            'val_acc': self.val_accuracies
        }


class Trainer:
    """
    Base trainer class for classification models.

    Handles the training loop, evaluation, checkpointing, and logging.
    """

    def __init__(
        self,
        model,
        optimizer: Optimizer,
        epochs: int = 200,
        batch_size: int = 64,
        verbose: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.history: Optional[TrainingHistory] = None
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_params: Optional[Dict] = None

    def create_minibatches(
        self,
        X: np.ndarray,
        y: np.ndarray,
        one_hot: bool = False,
        num_classes: int = 10
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create minibatches from data.

        Args:
            X: Input features (n_samples, n_features)
            y: Labels (n_samples,)
            one_hot: Whether to one-hot encode labels
            num_classes: Number of classes (for one-hot encoding)

        Returns:
            List of (X_batch, Y_batch) tuples
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        np.random.shuffle(indices)

        batches = []
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            if one_hot:
                Y_batch = self._one_hot(y_batch, num_classes)
            else:
                Y_batch = y_batch
            batches.append((X_batch, Y_batch))
        return batches

    def _one_hot(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert labels to one-hot encoding."""
        n = len(y)
        Y = np.zeros((n, num_classes))
        Y[np.arange(n), y] = 1
        return Y

    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, float]:
        """
        Train for one epoch.

        Pseudocode:
            for each minibatch:
                1. Forward pass
                2. Backward pass
                3. Update parameters
                4. Accumulate loss and accuracy

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Tuple of (average_loss, accuracy) for the epoch
        """
        raise NotImplementedError

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model on given data.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            Tuple of (loss, accuracy)
        """
        raise NotImplementedError

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        checkpoint_policy: str = 'best_val_loss'
    ) -> TrainingHistory:
        """
        Full training loop with validation tracking.

        Training Loop:
            for epoch in range(epochs):
                1. Train one epoch (update parameters)
                2. Evaluate on validation set
                3. Save best model if validation improved
                4. Log progress

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            checkpoint_policy: When to save checkpoints
                              ('best_val_loss' or 'final')

        Returns:
            TrainingHistory object with metrics
        """
        raise NotImplementedError

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        params = {}
        if hasattr(self.model, 'W'):
            params['W'] = self.model.W
        if hasattr(self.model, 'b'):
            params['b'] = self.model.b
        if hasattr(self.model, 'W1'):
            params['W1'] = self.model.W1
        if hasattr(self.model, 'b1'):
            params['b1'] = self.model.b1
        if hasattr(self.model, 'W2'):
            params['W2'] = self.model.W2
        if hasattr(self.model, 'b2'):
            params['b2'] = self.model.b2
        np.savez(path, **params)

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        data = np.load(path)
        if 'W' in data:
            self.model.W = data['W']
        if 'b' in data:
            self.model.b = data['b']
        if 'W1' in data:
            self.model.W1 = data['W1']
        if 'b1' in data:
            self.model.b1 = data['b1']
        if 'W2' in data:
            self.model.W2 = data['W2']
        if 'b2' in data:
            self.model.b2 = data['b2']


class SoftmaxTrainer(Trainer):
    """
    Trainer for Softmax Regression model.

    Implements the training loop specifically for softmax regression.
    """

    def __init__(
        self,
        model: SoftmaxRegression,
        optimizer: Optimizer = None,
        epochs: int = 200,
        batch_size: int = 64,
        reg_lambda: float = 1e-4,
        verbose: bool = True
    ):
        if optimizer is None:
            optimizer = SGD(learning_rate=0.05)

        super().__init__(model, optimizer, epochs, batch_size, verbose)
        self.reg_lambda = reg_lambda

    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, float]:
        """
        Train for one epoch on softmax model.

        Algorithm:
            1. Create minibatches
            2. For each batch:
               a. Forward pass: get probabilities
               b. Backward pass: compute gradients
               c. Apply regularization gradient
               d. Update parameters
            3. Return average loss and accuracy
        """
        n_samples = X_train.shape[0]
        num_classes = self.model.num_classes

        epoch_loss = 0.0
        correct = 0

        batches = self.create_minibatches(X_train, y_train, one_hot=True, num_classes=num_classes)

        for X_batch, Y_batch in batches:
            # Forward pass
            logits, P = self.model.forward(X_batch)

            # Backward pass
            grad_W, grad_b = self.model.backward(X_batch, Y_batch, P)

            # Add regularization gradient: ∂(λ/2)||W||²/∂W = λW
            grad_W_reg = grad_W + self.reg_lambda * self.model.W

            # Optimizer step
            self.optimizer.step(self.model, {'W': grad_W_reg, 'b': grad_b})

            # Track metrics
            batch_loss = self.model.compute_loss(X_batch, Y_batch, P)
            epoch_loss += batch_loss * len(X_batch)

            # Accuracy
            preds = np.argmax(P, axis=1)
            labels = np.argmax(Y_batch, axis=1)
            correct += np.sum(preds == labels)

        avg_loss = epoch_loss / n_samples
        accuracy = correct / n_samples
        return avg_loss, accuracy

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate softmax model.

        Args:
            X: Features
            y: Labels

        Returns:
            Tuple of (cross_entropy_loss, accuracy)
        """
        logits, P = self.model.forward(X)
        Y_onehot = self._one_hot(y, self.model.num_classes)
        eps = 1e-9
        cross_ent = -np.mean(np.sum(Y_onehot * np.log(P + eps), axis=1))
        preds = np.argmax(P, axis=1)
        accuracy = np.mean(preds == y)
        return cross_ent, accuracy

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        checkpoint_policy: str = 'best_val_loss'
    ) -> TrainingHistory:
        """
        Train softmax regression with history tracking.

        Args:
            checkpoint_policy: 'best_val_loss' = restore best epoch's model
                             'final' = keep final epoch's model
        """
        self.history = TrainingHistory(
            train_losses=[],
            val_losses=[],
            train_accuracies=[],
            val_accuracies=[]
        )

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(X_train, y_train)
            val_loss, val_acc = self.evaluate(X_val, y_val)

            # Append to history
            self.history.train_losses.append(train_loss)
            self.history.val_losses.append(val_loss)
            self.history.train_accuracies.append(train_acc)
            self.history.val_accuracies.append(val_acc)

            # Checkpoint logic:
            if checkpoint_policy == 'best_val_loss':
                if val_loss < self.best_val_loss:
                    # save best params
                    self.best_params = {'W': self.model.W.copy(), 'b': self.model.b.copy()}
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Restore best parameters if needed
        if checkpoint_policy == 'best_val_loss' and self.best_params is not None:
            self.model.W = self.best_params['W']
            self.model.b = self.best_params['b']
            if self.verbose:
                print(f"Restored best model from epoch {self.best_epoch+1}")

        return self.history


class NNTrainer(Trainer):
    """
    Trainer for One-Hidden-Layer Neural Network.
    """

    def __init__(
        self,
        model: OneHiddenLayerNN,
        optimizer: Optimizer = None,
        epochs: int = 200,
        batch_size: int = 64,
        reg_lambda: float = 1e-4,
        verbose: bool = True
    ):
        if optimizer is None:
            optimizer = SGD(learning_rate=0.05)

        super().__init__(model, optimizer, epochs, batch_size, verbose)
        self.reg_lambda = reg_lambda

    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, float]:
        """
        Train for one epoch on neural network.

        Same structure as SoftmaxTrainer but:
        - Different forward pass (returns cache)
        - Different backward pass (4 gradients)
        - Different parameter update (4 parameters)
        """
        n_samples = X_train.shape[0]
        num_classes = self.model.num_classes

        epoch_loss = 0.0
        correct = 0

        batches = self.create_minibatches(X_train, y_train, one_hot=True, num_classes=num_classes)

        for X_batch, Y_batch in batches:
            # Forward pass
            cache = self.model.forward(X_batch)

            # Backward pass
            grad_W1, grad_b1, grad_W2, grad_b2 = self.model.backward(X_batch, Y_batch, cache)

            # Add regularization gradient
            grad_W1_reg = grad_W1 + self.reg_lambda * self.model.W1
            grad_W2_reg = grad_W2 + self.reg_lambda * self.model.W2

            # Optimizer step
            reg_term = 0.5 * self.reg_lambda * (np.sum(self.model.W1**2) + np.sum(self.model.W2**2))

            self.optimizer.step(self.model, {'W1': grad_W1_reg, 'b1': grad_b1, 'W2': grad_W2_reg, 'b2': grad_b2})
            batch_loss = self.model.compute_loss(Y_batch, cache['P'], reg_term)
            epoch_loss += batch_loss * len(X_batch)

            # Accuracy
            preds = np.argmax(cache['P'], axis=1)
            labels = np.argmax(Y_batch, axis=1)
            correct += np.sum(preds == labels)

        avg_loss = epoch_loss / n_samples
        accuracy = correct / n_samples
        return avg_loss, accuracy

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate neural network.
        """
        cache = self.model.forward(X)
        P = cache['P']
        Y_onehot = self._one_hot(y, self.model.num_classes)
        eps = 1e-9
        cross_ent = -np.mean(np.sum(Y_onehot * np.log(P + eps), axis=1))
        preds = np.argmax(P, axis=1)
        accuracy = np.mean(preds == y)
        return cross_ent, accuracy

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        checkpoint_policy: str = 'best_val_loss'
    ) -> TrainingHistory:
        """
        Train neural network with history tracking.
        """
        self.history = TrainingHistory(
            train_losses=[],
            val_losses=[],
            train_accuracies=[],
            val_accuracies=[]
        )

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(X_train, y_train)
            val_loss, val_acc = self.evaluate(X_val, y_val)

            # Append to history
            self.history.train_losses.append(train_loss)
            self.history.val_losses.append(val_loss)
            self.history.train_accuracies.append(train_acc)
            self.history.val_accuracies.append(val_acc)

            # Checkpoint logic:
            if checkpoint_policy == 'best_val_loss':
                if val_loss < self.best_val_loss:
                    # save best params
                    self.best_params = {
                        'W1': self.model.W1.copy(), 'b1': self.model.b1.copy(),
                        'W2': self.model.W2.copy(), 'b2': self.model.b2.copy()
                    }
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Restore best parameters if needed
        if checkpoint_policy == 'best_val_loss' and self.best_params is not None:
            self.model.W1 = self.best_params['W1']
            self.model.b1 = self.best_params['b1']
            self.model.W2 = self.best_params['W2']
            self.model.b2 = self.best_params['b2']
            if self.verbose:
                print(f"Restored best model from epoch {self.best_epoch+1}")

        return self.history