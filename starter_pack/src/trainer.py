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
        
        # TODO: np.random.shuffle(indices) for shuffling
        pass
    
    def _one_hot(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert labels to one-hot encoding."""
        # TODO: n = len(y)
        # TODO: Y = np.zeros((n, num_classes))
        # TODO: Y[np.arange(n), y] = 1
        # TODO: return Y
        pass
    
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
        # TODO: np.savez(path, ...) - save all model parameters
        pass
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        # TODO: data = np.load(path)
        # TODO: assign parameters from data
        pass


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
        
        # TODO: Create minibatches with one-hot encoding
        # batches = self.create_minibatches(X_train, y_train, one_hot=True, num_classes=num_classes)
        
        # TODO: for X_batch, Y_batch in batches:
        # TODO:     # Forward pass
        # TODO:     logits, P = self.model.forward(X_batch)
        # TODO:     
        # TODO:     # Backward pass
        # TODO:     grad_W, grad_b = self.model.backward(X_batch, Y_batch, P)
        # TODO:     
        # TODO:     # Add regularization gradient: ∂(λ/2)||W||²/∂W = λW
        # TODO:     grad_W_reg = grad_W + self.reg_lambda * self.model.W
        # TODO:     
        # TODO:     # Optimizer step
        # TODO:     self.optimizer.step(self.model, {'W': grad_W_reg, 'b': grad_b})
        # TODO:     
        # TODO:     # Track metrics
        # TODO:     batch_loss = self.model.compute_loss(X_batch, Y_batch, P)
        # TODO:     epoch_loss += batch_loss * len(X_batch)
        # TODO:     
        # TODO:     # Accuracy
        # TODO:     preds = np.argmax(P, axis=1)
        # TODO:     labels = np.argmax(Y_batch, axis=1)
        # TODO:     correct += np.sum(preds == labels)
        
        # TODO: avg_loss = epoch_loss / n_samples
        # TODO: accuracy = correct / n_samples
        # TODO: return avg_loss, accuracy
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate softmax model.
        
        Args:
            X: Features
            y: Labels
        
        Returns:
            Tuple of (cross_entropy_loss, accuracy)
        """
        # TODO: logits, P = self.model.forward(X)
        # TODO: n = X.shape[0]
        # TODO: Y_onehot = self._one_hot(y, self.model.num_classes)
        # TODO: eps = 1e-9
        # TODO: cross_ent = -np.mean(np.sum(Y_onehot * np.log(P + eps), axis=1))
        # TODO: preds = np.argmax(P, axis=1)
        # TODO: accuracy = np.mean(preds == y)
        # TODO: return cross_ent, accuracy
        pass
    
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
            pass
            # # TODO: train_loss, train_acc = self.train_epoch(X_train, y_train)
            # # TODO: val_loss, val_acc = self.evaluate(X_val, y_val)
            # 
            # # TODO: Append to history
            # # self.history.train_losses.append(train_loss)
            # # self.history.val_losses.append(val_loss)
            # # self.history.train_accuracies.append(train_acc)
            # # self.history.val_accuracies.append(val_acc)
            # 
            # # TODO: Checkpoint logic:
            # # TODO: if checkpoint_policy == 'best_val_loss':
            # # TODO:     if val_loss < self.best_val_loss:
            # # TODO:         save best params
            # # TODO:         self.best_val_loss = val_loss
            # # TODO:         self.best_epoch = epoch
            # 
            # # TODO: if verbose and (epoch + 1) % 10 == 0:
            # # TODO:     print(...)
        
        # # TODO: Restore best parameters if needed


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
        
        # TODO: Similar structure to SoftmaxTrainer but:
        # TODO: - cache = self.model.forward(X_batch) instead of (logits, P)
        # TODO: - 4 gradients: grad_W1, grad_b1, grad_W2, grad_b2
        # TODO: - grads dict: {'W1': ..., 'b1': ..., 'W2': ..., 'b2': ...}
        # TODO: - reg_term = 0.5 * reg_lambda * (sum(W1²) + sum(W2²))
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate neural network.
        """
        # TODO: cache = self.model.forward(X)
        # TODO: P = cache['P']
        # TODO: Same as softmax evaluation
        pass
    
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
            pass
            # # TODO: Same structure as SoftmaxTrainer
            # # TODO: Save best params for all 4 parameters (W1, b1, W2, b2)
