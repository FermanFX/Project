import numpy as np
from typing import Tuple, Optional, Dict


class SoftmaxRegression:
    """
    Multiclass softmax regression (linear classifier).
    
    Model: s(x) = Wx + b,  p_j(x) = exp(s_j) / sum(exp(s_l))
    
    Parameters:
    -----------
    input_dim : int
        Feature dimension d
    num_classes : int
        Number of classes k
    learning_rate : float
        Learning rate for SGD
    reg_lambda : float
        L2 regularization coefficient
    
    Attributes:
    -----------
    W : np.ndarray
        Weight matrix (k, d)
    b : np.ndarray
        Bias vector (k,)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        learning_rate: float = 0.05,
        reg_lambda: float = 1e-4
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        
        self.W = None
        self.b = None
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights with small random values."""
        self.W = np.random.randn(self.num_classes, self.input_dim) * 0.01
        self.b = np.zeros((self.num_classes,))
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass: compute logits and probabilities.
        
        Args:
            X: Input features (batch_size, input_dim)
        
        Returns:
            Tuple of (logits, probabilities)
            logits: (batch_size, num_classes)
            probabilities: (batch_size, num_classes)
        
        Forward pass:
            logits = X @ W.T + b
            probabilities = softmax(logits)
        """
        logits = X @ self.W.T + self.b
        probabilities = softmax_stable(logits)
        return logits, probabilities
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Input features (batch_size, input_dim)
        
        Returns:
            Tuple of (predicted_labels, probabilities)
            predicted_labels: (batch_size,) - class indices
            probabilities: (batch_size, num_classes)
        """
        logits, probs = self.forward(X)
        labels = np.argmax(probs, axis=1)
        return labels, probs
    
    def backward(self, X: np.ndarray, Y: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backpropagation: compute gradients for softmax regression.
        
        Mathematical derivation:
        
        Cross-entropy loss: L = -sum(Y * log(P)) / n
        
        1. Output sensitivity (∂L/∂S):
           ∂L/∂S = (1/n) * (P - Y)
           Shape: (batch_size, num_classes)
        
        2. Weight gradient (∂L/∂W):
           Using chain rule: ∂L/∂W = (∂L/∂S).T @ X
           Shape: (num_classes, input_dim)
        
        3. Bias gradient (∂L/∂b):
           ∂L/∂b = (∂L/∂S).T @ 1
           Shape: (num_classes,)
        
        Args:
            X: Input features (batch_size, input_dim)
            Y: One-hot labels (batch_size, num_classes)
            P: Predicted probabilities (batch_size, num_classes)
        
        Returns:
            Tuple of (grad_W, grad_b)
            grad_W: (num_classes, input_dim)
            grad_b: (num_classes,)
        """
        n = X.shape[0]
        dL_dS = (P - Y) / n
        grad_W = dL_dS.T @ X
        grad_b = np.sum(dL_dS, axis=0)
        return grad_W, grad_b
    
    def update_parameters(self, grad_W: np.ndarray, grad_b: np.ndarray):
        """
        Update weights using gradients (Gradient Descent).
        
        Formula: θ = θ - learning_rate * gradient
        
        Args:
            grad_W: Gradient of W (num_classes, input_dim)
            grad_b: Gradient of b (num_classes,)
        """
        self.W = self.W - self.learning_rate * grad_W
        self.b = self.b - self.learning_rate * grad_b
    
    def compute_loss(self, X: np.ndarray, Y: np.ndarray, P: np.ndarray) -> float:
        """
        Compute cross-entropy loss with L2 regularization.
        
        Total loss = CrossEntropy + (λ/2) * ||W||²
        
        Args:
            X: Input features
            Y: One-hot labels
            P: Predicted probabilities
        
        Returns:
            Mean loss (scalar)
        """
        n = X.shape[0]
        cross_ent = -np.mean(np.sum(Y * np.log(P + 1e-9), axis=1))
        reg_term = 0.5 * self.reg_lambda * (np.sum(self.W**2) + np.sum(self.b**2))
        return cross_ent + reg_term


class OneHiddenLayerNN:
    """
    One-hidden-layer neural network with tanh activation and softmax output.
    
    Forward pass:
        Z1 = X @ W1.T + b1    (affine transformation)
        H = tanh(Z1)          (hidden activations)
        S = H @ W2.T + b2     (output scores/logits)
        P = softmax(S)        (probabilities)
    
    Parameters:
    -----------
    input_dim : int
        Feature dimension d
    hidden_dim : int
        Hidden layer width
    num_classes : int
        Number of classes k
    learning_rate : float
        Learning rate
    reg_lambda : float
        L2 regularization coefficient
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        learning_rate: float = 0.05,
        reg_lambda: float = 1e-4
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize all weight matrices with Xavier/He initialization."""
        self.W1 = np.random.randn(self.hidden_dim, self.input_dim) * np.sqrt(2.0/self.input_dim)
        self.b1 = np.zeros((self.hidden_dim,))
        self.W2 = np.random.randn(self.num_classes, self.hidden_dim) * np.sqrt(2.0/self.hidden_dim)
        self.b2 = np.zeros((self.num_classes,))
    
    def forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass through the network.
        
        Args:
            X: Input features (batch_size, input_dim)
        
        Returns:
            Dictionary containing all intermediate values:
            Z1: (batch_size, hidden_dim) - pre-activations
            H: (batch_size, hidden_dim) - hidden activations (tanh output)
            S: (batch_size, num_classes) - output logits
            P: (batch_size, num_classes) - probabilities
        
        Step-by-step:
            Z1 = X @ W1.T + b1
            H = tanh(Z1)
            S = H @ W2.T + b2
            P = softmax(S)
        """
        Z1 = X @ self.W1.T + self.b1
        H = tanh_activation(Z1)
        S = H @ self.W2.T + self.b2
        P = softmax_stable(S)
        return {'Z1': Z1, 'H': H, 'S': S, 'P': P}
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Input features
        
        Returns:
            Tuple of (predicted_labels, probabilities)
        """
        cache = self.forward(X)
        labels = np.argmax(cache['P'], axis=1)
        return labels, cache['P']
    
    def backward(self, X: np.ndarray, Y: np.ndarray, cache: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Backpropagation: FULL DERIVATION for one-hidden-layer network.
        
        ┌─────────────────────────────────────────────────────────────┐
        │                    COMPUTATION GRAPH                        │
        ├─────────────────────────────────────────────────────────────┤
        │                                                             │
        │     X ──► Z1 = X·W1ᵀ + b1 ──► H = tanh(Z1)               │
        │                              │                             │
        │                              ▼                             │
        │                    S = H·W2ᵀ + b2 ──► P = softmax(S)       │
        │                                             │              │
        │                                             ▼              │
        │                                           Loss = -log(P_y) │
        │                                                             │
        └─────────────────────────────────────────────────────────────┘
        
        KEY GRADIENT FORMULAS (MUST MEMORIZE):
        ─────────────────────────────────────
        
        1. Output layer sensitivity:
           ∂L/∂S = (1/n) * (P - Y)                    [Shape: (n, k)]
           
           Why? For softmax + cross-entropy, the derivative simplifies to this!
        
        2. W2 gradient (output weights):
           ∂L/∂W2 = (∂L/∂S).T @ H                      [Shape: (k, h)]
           ∂L/∂b2 = sum(∂L/∂S, axis=0)                 [Shape: (k,)]
        
        3. Hidden layer sensitivity (BACKPROPAGATION!):
           ∂L/∂Z1 = (∂L/∂S) @ W2 ⊙ (1 - H²)           [Shape: (n, h)]
           
           Where:
           - (∂L/∂S) @ W2: backprop through W2
           - ⊙ (1 - H²): multiply by tanh derivative
           - tanh'(z) = 1 - tanh²(z)
        
        4. W1 gradient (input weights):
           ∂L/∂W1 = (∂L/∂Z1).T @ X                      [Shape: (h, d)]
           ∂L/∂b1 = sum(∂L/∂Z1, axis=0)                 [Shape: (h,)]
        
        Args:
            X: Input features (n, d)
            Y: One-hot labels (n, k)
            cache: Dictionary from forward pass with keys: Z1, H, S, P
        
        Returns:
            Tuple of (grad_W1, grad_b1, grad_W2, grad_b2)
        """
        # ============================================
        # STEP 1: Output sensitivity ∂L/∂S
        # ============================================
        n = X.shape[0]
        dL_dS = (cache['P'] - Y) / n
        
        # ============================================
        # STEP 2: Gradients for W2 and b2
        # ============================================
        grad_W2 = dL_dS.T @ cache['H']
        grad_b2 = np.sum(dL_dS, axis=0)
        
        # ============================================
        # STEP 3: Backpropagate to hidden layer ∂L/∂Z1
        # ============================================
        # This is the key backpropagation step!
        dL_dH = dL_dS @ self.W2                          # [Shape: (n, h)]
        dL_dZ1 = dL_dH * tanh_derivative(cache['H'])      # [Shape: (n, h)]
        #       # tanh_derivative(H) = 1 - H²
        
        # ============================================
        # STEP 4: Gradients for W1 and b1
        # ============================================
        grad_W1 = dL_dZ1.T @ X
        grad_b1 = np.sum(dL_dZ1, axis=0)
        
        return grad_W1, grad_b1, grad_W2, grad_b2
    
    def update_parameters(self, grad_W1, grad_b1, grad_W2, grad_b2):
        """
        Update all parameters using Gradient Descent.
        
        Formula: θ = θ - learning_rate * gradient
        
        Args:
            grad_W1: Gradient of W1 (hidden_dim, input_dim)
            grad_b1: Gradient of b1 (hidden_dim,)
            grad_W2: Gradient of W2 (num_classes, hidden_dim)
            grad_b2: Gradient of b2 (num_classes,)
        """
        self.W1 = self.W1 - self.learning_rate * grad_W1
        self.b1 = self.b1 - self.learning_rate * grad_b1
        self.W2 = self.W2 - self.learning_rate * grad_W2
        self.b2 = self.b2 - self.learning_rate * grad_b2
    
    def compute_loss(self, Y: np.ndarray, P: np.ndarray, reg_term: float = 0) -> float:
        """
        Compute cross-entropy loss with L2 regularization.
        
        Total loss = -log(p_y) + (λ/2) * ||W1||² + (λ/2) * ||W2||²
        
        Args:
            Y: One-hot labels (n, k)
            P: Predicted probabilities (n, k)
            reg_term: Pre-computed L2 regularization term
        
        Returns:
            Mean loss (scalar)
        """
        n = Y.shape[0]
        cross_ent = -np.mean(np.sum(Y * np.log(P + 1e-9), axis=1))
        return cross_ent + reg_term


def softmax_stable(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    
    Problem: exp(large_number) → overflow
    Solution: subtract max before exp
    
    Formula: softmax(s_j) = exp(s_j - max(s)) / sum(exp(s_i - max(s)))
    
    Example:
        logits = [1000, 1001, 1002]
        Without stability: exp(1002) = inf → NaN
        With stability: [0.09, 0.24, 0.67] → correct!
    
    Args:
        logits: Raw score vectors (batch_size, num_classes)
    
    Returns:
        Probabilities summing to 1 per row
    """
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs


def tanh_activation(Z: np.ndarray) -> np.ndarray:
    """
    Tanh activation function.
    
    Formula: tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    
    Properties:
        - Range: (-1, 1)
        - Zero-centered
        - Smooth derivative
    
    Args:
        Z: Pre-activation values (any shape)
    
    Returns:
        Activated values in range (-1, 1)
    """
    return np.tanh(Z)


def tanh_derivative(H: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh with respect to pre-activation.
    
    Mathematical proof:
        tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
        
        d/dz tanh(z) = 1 - tanh²(z)
        
        Since H = tanh(Z), we have:
        dH/dZ = 1 - H²
    
    Why is this important?
        In backprop: ∂L/∂Z1 = ∂L/∂H ⊙ (1 - H²)
        
        Where ∂L/∂H is the gradient flowing back from the next layer.
    
    Args:
        H: tanh activations (output of tanh(Z))
    
    Returns:
        Element-wise derivative dH/dZ
    
    Example:
        H = [0, 0.5, -0.5]
        derivative = [1, 0.75, 0.75]
        # 1 - 0² = 1
        # 1 - 0.5² = 1 - 0.25 = 0.75
        # 1 - (-0.5)² = 1 - 0.25 = 0.75
    """
    return 1 - H**2
