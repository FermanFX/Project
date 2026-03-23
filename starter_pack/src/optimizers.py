import numpy as np
from typing import Dict, Callable, Optional


class Optimizer:
    """Base class for optimizers."""
    
    def __init__(self, learning_rate: float = 0.05):
        self.learning_rate = learning_rate
    
    def step(self, model, gradients: Dict):
        """
        Perform one optimization step.
        
        Args:
            model: Model with parameters to update
            gradients: Dictionary of gradients
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Formula:
        θ = θ - lr × ∇L
    
    Where:
        θ = parameters
        lr = learning rate
        ∇L = gradient of loss
    
    Properties:
        - Simple and effective
        - May oscillate in narrow valleys
        - Requires careful learning rate tuning
    """
    
    def __init__(self, learning_rate: float = 0.05):
        super().__init__(learning_rate)
    
    def step(self, model, grads: Dict[str, np.ndarray]):
        """
        Update model parameters using SGD.
        
        Args:
            model: Model with .W, .b (softmax) 
                   or .W1, .b1, .W2, .b2 (NN)
            grads: Dictionary of gradients keyed by parameter names
        
        For Softmax:
            grads = {'W': grad_W, 'b': grad_b}
        
        For Neural Network:
            grads = {'W1': grad_W1, 'b1': grad_b1, 'W2': grad_W2, 'b2': grad_b2}
        """
        # ============================================
        # TODO: For each parameter in grads:
        #       param = param - learning_rate * gradient
        # ============================================
        
        if hasattr(model, 'W') and 'W' in grads:
            # TODO: model.W = model.W - self.learning_rate * grads['W']
            pass
        if hasattr(model, 'b') and 'b' in grads:
            # TODO: model.b = model.b - self.learning_rate * grads['b']
            pass
        
        if hasattr(model, 'W1') and 'W1' in grads:
            # TODO: model.W1 = model.W1 - self.learning_rate * grads['W1']
            # TODO: model.b1 = model.b1 - self.learning_rate * grads['b1']
            # TODO: model.W2 = model.W2 - self.learning_rate * grads['W2']
            # TODO: model.b2 = model.b2 - self.learning_rate * grads['b2']
            pass


class Momentum(Optimizer):
    """
    SGD with Momentum.
    
    Idea: Accumulate past gradients like a rolling ball gaining momentum.
    
    Formulas:
        v_t = momentum × v_{t-1} + ∇L
        θ = θ - lr × v_t
    
    Where:
        v_t = velocity (accumulated gradient)
        momentum = decay factor (typically 0.9)
    
    Why momentum helps:
        1. Accelerates convergence (builds up velocity)
        2. Reduces oscillations (dampens perpendicular directions)
        3. Escapes local minima better
    
    Visual intuition:
        - Without momentum: walking in small steps
        - With momentum: rolling ball, builds up speed
    """
    
    def __init__(self, learning_rate: float = 0.05, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        # TODO: Initialize velocity dictionaries (for each parameter)
        self.velocity: Dict[str, np.ndarray] = {}
    
    def step(self, model, grads: Dict[str, np.ndarray]):
        """
        Update parameters with momentum.
        
        Algorithm:
            for each parameter θ:
                v = momentum * v + gradient
                θ = θ - lr * v
        
        Args:
            model: Model with parameters
            grads: Dictionary of gradients
        """
        # ============================================
        # TODO: For each parameter:
        #       1. Initialize velocity if not exists
        #       2. v = momentum * v + gradient
        #       3. θ = θ - lr * v
        # ============================================
        
        for param_name, grad in grads.items():
            # TODO: if param_name not in velocity, initialize with zeros
            # TODO: velocity[param_name] = momentum * velocity[param_name] + grad
            # TODO: model parameter = model parameter - lr * velocity[param_name]
            pass
    
    def reset(self):
        """Reset velocity for new training."""
        # TODO: self.velocity = {}
        pass


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    Combines ideas from:
        1. Momentum (first moment estimation)
        2. RMSProp (second moment estimation, per-parameter learning rate)
    
    Formulas:
        m_t = β₁ × m_{t-1} + (1 - β₁) × ∇L     ← biased first moment
        v_t = β₂ × v_{t-1} + (1 - β₂) × (∇L)² ← biased second moment
        
        m_hat = m_t / (1 - β₁ᵗ)  ← bias-corrected first moment
        v_hat = v_t / (1 - β₂ᵗ)  ← bias-corrected second moment
        
        θ = θ - lr × m_hat / (√v_hat + ε)
    
    Default parameters:
        β₁ = 0.9 (momentum decay)
        β₂ = 0.999 (RMSProp decay)
        ε = 1e-8 (numerical stability)
    
    Why Adam works well:
        1. Adaptive learning rates per parameter
        2. Momentum for fast convergence
        3. Bias correction for warm start
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # First moment estimates (like velocity in momentum)
        # TODO: Initialize m dictionary
        self.m: Dict[str, np.ndarray] = {}
        
        # Second moment estimates (like adaptive learning rate)
        # TODO: Initialize v dictionary
        self.v: Dict[str, np.ndarray] = {}
        
        self.t = 0  # timestep
    
    def step(self, model, grads: Dict[str, np.ndarray]):
        """
        Update parameters using Adam.
        
        Algorithm:
            t = t + 1
            for each parameter θ:
                m = β₁*m + (1-β₁)*gradient
                v = β₂*v + (1-β₂)*(gradient)²
                m_hat = m / (1 - β₁ᵗ)
                v_hat = v / (1 - β₂ᵗ)
                θ = θ - lr * m_hat / (√v_hat + ε)
        
        Args:
            model: Model with parameters
            grads: Dictionary of gradients
        """
        self.t += 1
        
        for param_name, grad in grads.items():
            # TODO: Initialize m[param_name] and v[param_name] if not exists
            
            # ============================================
            # UPDATE FIRST MOMENT (m_t)
            # m_t = β₁ * m_{t-1} + (1 - β₁) * ∇L
            # ============================================
            # TODO: self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            
            # ============================================
            # UPDATE SECOND MOMENT (v_t)
            # v_t = β₂ * v_{t-1} + (1 - β₂) * (∇L)²
            # ============================================
            # TODO: self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
            
            # ============================================
            # BIAS CORRECTION
            # m_hat = m_t / (1 - β₁ᵗ)
            # v_hat = v_t / (1 - β₂ᵗ)
            # ============================================
            # TODO: m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            # TODO: v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
            
            # ============================================
            # UPDATE PARAMETER
            # θ = θ - lr * m_hat / (√v_hat + ε)
            # ============================================
            # TODO: model parameter = model parameter - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            pass
    
    def reset(self):
        """Reset optimizer state."""
        # TODO: self.m = {}
        # TODO: self.v = {}
        # TODO: self.t = 0
        pass


def create_optimizer(name: str, learning_rate: float = 0.05, **kwargs) -> Optimizer:
    """
    Factory function to create optimizers.
    
    Args:
        name: Optimizer name ('sgd', 'momentum', 'adam')
        learning_rate: Base learning rate
        **kwargs: Additional optimizer-specific parameters
    
    Returns:
        Optimizer instance
    
    Examples:
        create_optimizer('sgd', 0.05)
        create_optimizer('momentum', 0.05, momentum=0.9)
        create_optimizer('adam', 0.001)
    """
    name = name.lower()
    
    if name == 'sgd':
        return SGD(learning_rate=learning_rate)
    elif name == 'momentum':
        momentum = kwargs.get('momentum', 0.9)
        return Momentum(learning_rate=learning_rate, momentum=momentum)
    elif name == 'adam':
        # TODO: Extract beta1, beta2, eps from kwargs if provided
        # TODO: return Adam(learning_rate, beta1, beta2, eps)
        pass
    else:
        raise ValueError(f"Unknown optimizer: {name}")
