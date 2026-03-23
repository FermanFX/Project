# `src/`

Text to be changed (maybe) later:

# **File 1: models.py - Core Neural Network Implementations**

### **Softmax Regression (Multiclass Logistic Regression)**

#### **Mathematical Foundation**
Softmax regression generalizes logistic regression to multiple classes. For an input vector $x \in \mathbb{R}^d$, we compute:
1.  **Logits**: $s_j = W_j \cdot x + b_j$ for $j = 1,\dots,k$ classes.
2.  **Probabilities**: $p_j = \frac{e^{s_j}}{\sum_{l=1}^k e^{s_l}}$
3.  **Prediction**: $\hat{y} = \arg\max_j p_j$



#### **Loss Function**
The cross-entropy loss for a single example with true class $y$:
$$L = -\log(p_y) = -\log\left(\frac{e^{s_y}}{\sum_l e^{s_l}}\right)$$

For a batch of $n$ examples with one-hot encoding $Y \in \{0,1\}^{n \times k}$:
$$L = -\frac{1}{n}\sum_{i=1}^n \sum_{j=1}^k Y_{ij} \log(p_{ij}) + \frac{\lambda}{2}\|W\|^2$$

#### **Gradient Derivation**
The key insight is that for softmax with cross-entropy, the gradient simplifies beautifully. By computing the derivative with respect to logits $s_j$, we find $\frac{\partial L}{\partial s_j} = p_j - Y_j$. For the true class, the derivative is $p_j - 1$; for false classes, it is simply $p_j$. We then use the chain rule to derive the weight gradients and add regularization.

### **OneHiddenLayerNN - Neural Network with Tanh Activation**

#### **Forward Pass Architecture**
The network consists of an input layer, a hidden layer using the $\tanh$ activation function, and an output layer utilizing softmax.

#### **Why Tanh?**
Tanh activation has desirable properties:
* **Zero-centered**: Outputs range $(-1, 1)$, helping with gradient flow.
* **Smooth derivative**: $\tanh'(z) = 1 - \tanh^2(z)$.
* **Saturating**: Prevents exploding activations.

#### **Backpropagation Derivation**
This is the most mathematically intensive part, involving the backpropagation of error from the output layer through the hidden layer and finally to the input layer. This requires calculating the gradients of the loss with respect to the weights and biases of both layers using the chain rule and the derivative of the $\tanh$ function.

#### **Numerical Stability**
To prevent numerical overflow (e.g., $e^{1000}$), the implementation uses a "stable" softmax. By subtracting the maximum logit from all logits before exponentiation, we ensure the largest exponent is $e^0 = 1$.

---

# **File 2: optimizers.py - Optimization Algorithms**

### **SGD (Stochastic Gradient Descent)**
The simplest optimizer, where parameters are updated by moving in the opposite direction of the gradient scaled by a learning rate $\eta$:
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

### **Momentum**
Introduces "velocity" to accelerate convergence and dampen oscillations:
$$v_t = \beta v_{t-1} + \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

### **Adam (Adaptive Moment Estimation)**
Adam combines Momentum (first moment) with per-parameter learning rates (second moment). It is highly effective because it handles sparse gradients well and is generally robust to hyperparameter choices.

---

# **File 3: trainer.py - Training Infrastructure**

### **Training Loop Structure**
The trainer manages the lifecycle of the model:
1.  Shuffling and creating minibatches.
2.  Executing the forward pass.
3.  Computing the loss.
4.  Executing the backward pass to find gradients.
5.  Updating parameters via the optimizer.
6.  Evaluating on validation data and checkpointing the best-performing version.

### **Minibatch Creation**
Shuffling the data every epoch is crucial. It ensures that the model doesn't learn spurious patterns based solely on the order in which data was collected or stored.

### **Checkpointing Strategy**
The system saves the model when it achieves the best validation loss. This prevents "overfitting," where a model begins to memorize noise in the training data at the expense of general performance.

---

# **File 4: evaluation.py - Model Evaluation**

### **Statistical Significance Testing**
Because neural network training is stochastic (due to random initialization), the library reports confidence intervals. Using a t-distribution for small sample sizes (e.g., 5 different random seeds) provides a mathematically sound way to say, "We are 95% confident the true accuracy falls within this range."

### **Gradient Checking**
This is a critical debugging tool. It compares the "analytical" gradients (derived via calculus) against "numerical" gradients (approximated by slightly nudging weights and measuring the change in loss). If they match, the backpropagation math is correct.

### **Model Calibration**
This measures if a model's "confidence" matches its actual accuracy. An overconfident model might predict a class with 99% probability but only be right 70% of the time—a dangerous trait in high-stakes fields like medicine or self-driving cars.


---

# **File 5: visualization.py - Visualization helpers**

## **1. Spatial & Structural Visualizations**

### **Decision Boundaries**
This tool reveals how a classifier "thinks" about the input space.
* **Mechanism**: It creates a dense grid of points (e.g., $200 \times 200$) across the feature range, predicts the class for every single point, and colors the background accordingly.
* **Insight**: You can see if a model is making simple linear splits (like Softmax) or complex, non-linear "curvy" boundaries (like a Neural Network).

### **PCA (Principal Component Analysis) Suite**
For high-dimensional data like images (e.g., handwritten digits), these plots help you "see" the data in 2D:
* **Scree Plot**: Shows how much information (variance) each principal component captures. You look for the "elbow" in the graph to decide how many components are actually necessary.
* **2D Visualization**: Projects complex data onto two axes. If the "3"s and "8"s cluster near each other but far from "1"s, you know which classes the model will likely struggle to distinguish.



---

## **2. Performance & Reliability Diagnostics**

### **Training Dynamics**
This creates a dual-plot or dual-axis view of **Loss** and **Accuracy** over time.
* **Overfitting Check**: If the training loss continues to drop but the validation loss starts to rise, the model is starting to "memorize" the training data rather than "learning" patterns that generalize.



### **Confidence vs. Accuracy (Reliability Diagram)**
Although we didn't use these visualizations, they are critical tools for track B.
* **Calibration**: A well-calibrated model that says it is "90% confident" should be right exactly 90% of the time.
* **Visualization**: It bins predictions by confidence and compares them to actual accuracy. If the bars are below the diagonal line, your model is **overconfident**.

### **Confusion Matrix**
A heatmap showing exactly where the model is failing.
* **Diagonal**: Represents correct predictions.
* **Off-diagonal**: High values here indicate specific "confusions" (e.g., the model frequently mistakes "4" for "9").

---

## **3. Experimental Comparison Tools**

### **Optimizer Comparison**
This plots different training algorithms (SGD, Momentum, Adam) on the same graph. It allows to visually confirm that Adam typically converges much faster than basic SGD.

### **Capacity Ablation**
This generates a side-by-side comparison of decision boundaries as the "capacity" (hidden width) of the neural network changes.
* **Small Width**: Results in simple, smoother boundaries (potential underfitting).
* **Large Width**: Results in highly complex, wiggly boundaries (potential overfitting).

### **Repeated Seed Reporting**
Since neural networks are sensitive to their random starting weights, this function plots the results of multiple runs with **error bars**. This ensures that if Model A is "better" than Model B, it's a statistically significant lead and not just a "lucky" random initialization.

---

## **Mathematical Concepts Demonstrated**

1.  **Probability Distributions**: Softmax and Cross-entropy.
2.  **Optimization Theory**: Steepest descent, momentum, and adaptive learning rates.
3.  **Statistical Inference**: Confidence intervals and variance measurement.
4.  **Information Theory**: Using KL divergence principles to measure the distance between true and predicted labels.