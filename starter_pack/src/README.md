# `src/`

Text to be changed (maybe) later:

# **models.py**

The provided `models.py` file contains the core logic for the machine learning architecture. It defines how data flows from input to prediction through the **Forward Pass** and how the system learns from errors via the **Backward Pass**.

## **Softmax Regression**

The `SoftmaxRegression` class implements a linear classifier. This model attempts to find straight-line boundaries (hyperplanes) to separate different categories within the data.

* **Operation:** The model multiplies the input features by a weight matrix and adds a bias vector ($Wx + b$). These results are raw "scores" known as **logits**.
* **The Softmax Function:** The model converts these raw scores into probabilities that sum to **100%** across all classes.
* **Numerical Stability:** The `softmax_stable` function subtracts the maximum logit value before exponentiation. This prevents the computer from encountering "NaN" (Not a Number) errors caused by calculating $e^{x}$ for very large numbers.



## **One-Hidden-Layer Neural Network**

While Softmax Regression is restricted to straight lines, the `OneHiddenLayerNN` can learn complex, curved boundaries. It achieves this by inserting a **Hidden Layer** equipped with a non-linear activation function.

### **The Architecture**
1.  **Input Layer:** Receives the raw data, such as image pixels.
2.  **Hidden Layer:** Utilizes the `tanh` activation function. This step "warps" the mathematical space so that data points that were mixed together become linearly separable.
3.  **Output Layer:** Uses a final Softmax layer to generate the final class probabilities.



## **Backpropagation Mechanics**

The `backward` methods represent the phase where the model actually "learns." This is a direct implementation of the **Chain Rule** from calculus to determine how much each weight contributed to the total error.

### **The Tanh Derivative Shortcut**
To update the weights in the first layer ($W_1$), the model calculates how the loss changes with respect to the hidden layer. The implementation uses a specific mathematical identity:
$$\frac{d}{dz} \tanh(z) = 1 - \tanh^2(z)$$
Since the model already calculated $H = \tanh(z)$ during the forward pass, the derivative is computed as `1 - H**2`. This optimization allows the system to run faster by avoiding redundant exponential calculations.



## **Parameter Initialization**

Within `OneHiddenLayerNN`, the weights are initialized using a specific scaling factor: `np.sqrt(2.0/input_dim)`.

This approach, known as **He Initialization**, is used for several reasons:
* **Preventing Vanishing Gradients:** If weights are too small, the learning signal dies out before reaching the earlier layers.
* **Preventing Exploding Gradients:** If weights are too large, the signal grows exponentially, leading to numerical overflow.
* **Balance:** The goal is to maintain a consistent variance in activations across all layers, ensuring the network can begin learning efficiently from the first epoch.



### **Architectural Comparison**

| Feature | Softmax Regression | One-Hidden-Layer NN |
| :--- | :--- | :--- |
| **Complexity** | Linear (Simple) | Non-Linear (Complex) |
| **Parameters** | $W, b$ | $W_1, b_1, W_2, b_2$ |
| **Activation** | Identity (None) | Tanh |
| **Ideal Use Case** | Linearly separable data | Complex patterns (e.g., Image Recognition) |


---


# **optimizers.py**

This `optimizers.py` file contains the logic for updating models' weights. Think of the loss function as a mountain range and the optimizer as the strategy used to find the lowest valley.

Here is the breakdown of the three optimization strategies implemented:


## **SGD (Stochastic Gradient Descent)**

It moves the parameters directly in the opposite direction of the gradient.

* **Logic**: If the gradient is positive (pointing uphill), subtract a small portion of it to move downhill.
* **Formula**: $\theta = \theta - \eta \nabla L$ (where $\eta$ is the learning rate).
* **Analogy**: Taking a single step directly downhill. It’s simple, but if the "hill" is a narrow ridge, SGD tends to bounce back and forth across the ridge rather than following it down to the valley.


## **Momentum**

Momentum addresses the "bouncing" problem of SGD by adding a sense of physical inertia.

* **Logic**: Instead of just looking at the current gradient, it maintains an **acceleration** ($v$). It adds a fraction of the *previous* step to the *current* step. Think of it as in this situation:
A car is stuck in the mud. At first, it is hard for the car to get out of it. But as the car's wheels
spin, it gradually accumulates the "thing" to get out of the mud, in other words, it becomes easier and
easier to get out of the mud.
* **Benefit**: This smooths out oscillations. In directions where the gradient keeps changing sign (bouncing), the velocity cancels out. In directions where the gradient is consistent, the velocity builds up, speeding up convergence.
* **Analogy**: A heavy ball rolling down a hill. It builds up speed and isn't easily diverted by small bumps.

---

## **Adam (Adaptive Moment Estimation)**

Adam is currently the industry standard in deep learning. It combines the Momentum and RMSProp.

### **The Two "Moments"**
1.  **First Moment ($m_t$)**: Similar to Momentum, it tracks the average direction of the gradients.
2.  **Second Moment ($v_t$)**: It tracks the average *squared* magnitude of the gradients. This tells the optimizer how much the gradient is vibrating.

### **Why it's "Adaptive"**
Adam gives each individual weight its own learning rate.
* If a weight has a very large, erratic gradient, Adam scales its learning rate **down** to stay stable.
* If a weight has a tiny, consistent gradient, Adam scales its learning rate **up** to move faster.

### **Bias Correction**
In the code, there are `m_hat` and `v_hat`. These are used because $m$ and $v$ are initialized at zero. Without correction, the optimizer would be very sluggish during the first few steps. The division by $(1 - \beta^t)$ "boosts" these values at the start of training. Also, it prevents division by zero.

Although Adam is the standard and is used widely, there is a certain critical point to consider.
We cannot use Adam for every problem in deep learning. It has its limitations.

Adaptive optimization methods, which perform local optimization with a metric constructed from the history of iterates, are becoming increasingly popular for training deep neural networks. Examples include AdaGrad, RMSProp, and Adam. We show that for simple overparameterized problems, adaptive methods often find drastically different solutions than gradient descent (GD) or stochastic gradient descent (SGD). We construct an illustrative binary classification problem where the data is linearly separable, GD and SGD achieve zero test error, and AdaGrad, Adam, and RMSProp attain test errors arbitrarily close to half. We additionally study the empirical generalization capability of adaptive methods on several state-of-the-art deep learning models. We observe that the solutions found by adaptive methods generalize worse (often significantly worse) than SGD, even when these solutions have better training performance. These results suggest that practitioners should reconsider the use of adaptive methods to train neural networks.
Source: https://doi.org/10.48550/arXiv.1705.08292

## **Implementation Insight: The Factory Pattern**

The `create_optimizer` function at the bottom is a **Factory Pattern**. It allows a user to simply pass a string like `"adam"` or `"sgd"` to set up the training environment. This makes the library much more flexible, as you can swap out the entire optimization strategy by changing a single word in your main configuration.

### **Summary Comparison**

| Optimizer | Speed | Stability | Tuning |
| :--- | :--- | :--- | :--- |
| **SGD** | Slow | Low (oscillates) | High (very sensitive to LR) |
| **Momentum** | Medium | Medium | Medium |
| **Adam** | Fast | High | Low (defaults usually work) |


---


# **trainer.py**

## **Data Management: Minibatches & One-Hot Encoding**

Training a model on all data at once is computationally expensive and often leads to poor convergence. The `Trainer` handles this via:

* **Shuffling**: By calling `np.random.shuffle(indices)`, the trainer ensures the model doesn't learn patterns based on the order of the data.
* **Minibatches**: It breaks the dataset into small chunks (e.g., 64 examples). This "stochastic" approach adds a bit of noise to the training, which actually helps the model escape local minima.
* **One-Hot Encoding**: Since the Softmax loss function requires a probability distribution, the `_one_hot` method converts integer labels (like `3`) into vectors (like `[0, 0, 0, 1, 0...]`).

---

## **The Training Loop Logic**

The `train` method follows a strict, repeating cycle for every epoch:

1.  **Train Epoch**: It iterates through every minibatch.
    * **Forward Pass**: The model makes a prediction.
    * **Backward Pass**: The model calculates the error (gradient).
    * **Regularization**: The trainer adds a penalty ($L_2$ regularization) to the gradients to keep the weights small and prevent overfitting.
    * **Optimizer Step**: The optimizer (SGD, Adam, etc.) updates the actual weights.
2.  **Validation**: After one full pass through the training data, the model is tested on "unseen" validation data.
3.  **Checkpointing**: If the model performs better on the validation set than ever before, the trainer saves a "snapshot" of the weights.



## **Softmax vs. Neural Networks**

The code uses **Inheritance**. There is a base `Trainer` class, but it branches into two specific versions:

### **SoftmaxTrainer**
* **Focus**: Single layer of weights ($W$ and $b$).
* **Complexity**: Low. It simply maps inputs directly to output classes.

### **NNTrainer**
* **Focus**: Two layers of weights ($W_1, b_1$ and $W_2, b_2$).
* **Complexity**: Higher. It must manage a **cache** from the forward pass because the backward pass through a neural network requires knowing what the intermediate "hidden" activations were.



## **The `TrainingHistory` Container**

This is a simple `dataclass` that acts as a "flight recorder." It stores the loss and accuracy for both training and validation at every epoch. This is what the `visualization.py` file uses to plot learning curves, as it keeps track of losses and metrics for train/val/test.



## **Regularization Gradient**

In both trainers, there is a line `grad_W_reg = grad_W + self.reg_lambda * self.model.W`. This is the implementation of **Weight Decay**. Mathematically, if the loss function includes an $L_2$ penalty:

$$L_{total} = L_{cross-entropy} + \frac{\lambda}{2} \|W\|^2$$

The derivative with respect to $W$ becomes:

$$\frac{\partial L_{total}}{\partial W} = \frac{\partial L_{ce}}{\partial W} + \lambda W$$

This effectively "pulls" the weights toward zero during every update, preventing any single feature from having too much influence.

---

# **evaluation.py**



## **Statistical Rigor: The 95% Confidence Interval**
In science, training a model once isn't enough it might be just a lucky experiment with the random initialization (the "seed"). The `RepeatedSeedResult` class implements a formal benchmarking protocol:
* **The 5-Seed Rule:** The exact same experiment is run 5 times with different starting points.
* **The CI Formula:** It calculates the **95% Confidence Interval**.
$$CI = \mu \pm 2.776 \times \frac{\sigma}{\sqrt{n}}$$
> **Why 2.776?** This is the "t-critical" value for a sample size of 5. It means that if the experiment
was repeated 100 times, the true average performance would fall within this range 95 times.



## **Model Reliability: Calibration & Entropy**
The `Evaluator` class goes beyond simple "Accuracy" to look at **how sure** the model is:
* **Confidence:** The maximum probability assigned to a class. If a model predicts "Dog" with 0.99 probability, it is highly confident.
* **Entropy:** A measure of "chaos" or uncertainty. High entropy means the model is "confused" and spreading its probability across many classes.
This term from the information theory is also used in cross-entropy. When calculating cross entropy, we use KL-divergence, which returns the "similarity score" of two probability distributions. The preferred
outcome would be to give higher probability to actual class, and KL-divergence measures that "thing".
* **Confidence Bins:** The `confidence_by_bin` method checks **calibration**. If a model says it's 80% confident about a group of images, it *should* get exactly 80% of them right. If it only gets 40% right, the model is "overconfident."

A calibrated probability is a probability estimate from a machine learning model that accurately reflects the true likelihood of an event occurring. In other words, if a well-calibrated classification model predicts a 70% chance of an event happening across multiple instances, we expect the event to occur in about 70% of those cases.
For example, if a calibrated weather model predicts a 30% chance of rain for 100 different days, we would expect it to rain on about 30 of those days. Similarly, in a binary classification task, if a calibrated model assigns a probability of 0.8 to the positive class for 100 samples, approximately 80 of those samples should actually belong to the positive class.
In summary, the output from a classifier with calibrated probability should match the true probability distribution.
Source: https://www.blog.trainindata.com/probability-calibration-in-machine-learning/



## **Sanity Checks**
Before trusting any results, the code runs three vital health checks:

### **A. Gradient Checking (`gradient_check`)**
This is the most important debugging tool in neural network development.
* **Analytical Gradient:** The fast math using calculus (backprop).
* **Numerical Gradient:** A slow, "brute force" estimation using the limit definition of a derivative:
$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}$$
If these two values don't match (relative error $> 10^{-5}$), there is a bug in implemented calculus or code.


### **B. Probability Summation**
Ensures the Softmax function is working. If the probabilities for a single image sum to **1.000001** or **0.999**, the model is leaking "probability mass," likely due to numerical instability.


### **C. NaN/Inf Detection**
In deep learning, "Exploding Gradients" can cause numbers to become so large that the computer turns them into `Inf` (Infinity) or `NaN` (Not a Number). This check catches those silent failures before they ruin your training history.
It is important for gradients to not be too large or small, because that makes them intractable when calculating derivatives.


## **Predictive Utilities**
Finally, the class provides standard methods like `predict_proba` (returns the percentages) and `predict` (returns the final prediction), abstracting away the differences between the simple Softmax model and the complex Neural Network.



### **Summary of Evaluation Metrics**
| Metric | What it tells you | High Value is... |
| :--- | :--- | :--- |
| **Accuracy** | How often is it right? | Good |
| **Cross-Entropy** | How "far" are predictions from the truth? | Bad (Lower is better) |
| **Confidence** | How certain is the model? | Context-dependent |
| **Entropy** | How uncertain/confused is the model? | Bad (Higher uncertainty) |


---


# **visualization.py**


## **Spatial & Structural Visualizations**


### **Decision Boundaries**
This tool reveals how a classifier "thinks" about the input space.
* **Mechanism**: It creates a dense grid of points (e.g., $200 \times 200$) across the feature range, predicts the class for every single point, and colors the background accordingly.
* **Insight**: You can see if a model is making simple linear splits (like Softmax) or complex, non-linear "curvy" boundaries (like a Neural Network).


### **PCA (Principal Component Analysis) Suite**
For high-dimensional data like images (e.g., handwritten digits), these plots help you "see" the data in 2D:
* **Scree Plot**: Shows how much information (variance) each principal component captures. You look for the "elbow" in the graph to decide how many components are actually necessary.
* **2D Visualization**: Projects complex data onto two axes. If the "3"s and "8"s cluster near each other but far from "1"s, you know which classes the model will likely struggle to distinguish.


## **Performance & Reliability Diagnostics**

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



## **Experimental Comparison Tools**

### **Optimizer Comparison**
This plots different training algorithms (SGD, Momentum, Adam) on the same graph. It allows to visually confirm that Adam typically converges much faster than basic SGD.

### **Capacity Ablation**
This generates a side-by-side comparison of decision boundaries as the "capacity" (hidden width) of the neural network changes.
* **Small Width**: Results in simple, smoother boundaries (potential underfitting).
* **Large Width**: Results in highly complex, wiggly boundaries (potential overfitting).

### **Repeated Seed Reporting**
Since neural networks are sensitive to their random starting weights, this function plots the results of multiple runs with **error bars**. This ensures that if Model A is "better" than Model B, it's a statistically significant lead and not just a "lucky" random initialization.