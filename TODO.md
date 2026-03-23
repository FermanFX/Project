# Math4AI Capstone - TODO List

## Ümumi qurulum

- [ ] Python və NumPy, Matplotlib quraşdırın
- [ ] GitHub repository yaradın və branch-based workflow qurun
- [ ] Hər komanda üzvü öz branch-ni yaratsın

---

## Phase 1: Modellerin implementasiyasi

### 1.1 Softmax Regression (`src/models.py`)

- [ ] `SoftmaxRegression._initialize_parameters()` - W və b başlatma
- [ ] `softmax_stable()` - numerik stabil softmax
- [ ] `SoftmaxRegression.forward()` - logits və probabilities hesabla
- [ ] `SoftmaxRegression.backward()` - gradientləri hesabla
  - Formula: `∂L/∂S = (1/n)(P - Y)`
  - `∂L/∂W = (∂L/∂S).T @ X + λW`
  - `∂L/∂b = (∂L/∂S).T @ 1`
- [ ] `SoftmaxRegression.update_parameters()` - parametr yeniləmə
- [ ] `SoftmaxRegression.compute_loss()` - cross-entropy + L2 regularization

### 1.2 One-Hidden-Layer NN (`src/models.py`)

- [ ] `OneHiddenLayerNN._initialize_parameters()` - W1, b1, W2, b2 başlatma
- [ ] `tanh_activation()` - tanh funksiyası
- [ ] `tanh_derivative()` - tanh-in törəməsi: `1 - tanh²(z)`
- [ ] `OneHiddenLayerNN.forward()` - tam forward pass
  - `Z1 = X @ W1.T + b1`
  - `H = tanh(Z1)`
  - `S = H @ W2.T + b2`
  - `P = softmax(S)`
- [ ] `OneHiddenLayerNN.backward()` - BACKPROPAGATION (ƏSAS HİSSƏ!)
  ```
  ∂L/∂S = (1/n)(P - Y)
  ∂L/∂W2 = (∂L/∂S).T @ H
  ∂L/∂b2 = (∂L/∂S).T @ 1
  ∂L/∂Z1 = (∂L/∂S) @ W2 * (1 - H²)
  ∂L/∂W1 = (∂L/∂Z1).T @ X
  ∂L/∂b1 = (∂L/∂Z1).T @ 1
  ```
- [ ] `OneHiddenLayerNN.update_parameters()` - 4 parametr yenilə
- [ ] `OneHiddenLayerNN.compute_loss()` - loss hesabla

### 1.3 Optimizers (`src/optimizers.py`)

- [ ] `SGD.step()` - sadə gradient descent
- [ ] `Momentum.step()` - momentum ilə SGD
  - `v = momentum * v + ∇L`
  - `θ = θ - lr * v`
- [ ] `Adam.step()` - Adam optimizer
  - `m = β1*m + (1-β1)*∇L`
  - `v = β2*v + (1-β2)*(∇L)²`
  - `θ = θ - lr * m / (√v + ε)`

### 1.4 Sanity Checks (REQUIRED!)

- [ ] Gradient check (numerical vs analytical)
- [ ] Probability sum check (hər sətr 1-ə bərabər olmalı)
- [ ] NaN/Inf yoxlanışı
- [ ] Kiçik subset-də loss azalması
- [ ] Kiçik subset-də overfitting

---

## Phase 2: Eksperimentlər

### 2.1 Synthetic Datasets

#### Linear Gaussian
- [ ] Softmax modeli train et
- [ ] NN modeli train et (hidden_width=8)
- [ ] Decision boundary plotları
- [ ] Training dynamics plotları
- [ ] Analiz: Xətti model kifayətdir?

#### Moons
- [ ] Softmax modeli train et
- [ ] NN modeli train et (hidden_width=32)
- [ ] Decision boundary plotları
- [ ] Training dynamics plotları
- [ ] Analiz: Nonlinear model lazımdır?

### 2.2 Digits Benchmark

- [ ] Digits data yüklə (1074 train / 355 val / 368 test)
- [ ] Softmax train et (LR=0.05, 200 epochs, best val checkpoint)
- [ ] NN train et (hidden_width=32, LR=0.05)
- [ ] Training dynamics plotları
- [ ] Test accuracy və loss hesabla

### 2.3 Repeated-Seed Evaluation (Digits)

- [ ] 5 fərqli seed ilə train et (42, 123, 456, 789, 1000)
- [ ] Hər seed üçün test accuracy və loss
- [ ] Orta və standart deviation hesabla
- [ ] 95% CI hesabla: `mean ± 2.776 * std / √5`
- [ ] Nəticələri cədvələ yaz

---

## Phase 3: Ablations (REQUIRED)

### 3.1 Capacity Ablation (Moons)

- [ ] Moons üzərində hidden_width={2, 8, 32} train et
- [ ] Hər biri üçün decision boundary çək
- [ ] Training curves müqayisə et
- [ ] Interpretasiya: Kapasitə artdıqca nə dəyişir?

### 3.2 Optimizer Study (Digits)

- [ ] NN üzərində SGD train et (LR=0.05)
- [ ] NN üzərində Momentum train et (LR=0.05, momentum=0.9)
- [ ] NN üzərində Adam train et (LR=0.001)
- [ ] Training curves müqayisə plotu
- [ ] Hər optimizer-in davranışını təhlil et

### 3.3 Failure Case Analysis

- [ ] Bir failure halı seç:
  - [ ] Seçim: hidden_width=1 (under-capacity)
  - [ ] VƏ YA: çox yüksək LR (instability)
  - [ ] VƏ YA: overfitting (çox epoch)
- [ ] Failure-u təsvir et
- [ ] SƏBƏBİNİ izah et! (sadəcə göstərmək yetərli deyil)

---

## Phase 4: Advanced Track (Birini seçin)

### Track A: PCA/SVD (Input Geometry)

- [ ] SVD ilə eigenvalues hesabla
- [ ] Scree plot çək (ilk 20 component)
- [ ] 2D PCA vizualizasiyası (digits)
- [ ] Softmax at PCA dimensions {10, 20, 40}
- [ ] Nəticələri müqayisə et
- [ ] Input geometry ilə bağlı yorum yaz

### Track B: Confidence & Reliability

- [ ] Confidence = max predicted probability
- [ ] Predictive entropy = -Σ(p * log(p))
- [ ] 5-bin confidence vs accuracy cədvəli/plotu
- [ ] Correct vs Incorrect müqayisəsi
- [ ] Model calibration haqqında yorum

---

## Phase 5: Report

### 5.1 Riyazi İş (REQUIRED)

- [ ] Softmax → Negative Log-Likelihood törəməsi yaz
- [ ] Backpropagation conceptual izahı yaz
- [ ] NN gradient törəmələrini TAM yaz
- [ ] Linear modelin həndəsi interpretasiyası
- [ ] Moons üçün nonlinear ehtiyacın izahı

### 5.2 Report Bölmələri

- [ ] Introduction / Framing
- [ ] Background and Mathematical Setup
- [ ] Methods
- [ ] Implementation Sanity Checks (REQUIRED!)
- [ ] Experiments
- [ ] Advanced Track
- [ ] Discussion
- [ ] Limitations (REQUIRED!)
- [ ] References

### 5.3 Interpretasiya Sualları (Cavablandır!)

- [ ] Xətti model nə vaxt kifayətdir?
- [ ] Hidden layer nə dəyişdirir?
- [ ] Əlavə mürəkkəblik nə vaxt əsassızdır?
- [ ] Failure case nə öyrədir?
- [ ] Repeated-seed nə əlavə edir?

---

## Phase 6: Prezentasiya

- [ ] 10 dəqiqəlik slayd hazırla
- [ ] Hər komanda üzvü danışacaq
- [ ] Əsas sualları təkrar et
- [ ] Q&A üçün hazırlaş

---

## GitHub Tələbləri

- [ ] Meaningful commit messages
- [ ] Branch-based workflow
- [ ] Hər üzvün merged branch-i var
- [ ] README-də setup və reproduction
- [ ] Müvafiq .gitignore

---

## Deadline: Mart 29, 11:59 PM

### Təqdimat siyahısı

- [ ] GitHub repository linki
- [ ] 6-8 səhifəlik PDF report
- [ ] Slaydlar
- [ ] Hər üzvün contribution statement-i
- [ ] 10 dəqiqə prezentasiya + 5 dəqiqə Q&A

---

## Diqqət!

⚠️ PyTorch, TensorFlow, JAX İSTİFADƏ ETMƏYİN!  
⚠️ Sklearn model class-ları İSTİFADƏ ETMƏYİN!  
⚠️ Raw accuracy əsas deyil - riyazi anlayış və interpretasiya əsasdır!

---

## Komanda rolları (Nümunə)

- [ ] Member 1: Softmax + NN implementation + backprop
- [ ] Member 2: Experiments + plots + ablations
- [ ] Member 3: Advanced track + report + presentation
