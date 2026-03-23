# Math4AI Capstone - Layihə icmal

## Layihə haqqında

**Başlıq:** "From Linear Scores to a Single Hidden Layer: A Mathematical Study of Simple Learning Systems"

**Sual:** Bir gizli qatlı qeyri-xətti klassifikator xətti qərar qaydasına nə vaxt yaxşılaşdırır və nə vaxt əlavə mürəkkəblik lazımsızdır?

**Qurum:** National AI Center - AI Academy

**Deadline:** Mart 30, 11:59 PM

---

## Layihə strukturu

```
starter_pack/
├── src/                    # Kod skeletonları
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   ├── models.py           # Softmax & NN modelləri (SKELETON)
│   ├── optimizers.py       # SGD, Momentum, Adam
│   ├── trainer.py           # Training loop
│   ├── evaluation.py        # Metrics & evaluation
│   └── visualization.py     # Plot funksiyaları
├── data/                   # Dataset faylları
├── figures/                # Plot输出的
└── results/                # Nəticələr
```

---

## Tələb olunan modellər

### Model 1: Softmax Regression (Baseline)
Xətti klassifikator - riyazi olaraq:
```
s(x) = Wx + b              # Score functions (logits)
p_j(x) = exp(s_j) / Σexp(s_ℓ)  # Softmax probabilities
L = -log(p_y(x))           # Cross-entropy loss
```

### Model 2: One-Hidden-Layer Neural Network
Bir gizli qatlı neural şəbəkə:
```
Z₁ = XW₁ᵀ + b₁            # Affine transformation
H = tanh(Z₁)              # Hidden activations
S = HW₂ᵀ + b₂             # Output scores
P = softmax(S)             # Probabilities
```

**Diqqət:** Tələb olunan aktivasiya `tanh`-dır (ReLU yox!)

---

## Dataset-lər

| Dataset | Nümunə sayı | Xüsusiyyət ölçüsü | Siniflər |
|---------|------------|---------------------|----------|
| Linear Gaussian | 400 (240/80/80) | 2 | 2 |
| Moons | 400 (240/80/80) | 2 | 2 |
| Digits | 1797 (1074/355/368) | 64 | 10 |

---

## Tələb olunan eksperimentlər

### Core Experiments (Hamı üçün)
1. ✅ Linear Gaussian: Hər iki modelin müqayisəsi + decision boundary
2. ✅ Moons: Hər iki modelin müqayisəsi + decision boundary
3. ✅ Digits: Hər iki modelin müqayisəsi

### Required Ablations (Hamı üçün)
1. **Capacity Ablation:** Moons üzərində hidden width {2, 8, 32} müqayisəsi
2. **Optimizer Study:** Digits üzərində SGD, Momentum, Adam müqayisəsi
3. **Failure Case Analysis:** Bir uğursuzluq halının təhlili

### Repeated-Seed Evaluation (Digits üçün)
- 5 fərqli random seed ilə təlim
- Ortalama və 95% confidence interval hesablanmalı

---

## Advanced Track (Birini seçin)

### Track A: PCA/SVD və Input Geometry
- Scree plot
- 2D PCA vizualizasiyası
- PCA dimension {10, 20, 40} müqayisəsi

### Track B: Prediction Confidence & Reliability
- Confidence (max probability) analizi
- Predictive entropy
- 5-bin confidence vs accuracy cədvəli

---

## Hiperparametrlər (PDF-dən)

| Parametr | Dəyər |
|----------|-------|
| Hidden width (default) | 32 |
| L2 regularization (λ) | 10⁻⁴ |
| Batch size | 64 |
| Epoch budget | 200 |
| LR (Softmax/SGD) | 0.05 |
| LR (Adam) | 0.001 |
| Momentum | 0.9 |
| Adam β₁, β₂ | 0.9, 0.999 |

---

## Qadağan olunmuş alətlər

❌ PyTorch, TensorFlow, JAX  
❌ autograd  
❌ scikit-learn model classes

✅ İcazəli: Python, NumPy, Matplotlib, GitHub

---

## Nəticələrin qiymətləndirilmə meyarları

| Kateqoriya | Çəki |
|------------|------|
| Riyazi anlayış və törəmələr | 20% |
| İmplementasiya düzgünlüyü | 16% |
| Eksperimental dizayn | 18% |
| İnterpretasiya və təhlil | 18% |
| Report keyfiyyəti | 10% |
| Prezentasiya | 10% |
| Repository keyfiyyəti | 8% |

---

## Nəticələrin təqdimatı

- GitHub repository linki
- 6-8 səhifəlik PDF report
- 10 dəqiqəlik texniki prezentasiya + 5 dəqiqə Q&A
- Hər komanda üzvü danışmalıdır

---

## Yorğunluq haqqında

**Sual:** Plot-u özüm etdim - bəs indi necə işləyim?

**Cavab:** `starter_pack/src/` qovluğunda skeleton fayllar var. Hər funksiyanın docstring-ində nə gözlənildiyi yazılıb. Dostlarınla bölüş:

```python
from src.models import SoftmaxRegression, OneHiddenLayerNN
from src.optimizers import SGD, Momentum, Adam
from src.trainer import SoftmaxTrainer, NNTrainer

# Model yaradın
model = SoftmaxRegression(input_dim=64, num_classes=10)

# Təlim
trainer = SoftmaxTrainer(model, optimizer=SGD(lr=0.05))
history = trainer.train(X_train, y_train, X_val, y_val)

# Qrafiklər üçün visualization.py istifadə edin
from src.visualization import plot_decision_boundary, plot_training_dynamics
```
