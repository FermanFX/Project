# Football Match Outcome Prediction

## NAIC - Math4AI Capstone Project

---

## Project Objective

Predict football match outcomes (Home Win / Draw / Away Win) using machine learning and neural network approaches.

---

## Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd starter_pack

# Install dependencies
pip install -r requirements.txt
```

---

## Reproducing Main Experiments

### Part 1: Data Analysis & Visualization (25 points)
```bash
python scripts/run_analysis.py
```

### Part 2: Machine Learning Models (40 points)
```bash
python scripts/run_ml_models.py
```

### Part 3: Neural Network (20 points)
```bash
python scripts/run_neural_network.py
```

### Part 4: Feature Importance (15 points)
```bash
python scripts/run_feature_importance.py
```

### Complete Pipeline
```bash
python scripts/main.py
```

---

## Folder Structure

| Folder | Description |
|--------|-------------|
| `data/` | Raw and processed datasets (football_data.csv) |
| `scripts/` | Executable scripts for running experiments |
| `src/` | Source code for all project modules |
| `figures/` | Generated visualizations and plots |
| `results/` | Model outputs, predictions, metrics |
| `report/` | Final project report and documentation |
| `slides/` | Presentation slides |

### Source Code Modules (`src/`)

- `part1_data_analysis/` - Data loading, cleaning, feature engineering, visualization
- `part2_ml_models/` - Logistic Regression, Decision Tree, Random Forest, XGBoost
- `part3_neural_network/` - Neural network architecture and training
- `part4_feature_importance/` - SHAP analysis and feature importance visualization

---

## Team Contributions

| Team Member | Responsibilities |
|-------------|-------------------|
| Member 1 | Data preprocessing & feature engineering |
| Member 2 | Machine learning model implementation |
| Member 3 | Neural network architecture & training |
| Member 4 | Feature importance analysis & visualization |

---

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies
