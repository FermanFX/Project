"""
PART 4.1: FEATURE IMPORTANCE ANALYSIS (8 points)
==================================================
SHAP and feature importance - ALGORITHM parts marked with TODO
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


def calculate_tree_importance(model, feature_names):
    """
    Get feature importance from tree-based models.
    
    ALGORITHM:
    TODO: Implement Gini-based feature importance:
    
    For each feature:
    TODO: importance = sum(Gini_decrease_at_splits_using_this_feature) / total_Gini_decrease
    
    Alternative (Mean Decrease Impurity):
    TODO: importance_i = sum(N_t * Gini_t) for nodes where feature i is used
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def calculate_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10):
    """
    Calculate permutation importance.
    
    ALGORITHM:
    TODO: Implement permutation importance:
    
    1. Calculate baseline accuracy: baseline_acc
    2. For each feature:
       TODO: Shuffle values of feature i
       TODO: Calculate new accuracy: shuffled_acc
       TODO: importance_i = baseline_acc - shuffled_acc
       3. Repeat n_repeats times and average
    """
    from sklearn.inspection import permutation_importance
    
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, 
                                    random_state=42, n_jobs=-1)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)
    
    return importance_df


def calculate_shap_values(model, X_train, X_test, feature_names):
    """
    Calculate SHAP values for model interpretability.
    
    ALGORITHM:
    TODO: SHAP (SHapley Additive exPlanations) values:
    
    1. Background data: Use X_train to compute expected value
    2. For each sample x:
       TODO: Compute SHAP value for each feature i
       
    SHAP value formula:
    TODO: phi_i = sum_over_subsets_S[|S|! * (n-|S|-1)! / n!] * 
                    [f(S union {i}) - f(S)]
    
    Where:
    - S: subset of features
    - f(S): model prediction with features in S
    - n: total number of features
    
    Interpretation:
    TODO: prediction = base_value + sum(SHAP_values)
    """
    explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X_test)
    
    shap_values_dict = {}
    for i, name in enumerate(feature_names):
        if isinstance(shap_values, list):
            shap_values_dict[name] = [sv[:, i] for sv in shap_values]
        else:
            shap_values_dict[name] = shap_values[:, i]
    
    return shap_values, explainer


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations).
    
    ALGORITHM:
    TODO: Implement local surrogate model:
    
    1. For prediction x:
       TODO: Generate perturbed samples around x
       2. Weight samples by distance to x
       3. Train simple interpretable model (linear) on perturbed samples
       4. Extract local explanation
    """
    
    def __init__(self, model, feature_names, class_names=None):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ['Home Win', 'Draw', 'Away Win']
    
    def explain_instance(self, x, num_features=10):
        """
        Explain single prediction.
        
        ALGORITHM:
        TODO: 
        1. Generate perturbed samples
           TODO: Randomly perturb features, especially important ones
        2. Get model predictions for perturbations
        3. Weight by similarity to original
           TODO: weight = exp(-distance^2 / kernel_width^2)
        4. Fit weighted linear model
        5. Return coefficients as importance scores
        """
        # TODO: Implement LIME explanation
        pass


def feature_interaction_analysis(shap_values, X_test, feature_names, feature1, feature2):
    """
    Analyze interaction between two features using SHAP.
    
    ALGORITHM:
    TODO: Compute interaction effect:
    
    interaction_strength = correlation between(SHAP_1, SHAP_2)
    
    Or compute interaction SHAP values:
    TODO: phi_ij = SHAP_ij where both features change together
    """
    idx1 = feature_names.index(feature1)
    idx2 = feature_names.index(feature2)
    
    shap1 = shap_values[:, idx1] if shap_values.ndim == 2 else shap_values[0][:, idx1]
    shap2 = shap_values[:, idx2] if shap_values.ndim == 2 else shap_values[0][:, idx2]
    
    correlation = np.corrcoef(shap1, shap2)[0, 1]
    
    return correlation


def aggregate_shap_values(shap_values, feature_names):
    """
    Aggregate SHAP values across samples.
    
    ALGORITHM:
    TODO: Compute mean absolute SHAP value per feature:
    mean_importance_i = mean(|SHAP_value_i|) across all samples
    
    This gives global feature importance from SHAP perspective.
    """
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    
    if shap_values.ndim == 3:
        shap_values = np.mean(shap_values, axis=0)
    
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': mean_abs_shap
    }).sort_values('shap_importance', ascending=False)
    
    return importance_df
