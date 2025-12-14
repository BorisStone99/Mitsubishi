# Mitsubishi
# Mitsubishi Scripts

This directory contains a collection of Python scripts for machine learning optimization, analysis, and interpretability. The scripts are designed to work with neural network models (scikit-learn and PyTorch) and perform various data analysis tasks such as correlation analysis, Pareto frontier visualization, and sensitivity analysis.

## Scripts Overview

### 1. BayesOptimisation.py
**Purpose:** Optimizes hyperparameters for a Multi-Layer Perceptron (MLP) regressor using Bayesian Optimization.
**Key Features:**
- Uses `scikit-optimize` (skopt) for efficient parameter search.
- Optimizes architecture (hidden layers), alpha, max_iter, and tolerance.
- **Workarounds:** Implements specific strategies to handle a known issue in `skopt` with tuple parameters (like `hidden_layer_sizes`), including:
    - Separate optimization runs for each architecture.
    - Integer encoding wrapper.
    - Single-layer architecture optimization.

### 2. GridOptimisation.py
**Purpose:** Performs hyperparameter optimization for an MLP regressor using a Randomized Grid Search approach.
**Key Features:**
- Generates a comprehensive grid of hyperparameters.
- Randomly samples a subset of combinations to test (Random Search).
- Uses Cross-Validation (CV) to evaluate model performance.
- Exports results to a CSV file for further analysis.
- Reports R2 score and MAE on a holdout validation set.

### 3. CorrelationHeatMap.py
**Purpose:** Comprehensive tool for analyzing and visualizing feature correlations in a dataset.
**Key Features:**
- Calculates Pearson, Spearman, and Kendall correlation coefficients.
- **Visualization:** Generates detailed heatmaps, including full matrices, clustered heatmaps (dendrograms), and high-correlation networks.
- **Analysis:** Identifies high-correlation pairs and provides feature selection recommendations to reduce redundancy.
- Assesses overall correlation quality and distribution.

### 4. Pareto.py
**Purpose:** Analyzes and visualizes Pareto efficiency for multi-objective minimization problems.
**Key Features:**
- Identifies strict Pareto-optimal points and ε-Pareto (extended) points.
- **3D Visualization:** Provides advanced 3D interactive plots of the Pareto frontier.
- Visualizes the trade-off surface using Convex Hulls and interpolation.
- Includes functions for zoomed-in detailed analysis of specific Pareto regions.

### 5. SHAP.py
**Purpose:** A suite of model interpretability tools to explain model predictions.
**Key Features:**
- **Custom Implementations:** Includes custom classes for:
    - **Shapley Values:** `CustomShapleyExplainer`
    - **Permutation Importance:** `PermutationExplainer`
    - **LIME:** `LIMEExplainer` (Local Interpretable Model-agnostic Explanations)
- **Comparison:** The `FeatureImportanceAnalyzer` class calculates and visualizes comparisons between these different interpretability methods.
- **Interactions:** Analyzes and plots pairwise feature interactions.

### 6. SensitivityAnalysis.py
**Purpose:** Performs sensitivity analysis on trained PyTorch models to understand feature importance and model robustness.
**Key Features:**
- **Methods:** Implements three sensitivity measures:
    - **Integrated Gradients**
    - **Perturbation Sensitivity**
    - **Feature Ablation Sensitivity**
- **Model Handling:** Load PyTorch models dynamically, handling input size mismatches by padding or trimming input features as needed.
- **Batch Processing:** efficiently processes large datasets in batches.
- **Reporting:** and saves detailed CSV reports including raw sensitivity scores and summary rankings.

## Setup & Usage Notes
- **Dependencies:** Ensure you have the required libraries installed (`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`, `skopt`, `torch`).
- **Data Paths:** Some scripts may contain hardcoded file paths (e.g., `C:\Users\...`). Please review the data loading sections at the beginning of each script and update the paths to point to your local data files.
