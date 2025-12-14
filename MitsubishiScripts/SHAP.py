import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import torch
import torch.nn as nn
import warnings
from pathlib import Path
import joblib
from scipy import stats
from collections import defaultdict
import matplotlib.patches as mpatches
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')

class PyTorchModelWrapper:
    """Wrapper to make PyTorch models compatible with interpretability methods"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict(self, X):
        """Predict method for interpretability compatibility"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if len(outputs.shape) == 1:
                return outputs.cpu().numpy()
            else:
                return outputs.cpu().numpy()[:, 0] if outputs.shape[1] == 1 else outputs.cpu().numpy()
    
    def predict_proba(self, X):
        """Predict probabilities for interpretability compatibility"""
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if len(outputs.shape) == 1:
                probs = torch.sigmoid(outputs)
                return np.column_stack([1 - probs.cpu().numpy(), probs.cpu().numpy()])
            else:
                if outputs.shape[1] == 1:
                    probs = torch.sigmoid(outputs)
                    return np.column_stack([1 - probs.cpu().numpy(), probs.cpu().numpy()])
                else:
                    return torch.softmax(outputs, dim=1).cpu().numpy()

class CustomShapleyExplainer:
    """Custom implementation of Shapley value calculation"""
    
    def __init__(self, model, background_data, max_evals=1000):
        self.model = model
        self.background_data = background_data
        self.max_evals = max_evals
        self.baseline_value = self._calculate_baseline()
        
    def _calculate_baseline(self):
        """Calculate baseline prediction (expected value)"""
        if hasattr(self.model, 'predict_proba'):
            baseline_preds = self.model.predict_proba(self.background_data)
            return np.mean(baseline_preds[:, 1]) if baseline_preds.shape[1] > 1 else np.mean(baseline_preds)
        else:
            baseline_preds = self.model.predict(self.background_data)
            return np.mean(baseline_preds)
    
    def _predict_coalition(self, instance, coalition, background_sample):
        """Predict with a coalition of features"""
        # Create prediction input by combining instance features (in coalition) 
        # with background features (not in coalition)
        pred_input = background_sample.copy()
        
        if hasattr(pred_input, 'iloc'):
            for feature_idx in coalition:
                pred_input.iloc[:, feature_idx] = instance[feature_idx] if hasattr(instance, '__getitem__') else instance.iloc[feature_idx]
        else:
            for feature_idx in coalition:
                pred_input[:, feature_idx] = instance[feature_idx] if hasattr(instance, '__getitem__') else instance.iloc[feature_idx]
        
        if hasattr(self.model, 'predict_proba'):
            preds = self.model.predict_proba(pred_input)
            return np.mean(preds[:, 1]) if preds.shape[1] > 1 else np.mean(preds)
        else:
            preds = self.model.predict(pred_input)
            return np.mean(preds)
    
    def explain_instance(self, instance, max_coalitions=None):
        """Calculate Shapley values for a single instance using sampling approximation"""
        n_features = len(instance) if hasattr(instance, '__len__') else len(instance.values)
        
        if max_coalitions is None:
            max_coalitions = min(self.max_evals, 2**n_features)
        
        # Sample background instances for coalition evaluation
        n_background_samples = min(10, len(self.background_data))
        background_indices = np.random.choice(len(self.background_data), n_background_samples, replace=False)
        background_samples = self.background_data.iloc[background_indices] if hasattr(self.background_data, 'iloc') else self.background_data[background_indices]
        
        shapley_values = np.zeros(n_features)
        
        # Sample coalitions
        for _ in range(max_coalitions):
            # Random coalition size
            coalition_size = np.random.randint(0, n_features + 1)
            coalition = set(np.random.choice(n_features, coalition_size, replace=False))
            
            for feature_idx in range(n_features):
                # Coalition with feature
                coalition_with = coalition | {feature_idx}
                # Coalition without feature
                coalition_without = coalition - {feature_idx}
                
                # Calculate marginal contribution
                marginal_contributions = []
                for bg_idx in range(len(background_samples)):
                    bg_sample = background_samples.iloc[[bg_idx]] if hasattr(background_samples, 'iloc') else background_samples[[bg_idx]]
                    
                    value_with = self._predict_coalition(instance, coalition_with, bg_sample)
                    value_without = self._predict_coalition(instance, coalition_without, bg_sample)
                    
                    marginal_contributions.append(value_with - value_without)
                
                # Weight by coalition probability
                weight = 1.0 / (n_features * max(1, len(marginal_contributions)))
                shapley_values[feature_idx] += weight * np.mean(marginal_contributions)
        
        return shapley_values
    
    def explain_dataset(self, X, max_samples=100):
        """Calculate Shapley values for multiple instances"""
        n_samples = min(max_samples, len(X))
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        
        shapley_matrix = []
        
        print(f"Calculating Shapley values for {n_samples} samples...")
        for i, idx in enumerate(sample_indices):
            if i % 10 == 0:
                print(f"  Progress: {i}/{n_samples}")
                
            instance = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
            shapley_vals = self.explain_instance(instance)
            shapley_matrix.append(shapley_vals)
        
        return np.array(shapley_matrix)

class PermutationExplainer:
    """Permutation-based feature importance"""
    
    def __init__(self, model, X_test, y_test, metric='accuracy'):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.metric = metric
        self.baseline_score = self._calculate_baseline_score()
    
    def _calculate_baseline_score(self):
        """Calculate baseline performance"""
        if hasattr(self.model, 'predict_proba') and self.metric == 'auc':
            y_proba = self.model.predict_proba(self.X_test)
            if len(np.unique(self.y_test)) == 2:
                return roc_auc_score(self.y_test, y_proba[:, 1])
            else:
                return roc_auc_score(self.y_test, y_proba, multi_class='ovr')
        else:
            y_pred = self.model.predict(self.X_test)
            if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1:
                y_pred = np.argmax(y_pred, axis=1)
            elif not isinstance(y_pred[0], (int, np.integer)):
                y_pred = (y_pred > 0.5).astype(int)
            return accuracy_score(self.y_test, y_pred)
    
    def calculate_importance(self, n_repeats=10):
        """Calculate permutation importance for all features"""
        n_features = self.X_test.shape[1]
        importance_scores = []
        
        print(f"Calculating permutation importance...")
        
        for feature_idx in range(n_features):
            feature_scores = []
            
            for repeat in range(n_repeats):
                # Create permuted version
                X_permuted = self.X_test.copy()
                if hasattr(X_permuted, 'iloc'):
                    X_permuted.iloc[:, feature_idx] = np.random.permutation(X_permuted.iloc[:, feature_idx])
                else:
                    X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])
                
                # Calculate performance with permuted feature
                if hasattr(self.model, 'predict_proba') and self.metric == 'auc':
                    y_proba = self.model.predict_proba(X_permuted)
                    if len(np.unique(self.y_test)) == 2:
                        permuted_score = roc_auc_score(self.y_test, y_proba[:, 1])
                    else:
                        permuted_score = roc_auc_score(self.y_test, y_proba, multi_class='ovr')
                else:
                    y_pred = self.model.predict(X_permuted)
                    if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1:
                        y_pred = np.argmax(y_pred, axis=1)
                    elif not isinstance(y_pred[0], (int, np.integer)):
                        y_pred = (y_pred > 0.5).astype(int)
                    permuted_score = accuracy_score(self.y_test, y_pred)
                
                # Importance is the decrease in performance
                feature_scores.append(self.baseline_score - permuted_score)
            
            importance_scores.append({
                'mean': np.mean(feature_scores),
                'std': np.std(feature_scores),
                'scores': feature_scores
            })
        
        return importance_scores

class LIMEExplainer:
    """Local Interpretable Model-agnostic Explanations"""
    
    def __init__(self, model, training_data, feature_names):
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.feature_means = np.mean(training_data, axis=0)
        self.feature_stds = np.std(training_data, axis=0)
    
    def explain_instance(self, instance, num_samples=1000, num_features=10):
        """Explain a single instance using LIME methodology"""
        from sklearn.linear_model import LinearRegression
        
        # Generate perturbed samples around the instance
        perturbed_samples = self._generate_samples(instance, num_samples)
        
        # Get model predictions for perturbed samples
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(perturbed_samples)[:, 1]
        else:
            predictions = self.model.predict(perturbed_samples)
        
        # Calculate distances from original instance
        distances = np.sqrt(np.sum((perturbed_samples - instance.reshape(1, -1))**2, axis=1))
        
        # Create sample weights (closer samples get higher weights)
        weights = np.exp(-(distances**2) / (2 * np.std(distances)**2))
        
        # Fit linear model
        linear_model = LinearRegression()
        linear_model.fit(perturbed_samples, predictions, sample_weight=weights)
        
        # Get feature coefficients as importance scores
        coefficients = linear_model.coef_
        
        # Select top features
        top_indices = np.argsort(np.abs(coefficients))[-num_features:]
        
        return {
            'feature_indices': top_indices,
            'coefficients': coefficients[top_indices],
            'intercept': linear_model.intercept_,
            'r2_score': linear_model.score(perturbed_samples, predictions, sample_weight=weights)
        }
    
    def _generate_samples(self, instance, num_samples):
        """Generate perturbed samples around an instance"""
        samples = []
        
        for _ in range(num_samples):
            # Create perturbation by sampling from normal distribution
            perturbation = np.random.normal(0, self.feature_stds * 0.1, len(instance))
            perturbed_sample = instance + perturbation
            samples.append(perturbed_sample)
        
        return np.array(samples)

class FeatureImportanceAnalyzer:
    """Advanced feature importance analysis without SHAP dependency"""
    
    def __init__(self, models_dict, X_train, X_test, y_train, y_test, feature_names):
        self.models = models_dict
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.importance_scores = {}
        self.shapley_values = {}
        
    def calculate_all_importance_methods(self, max_samples=100):
        """Calculate importance using multiple methods"""
        print("=" * 60)
        print("CALCULATING COMPREHENSIVE FEATURE IMPORTANCE")
        print("=" * 60)
        
        for model_name, model in self.models.items():
            print(f"\n--- Analyzing {model_name} ---")
            
            # 1. Permutation Importance
            print("Calculating permutation importance...")
            perm_explainer = PermutationExplainer(model, self.X_test, self.y_test)
            perm_importance = perm_explainer.calculate_importance()
            
            # 2. Shapley Values (approximate)
            print("Calculating Shapley values...")
            background_sample = self.X_train.sample(min(50, len(self.X_train))) if hasattr(self.X_train, 'sample') else self.X_train[:min(50, len(self.X_train))]
            shapley_explainer = CustomShapleyExplainer(model, background_sample)
            shapley_values = shapley_explainer.explain_dataset(self.X_test, max_samples)
            
            # 3. LIME explanations for sample instances
            print("Calculating LIME explanations...")
            lime_explainer = LIMEExplainer(model, self.X_train, self.feature_names)
            lime_explanations = []
            
            sample_size = min(10, len(self.X_test))
            for i in range(sample_size):
                instance = self.X_test.iloc[i].values if hasattr(self.X_test, 'iloc') else self.X_test[i]
                lime_exp = lime_explainer.explain_instance(instance)
                lime_explanations.append(lime_exp)
            
            # Store results
            self.importance_scores[model_name] = {
                'permutation': perm_importance,
                'shapley': shapley_values,
                'lime': lime_explanations
            }
            
            self.shapley_values[model_name] = shapley_values
        
        return self.importance_scores
    
    def calculate_feature_interactions(self, model_name, top_k=10):
        """Calculate feature interactions using difference in predictions"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        n_features = len(self.feature_names)
        
        # Sample data for interaction analysis
        sample_size = min(100, len(self.X_test))
        X_sample = self.X_test.iloc[:sample_size] if hasattr(self.X_test, 'iloc') else self.X_test[:sample_size]
        
        interaction_strengths = []
        
        print(f"Calculating feature interactions for {model_name}...")
        
        # Calculate pairwise interactions
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction_strength = self._calculate_pairwise_interaction(
                    model, X_sample, i, j
                )
                
                interaction_strengths.append({
                    'feature1': self.feature_names[i],
                    'feature2': self.feature_names[j],
                    'feature1_idx': i,
                    'feature2_idx': j,
                    'interaction_strength': interaction_strength
                })
        
        # Sort by interaction strength
        interaction_strengths.sort(key=lambda x: abs(x['interaction_strength']), reverse=True)
        
        return interaction_strengths[:top_k]
    
    def _calculate_pairwise_interaction(self, model, X_sample, feat1_idx, feat2_idx):
        """Calculate interaction strength between two features"""
        baseline_preds = model.predict_proba(X_sample)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_sample)
        
        # Create versions with features permuted individually and together
        X_perm1 = X_sample.copy()
        X_perm2 = X_sample.copy()
        X_perm_both = X_sample.copy()
        
        if hasattr(X_sample, 'iloc'):
            X_perm1.iloc[:, feat1_idx] = np.random.permutation(X_perm1.iloc[:, feat1_idx])
            X_perm2.iloc[:, feat2_idx] = np.random.permutation(X_perm2.iloc[:, feat2_idx])
            X_perm_both.iloc[:, feat1_idx] = np.random.permutation(X_perm_both.iloc[:, feat1_idx])
            X_perm_both.iloc[:, feat2_idx] = np.random.permutation(X_perm_both.iloc[:, feat2_idx])
        else:
            X_perm1[:, feat1_idx] = np.random.permutation(X_perm1[:, feat1_idx])
            X_perm2[:, feat2_idx] = np.random.permutation(X_perm2[:, feat2_idx])
            X_perm_both[:, feat1_idx] = np.random.permutation(X_perm_both[:, feat1_idx])
            X_perm_both[:, feat2_idx] = np.random.permutation(X_perm_both[:, feat2_idx])
        
        preds1 = model.predict_proba(X_perm1)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_perm1)
        preds2 = model.predict_proba(X_perm2)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_perm2)
        preds_both = model.predict_proba(X_perm_both)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_perm_both)
        
        # Interaction strength = effect of permuting both - sum of individual effects
        individual_effects = (np.mean(baseline_preds) - np.mean(preds1)) + (np.mean(baseline_preds) - np.mean(preds2))
        combined_effect = np.mean(baseline_preds) - np.mean(preds_both)
        
        return combined_effect - individual_effects
    
    def plot_comprehensive_importance(self):
        """Create comprehensive importance visualization"""
        if not self.importance_scores:
            print("No importance scores calculated. Run calculate_all_importance_methods() first.")
            return
        
        n_models = len(self.importance_scores)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        # Plot 1: Permutation importance comparison
        ax = axes[0]
        self._plot_permutation_comparison(ax)
        
        # Plot 2: Shapley values comparison
        ax = axes[1]
        self._plot_shapley_comparison(ax)
        
        # Plot 3: Method correlation
        ax = axes[2]
        self._plot_method_correlation(ax)
        
        # Plot 4: Feature stability across models
        ax = axes[3]
        self._plot_feature_stability(ax)
        
        # Plot 5: Top feature interactions
        ax = axes[4]
        self._plot_top_interactions(ax)
        
        # Plot 6: Feature importance distribution
        ax = axes[5]
        self._plot_importance_distribution(ax)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_permutation_comparison(self, ax):
        """Plot permutation importance across models"""
        perm_data = {}
        for model_name, scores in self.importance_scores.items():
            perm_scores = [p['mean'] for p in scores['permutation']]
            perm_data[model_name] = perm_scores
        
        df = pd.DataFrame(perm_data, index=self.feature_names)
        top_features = df.mean(axis=1).nlargest(15)
        df_top = df.loc[top_features.index]
        
        x_pos = np.arange(len(df_top))
        width = 0.8 / len(df_top.columns)
        colors = plt.cm.Set3(np.linspace(0, 1, len(df_top.columns)))
        
        for i, (model, color) in enumerate(zip(df_top.columns, colors)):
            ax.bar(x_pos + i * width, df_top[model], width, 
                  label=model, color=color, alpha=0.7)
        
        ax.set_title('Permutation Importance Comparison')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance Score')
        ax.set_xticks(x_pos + width * (len(df_top.columns) - 1) / 2)
        ax.set_xticklabels(df_top.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_shapley_comparison(self, ax):
        """Plot Shapley values across models"""
        shapley_data = {}
        for model_name, shapley_vals in self.shapley_values.items():
            mean_shapley = np.abs(shapley_vals).mean(axis=0)
            shapley_data[model_name] = mean_shapley
        
        df = pd.DataFrame(shapley_data, index=self.feature_names)
        top_features = df.mean(axis=1).nlargest(15)
        df_top = df.loc[top_features.index]
        
        x_pos = np.arange(len(df_top))
        width = 0.8 / len(df_top.columns)
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_top.columns)))
        
        for i, (model, color) in enumerate(zip(df_top.columns, colors)):
            ax.bar(x_pos + i * width, df_top[model], width, 
                  label=model, color=color, alpha=0.7)
        
        ax.set_title('Shapley Values Comparison')
        ax.set_xlabel('Features')
        ax.set_ylabel('Mean |Shapley Value|')
        ax.set_xticks(x_pos + width * (len(df_top.columns) - 1) / 2)
        ax.set_xticklabels(df_top.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_method_correlation(self, ax):
        """Plot correlation between different importance methods"""
        if len(self.importance_scores) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Method Correlation')
            return
        
        # Use the first model for comparison
        first_model = list(self.importance_scores.keys())[0]
        scores = self.importance_scores[first_model]
        
        perm_scores = np.array([p['mean'] for p in scores['permutation']])
        shapley_scores = np.abs(scores['shapley']).mean(axis=0)
        
        # Calculate correlation
        correlation = np.corrcoef(perm_scores, shapley_scores)[0, 1]
        
        ax.scatter(perm_scores, shapley_scores, alpha=0.6, s=50)
        ax.set_xlabel('Permutation Importance')
        ax.set_ylabel('Mean |Shapley Value|')
        ax.set_title(f'Method Correlation (r={correlation:.3f})')
        
        # Add diagonal line
        max_val = max(max(perm_scores), max(shapley_scores))
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    def _plot_feature_stability(self, ax):
        """Plot feature importance stability across models"""
        if len(self.importance_scores) < 2:
            ax.text(0.5, 0.5, 'Need multiple models\nfor stability analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Stability Analysis')
            return
        
        # Get permutation importance for all models
        perm_matrix = []
        for model_name in self.importance_scores.keys():
            perm_scores = [p['mean'] for p in self.importance_scores[model_name]['permutation']]
            perm_matrix.append(perm_scores)
        
        perm_matrix = np.array(perm_matrix)
        
        # Calculate coefficient of variation
        mean_importance = perm_matrix.mean(axis=0)
        std_importance = perm_matrix.std(axis=0)
        cv = std_importance / (mean_importance + 1e-8)
        
        # Sort by stability
        stability_order = np.argsort(cv)
        top_stable = stability_order[:15]
        
        colors = ['green' if cv[i] < 0.3 else 'orange' if cv[i] < 0.6 else 'red' 
                  for i in top_stable]
        
        ax.bar(range(len(top_stable)), cv[top_stable], color=colors, alpha=0.7)
        ax.set_xlabel('Features (Ranked by Stability)')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Feature Importance Stability')
        ax.set_xticks(range(len(top_stable)))
        ax.set_xticklabels([self.feature_names[i] for i in top_stable], 
                          rotation=45, ha='right')
        
        # Add legend
        green_patch = mpatches.Patch(color='green', alpha=0.7, label='High Stability')
        orange_patch = mpatches.Patch(color='orange', alpha=0.7, label='Medium Stability')
        red_patch = mpatches.Patch(color='red', alpha=0.7, label='Low Stability')
        ax.legend(handles=[green_patch, orange_patch, red_patch])
        ax.grid(True, alpha=0.3)
    
    def _plot_top_interactions(self, ax):
        """Plot top feature interactions"""
        # Calculate interactions for the first model
        first_model = list(self.models.keys())[0]
        interactions = self.calculate_feature_interactions(first_model, top_k=10)
        
        if interactions:
            interaction_names = [f"{int['feature1']}\n×\n{int['feature2']}" 
                               for int in interactions]
            interaction_strengths = [abs(int['interaction_strength']) 
                                   for int in interactions]
            
            colors = plt.cm.plasma(np.linspace(0, 1, len(interaction_strengths)))
            bars = ax.bar(range(len(interactions)), interaction_strengths, 
                         color=colors, alpha=0.7)
            
            ax.set_title(f'Top Feature Interactions ({first_model})')
            ax.set_xlabel('Feature Pairs')
            ax.set_ylabel('Interaction Strength')
            ax.set_xticks(range(len(interactions)))
            ax.set_xticklabels(interaction_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No interactions calculated', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Interactions')
    
    def _plot_importance_distribution(self, ax):
        """Plot distribution of feature importance scores"""
        first_model = list(self.importance_scores.keys())[0]
        perm_scores = [p['mean'] for p in self.importance_scores[first_model]['permutation']]
        
        ax.hist(perm_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(perm_scores), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(perm_scores):.3f}')
        ax.axvline(np.median(perm_scores), color='green', linestyle='--', 
                  label=f'Median: {np.median(perm_scores):.3f}')
        
        ax.set_title('Feature Importance Distribution')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def generate_importance_report(self):
        """Generate comprehensive importance analysis report"""
        if not self.importance_scores:
            print("No importance scores available. Run calculate_all_importance_methods() first.")
            return None
        
        print("=" * 80)
        print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS REPORT")
        print("