import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, 150)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(150, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def calculate_integrated_gradients(model, x_sample, baseline=None, steps=50):
    baseline = baseline if baseline is not None else torch.zeros_like(x_sample)
    alphas = torch.linspace(0, 1, steps, device=x_sample.device).view(-1, 1)
    
    with torch.set_grad_enabled(True):
        interpolated = baseline + alphas * (x_sample - baseline)
        interpolated.requires_grad_(True)
        
        outputs = model(interpolated)
        grads = torch.autograd.grad(outputs.sum(), interpolated, create_graph=False)[0]
        
    avg_grads = grads.mean(dim=0)
    integrated_gradients = (x_sample - baseline) * avg_grads
    return torch.abs(integrated_gradients).squeeze().detach().cpu().numpy()

def calculate_perturbation_sensitivity(model, x_sample, perturbation_std=0.01, n_samples=100):
    model.eval()
    input_dim = x_sample.shape[1]
    
    with torch.no_grad():
        baseline_output = model(x_sample).item()
        noise = torch.randn(n_samples, input_dim, device=x_sample.device) * perturbation_std
        
        perturbed_samples = x_sample.repeat(n_samples, 1) + noise
        outputs = model(perturbed_samples).squeeze()
        
        sensitivities = torch.abs(outputs - baseline_output).mean(dim=0).cpu().numpy()
    
    return sensitivities

def calculate_feature_ablation_sensitivity(model, x_sample, ablation_values=None):
    model.eval()
    ablation_values = ablation_values if ablation_values is not None else torch.zeros_like(x_sample)
    
    with torch.no_grad():
        original_output = model(x_sample)
        
        # Create all ablated samples at once
        ablated_samples = x_sample.repeat(x_sample.shape[1], 1)
        ablated_samples[torch.arange(x_sample.shape[1]), torch.arange(x_sample.shape[1])] = ablation_values.squeeze()[torch.arange(x_sample.shape[1])]
        
        ablated_outputs = model(ablated_samples)
        sensitivities = torch.abs(original_output - ablated_outputs).squeeze().cpu().numpy()
    
    return sensitivities

def get_model_input_size(model_path):
    """Extract the input size from a saved model"""
    try:
        # First try with weights_only=True (safer)
        model_state = torch.load(model_path, map_location='cpu', weights_only=True)
        first_layer_weight = model_state['fc1.weight']
        input_size = first_layer_weight.shape[1]
        return input_size
    except Exception as e1:
        try:
            # If that fails, try loading the full model (less safe but needed for older saves)
            logger.info(f"Trying full model load for {model_path}")
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            if hasattr(model, 'fc1'):
                input_size = model.fc1.in_features
                return input_size
            elif hasattr(model, 'state_dict'):
                # If it's a model with state_dict
                state_dict = model.state_dict()
                first_layer_weight = state_dict['fc1.weight']
                input_size = first_layer_weight.shape[1]
                return input_size
        except Exception as e2:
            logger.warning(f"Could not determine input size from {model_path}: {e1}, {e2}")
            return None

def pad_or_trim_features(X, target_input_size, feature_names):
    """Pad with zeros or trim features to match model input size"""
    current_size = X.shape[1]
    
    if current_size == target_input_size:
        return X, feature_names
    elif current_size < target_input_size:
        # Pad with zeros
        padding = torch.zeros(X.shape[0], target_input_size - current_size, device=X.device)
        X_padded = torch.cat([X, padding], dim=1)
        
        # Add dummy feature names for padded columns
        padded_feature_names = feature_names + [f'padded_feature_{i}' for i in range(target_input_size - current_size)]
        
        logger.info(f"Padded input from {current_size} to {target_input_size} features")
        return X_padded, padded_feature_names
    else:
        # Trim features (take first target_input_size features)
        X_trimmed = X[:, :target_input_size]
        trimmed_feature_names = feature_names[:target_input_size]
        
        logger.warning(f"Trimmed input from {current_size} to {target_input_size} features")
        return X_trimmed, trimmed_feature_names

def save_feature_mapping(original_features, model_features, output_dir, model_name):
    """Save the mapping between original and model features"""
    mapping = {
        'original_features': original_features,
        'model_features': model_features,
        'model_name': model_name
    }
    
    mapping_path = output_dir / f"feature_mapping_{model_name}.pkl"
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping, f)
    
    logger.info(f"Feature mapping saved to {mapping_path}")

def test_model_with_comprehensive_sensitivity(model_path, x_test, y_test, column_name, feature_names, output_dir):
    logger.info(f"\n--- Comprehensive Testing: {column_name} ---")
    
    # First, determine the expected input size from the model
    model_input_size = get_model_input_size(model_path)
    if model_input_size is None:
        logger.error(f"Could not determine input size for model {column_name}")
        return None
    
    logger.info(f"Model expects {model_input_size} input features, data has {x_test.shape[1]} features")
    
    # Adjust input data to match model expectations
    x_test_adjusted, adjusted_feature_names = pad_or_trim_features(x_test, model_input_size, feature_names)
    
    # Save feature mapping for reference
    save_feature_mapping(feature_names, adjusted_feature_names, output_dir, column_name)
    
    try:
        # Try loading with weights_only=False since models contain custom class
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        
        # Verify the model is compatible
        if not hasattr(model, 'fc1'):
            logger.error(f"Model {column_name} doesn't have expected fc1 layer")
            return None
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

    # Pre-allocate results dictionary
    num_samples = x_test_adjusted.shape[0]
    num_features = len(adjusted_feature_names)
    
    results = {
        'sample_index': np.arange(num_samples),
        'true_value': np.zeros(num_samples),
        'predicted_value': np.zeros(num_samples),
        'relative_error': np.zeros(num_samples),
        'absolute_error': np.zeros(num_samples),
        'squared_error': np.zeros(num_samples),
        'target_column': [column_name] * num_samples
    }
    
    # Add sensitivity measure columns
    for method in ['integrated_gradients', 'perturbation_sensitivity', 'ablation_sensitivity']:
        for fname in adjusted_feature_names:
            results[f'{method}_{fname}'] = np.zeros(num_samples)

    # Batch processing for predictions
    with torch.no_grad():
        y_pred_all = model(x_test_adjusted).squeeze()
        y_test_flat = y_test.view(-1)
        
        results['true_value'] = y_test_flat.cpu().numpy()
        results['predicted_value'] = y_pred_all.cpu().numpy()
        
        epsilon = 1e-8
        relative_error = torch.abs((y_test_flat - y_pred_all) / (y_test_flat + epsilon))
        results['relative_error'] = relative_error.cpu().numpy()
        results['absolute_error'] = torch.abs(y_test_flat - y_pred_all).cpu().numpy()
        results['squared_error'] = (y_test_flat - y_pred_all).pow(2).cpu().numpy()

    # Calculate metrics
    mse = F.mse_loss(y_pred_all, y_test_flat)
    mae = F.l1_loss(y_pred_all, y_test_flat)
    logger.info(f"Overall MSE: {mse.item():.6f}")
    logger.info(f"Overall MAE: {mae.item():.6f}")

    # Process samples in batches for sensitivity analysis
    batch_size = 100  # Adjust based on available memory
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    logger.info(f"Analyzing {num_samples} samples for '{column_name}' in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        
        current_x = x_test_adjusted[start_idx:end_idx]
        
        # Vectorized sensitivity calculations
        for i in range(current_x.shape[0]):
            sample_idx = start_idx + i
            x_sample = current_x[i].unsqueeze(0)
            
            try:
                # Calculate sensitivities
                ig = calculate_integrated_gradients(model, x_sample)
                ps = calculate_perturbation_sensitivity(model, x_sample)
                as_ = calculate_feature_ablation_sensitivity(model, x_sample)
                
                # Store results
                for j, fname in enumerate(adjusted_feature_names):
                    results[f'integrated_gradients_{fname}'][sample_idx] = ig[j]
                    results[f'perturbation_sensitivity_{fname}'][sample_idx] = ps[j]
                    results[f'ablation_sensitivity_{fname}'][sample_idx] = as_[j]
            except Exception as e:
                logger.warning(f"Error processing sample {sample_idx}: {e}")
                continue

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            logger.info(f"  Processed {end_idx}/{num_samples} samples.")

    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

def load_and_preprocess_data(filepath):
    logger.info("Loading and preprocessing data...")
    
    df = pd.read_csv(filepath, na_values=['#DIV/0!'], dtype='float', header=1)
    initial_num_input_cols = 67
    
    # Check for NaN columns
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        logger.info(f"Found {len(nan_cols)} columns with NaN values: {nan_cols[:5]}..." if len(nan_cols) > 5 else f"Found columns with NaN values: {nan_cols}")
    
    # Remove NaN columns
    df_clean = df.dropna(axis=1, how='any')
    removed_input_cols = len(nan_cols)
    
    final_num_input_cols = initial_num_input_cols - removed_input_cols
    num_output_cols = df_clean.shape[1] - final_num_input_cols
    
    logger.info(f"Data preprocessing complete:")
    logger.info(f"- Original input columns: {initial_num_input_cols}")
    logger.info(f"- Removed columns (NaN): {removed_input_cols}")
    logger.info(f"- Final input columns: {final_num_input_cols}")
    logger.info(f"- Output columns: {num_output_cols}")
    logger.info(f"- Total samples: {df_clean.shape[0]}")
    logger.info(f"- Final DF shape: {df_clean.shape}")
    
    X = df_clean.iloc[:, :final_num_input_cols]
    y_all_df = df_clean.iloc[:, final_num_input_cols:]
    feature_names = X.columns.tolist()
    
    return torch.from_numpy(X.to_numpy()).float().to(device), y_all_df, feature_names

def analyze_feature_importance(results_df, feature_names):
    summary_stats = []
    sensitivity_rankings = []
    
    for col in results_df['target_column'].unique():
        col_data = results_df[results_df['target_column'] == col]
        
        # Basic statistics
        stats = {
            'target_column': col,
            'mean_relative_error': col_data['relative_error'].mean(),
            'std_relative_error': col_data['relative_error'].std(),
            'mean_absolute_error': col_data['absolute_error'].mean(),
            'rmse': np.sqrt(col_data['squared_error'].mean()),
            'r2_score': 1 - (col_data['squared_error'].sum() / 
                           ((col_data['true_value'] - col_data['true_value'].mean())**2).sum())
        }
        
        # Sensitivity statistics
        for method in ['integrated_gradients', 'perturbation_sensitivity', 'ablation_sensitivity']:
            for fname in feature_names:
                col_name = f'{method}_{fname}'
                if col_name in col_data.columns:
                    stats[f'mean_{method}_{fname}'] = col_data[col_name].mean()
        
        summary_stats.append(stats)
        
        # Feature rankings
        for fname in feature_names:
            ranking_info = {
                'target_column': col,
                'feature_name': fname,
                'average_rank': 0
            }
            
            method_ranks = []
            for method in ['integrated_gradients', 'perturbation_sensitivity', 'ablation_sensitivity']:
                # Get all features that exist for this method
                method_cols = [f'{method}_{f}' for f in feature_names if f'{method}_{f}' in col_data.columns]
                if method_cols:
                    # Get mean sensitivity for each feature
                    feature_sensitivities = {f.replace(f'{method}_', ''): col_data[f].mean() for f in method_cols}
                    sorted_features = sorted(feature_sensitivities.keys(), 
                                           key=lambda x: feature_sensitivities[x], 
                                           reverse=True)
                    if fname in sorted_features:
                        method_ranks.append(sorted_features.index(fname) + 1)
            
            if method_ranks:
                ranking_info.update({
                    'integrated_gradients_rank': method_ranks[0] if len(method_ranks) > 0 else None,
                    'perturbation_rank': method_ranks[1] if len(method_ranks) > 1 else None,
                    'ablation_rank': method_ranks[2] if len(method_ranks) > 2 else None,
                    'average_rank': np.mean(method_ranks)
                })
            
            sensitivity_rankings.append(ranking_info)
    
    return pd.DataFrame(summary_stats), pd.DataFrame(sensitivity_rankings)

def main():
    data_path = Path(r"C:\Users\U375383\Documents\work\5000FEMnormalisedNewPyReady.csv")
    model_dir = Path(r"C:\Users\U375383\Documents\work\1000EpochModelBycolumn")
    output_dir = Path(r"C:\Users\U375383\Documents\work")
    
    # Add safe globals for model loading
    torch.serialization.add_safe_globals([Net])
    
    X, y_all_df, feature_names = load_and_preprocess_data(data_path)
    
    all_results = []
    logger.info(f"\nStarting comprehensive sensitivity analysis for {len(y_all_df.columns)} models...")
    
    # Test with a few models first to avoid processing all if there are issues
    test_models = list(y_all_df.columns)[:3]  # Test first 3 models
    logger.info(f"Testing with first {len(test_models)} models: {test_models}")
    
    for col in test_models:
        model_path = model_dir / f"model_{col}.pth"
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            continue
            
        y_test = torch.from_numpy(y_all_df[col].to_numpy()).float().to(device)
        
        results = test_model_with_comprehensive_sensitivity(model_path, X, y_test, col, feature_names, output_dir)
        if results:
            all_results.append(pd.DataFrame(results))
            logger.info(f"✓ Completed analysis for {col}")
        else:
            logger.warning(f"✗ Failed to analyze {col}")
    
    # If test models work, process all models
    if all_results:
        logger.info("Test models successful. Processing all models...")
        all_results = []  # Reset for full run
        
        for col in y_all_df.columns:
            model_path = model_dir / f"model_{col}.pth"
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                continue
                
            y_test = torch.from_numpy(y_all_df[col].to_numpy()).float().to(device)
            
            results = test_model_with_comprehensive_sensitivity(model_path, X, y_test, col, feature_names, output_dir)
            if results:
                all_results.append(pd.DataFrame(results))
                logger.info(f"✓ Completed analysis for {col}")
            else:
                logger.warning(f"✗ Failed to analyze {col}")
    
    if all_results:
        df_to_export = pd.concat(all_results, ignore_index=True)
        
        # Use the feature names from the first model's results for consistency
        first_result_cols = [col for col in df_to_export.columns if col.startswith('integrated_gradients_')]
        result_feature_names = [col.replace('integrated_gradients_', '') for col in first_result_cols]
        
        summary_df, rankings_df = analyze_feature_importance(df_to_export, result_feature_names)
        
        # Save results
        df_to_export.to_csv(output_dir / "SensitivityAnalysis.csv", index=False)
        summary_df.to_csv(output_dir / "SensitivitySummary.csv", index=False)
        rankings_df.to_csv(output_dir / "FeatureImportanceRankings.csv", index=False)
        
        logger.info("\n=== ANALYSIS COMPLETE ===")
        logger.info(f"📊 Total samples processed: {len(df_to_export):,}")
        logger.info(f"🎯 Models tested: {len(all_results)}")
        logger.info(f"🔧 Features analyzed: {len(result_feature_names)}")
        logger.info(f"💾 Results saved to: {output_dir}")
    else:
        logger.error("❌ No models were successfully tested. CSV files not created.")

if __name__ == "__main__":
    main()