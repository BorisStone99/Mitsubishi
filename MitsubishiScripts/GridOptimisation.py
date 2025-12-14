import numpy as np
import time
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from itertools import product
import random

# Load and prepare data
df = pd.read_csv(rf"C:\Users\U375383\Documents\work\normalisedFEMPyReady.csv", header=None, dtype=float)
df = df.dropna(how='all')
X = df.iloc[:, 0:58]  # cols 1 to 58 are the input variables
y = df.iloc[:, 58:180]  # Remaining 122 columns are the output variables

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid for manual optimization
param_grid = {
    'hidden_layer_sizes': [(32,), (64,), (128,), (256,), 
                          (32, 32), (64, 64), (128, 128), 
                          (32, 32, 32), (64, 64, 64)],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [200, 500, 1000],
    'tol': [1e-6, 1e-5, 1e-4],
    'n_iter_no_change': [10, 20, 50]
}

# Create all combinations
param_combinations = list(product(
    param_grid['hidden_layer_sizes'],
    param_grid['alpha'],
    param_grid['learning_rate_init'],
    param_grid['max_iter'],
    param_grid['tol'],
    param_grid['n_iter_no_change']
))

print(f"Total parameter combinations: {len(param_combinations)}")
print("Using random search to test a subset of combinations...")

# Random search - test a subset of combinations
n_random_search = min(100, len(param_combinations))  # Test up to 100 combinations
random.seed(42)
random_combinations = random.sample(param_combinations, n_random_search)

best_score = float('-inf')
best_params = None
best_model = None
results = []

print(f"Testing {n_random_search} random parameter combinations...")
start_time = time.time()

for i, (hidden_layers, alpha, lr, max_iter, tol, n_iter_no_change) in enumerate(random_combinations):
    print(f"Progress: {i+1}/{n_random_search} - Testing {hidden_layers}")
    
    # Create model with current parameters
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        alpha=alpha,
        learning_rate_init=lr,
        max_iter=max_iter,
        tol=tol,
        n_iter_no_change=n_iter_no_change,
        activation='relu',
        solver='adam',
        early_stopping=True,
        verbose=False,
        validation_fraction=0.2,
        batch_size=64,
        random_state=42
    )
    
    try:
        # Use cross-validation to evaluate the model
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=3, 
            scoring='neg_mean_absolute_error',
            n_jobs=1
        )
        
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        # Store results
        results.append({
            'hidden_layer_sizes': hidden_layers,
            'alpha': alpha,
            'learning_rate_init': lr,
            'max_iter': max_iter,
            'tol': tol,
            'n_iter_no_change': n_iter_no_change,
            'cv_score': mean_cv_score,
            'cv_std': std_cv_score
        })
        
        # Check if this is the best so far
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_params = {
                'hidden_layer_sizes': hidden_layers,
                'alpha': alpha,
                'learning_rate_init': lr,
                'max_iter': max_iter,
                'tol': tol,
                'n_iter_no_change': n_iter_no_change
            }
            
            # Train the best model on full training data
            best_model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                alpha=alpha,
                learning_rate_init=lr,
                max_iter=max_iter,
                tol=tol,
                n_iter_no_change=n_iter_no_change,
                activation='relu',
                solver='adam',
                early_stopping=True,
                verbose=False,
                validation_fraction=0.2,
                batch_size=64,
                random_state=42
            )
            best_model.fit(X_train, y_train)
            
            print(f"New best score: {best_score:.4f} (±{std_cv_score:.4f})")
    
    except Exception as e:
        print(f"Error with parameters {hidden_layers}: {e}")
        continue

optimization_time = time.time() - start_time
print(f"\nOptimization completed in {optimization_time:.2f} seconds")

if best_model is not None:
    print(f"\nBest parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    # Make predictions on validation data
    start_time = time.time()
    y_pred = best_model.predict(X_val)
    inference_time = time.time() - start_time
    
    # Evaluate performance
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f'R2 Score: {r2:.4f}')
    print(f'Mean Absolute Error on Validation Set: {mae:.4f}')
    print(f'Inference time: {inference_time:.4f} seconds')
    
    # Print model details
    print(f"\nFinal model configuration:")
    print(f"Hidden layers: {best_model.hidden_layer_sizes}")
    print(f"Alpha (L2 regularization): {best_model.alpha}")
    print(f"Learning rate: {best_model.learning_rate_init}")
    print(f"Max iterations: {best_model.max_iter}")
    print(f"Tolerance: {best_model.tol}")
    print(f"N iter no change: {best_model.n_iter_no_change}")
    
    # Show top 10 results
    print("\nTop 10 parameter combinations:")
    sorted_results = sorted(results, key=lambda x: x['cv_score'], reverse=True)[:10]
    for i, result in enumerate(sorted_results):
        print(f"{i+1}. Score: {result['cv_score']:.4f} (±{result['cv_std']:.4f}) - "
              f"Layers: {result['hidden_layer_sizes']}, Alpha: {result['alpha']}, "
              f"LR: {result['learning_rate_init']}")
    
    # Optional: Save the best model
    # import joblib
    # joblib.dump(best_model, 'best_mlp_model.pkl')
    
    # Save results to CSV for later analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv('mlp_optimization_results.csv', index=False)
    print(f"\nResults saved to 'mlp_optimization_results.csv'")
    
else:
    print("No successful optimization found!")