import numpy as np
import time
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load and prepare data
df = pd.read_csv(rf"C:\Users\U375383\Documents\work\normalisedFEMPyReady.csv", header=None, dtype=float)
df = df.dropna(how='all')
X = df.iloc[:, 0:58]  # cols 1 to 58 are the input variables
y = df.iloc[:, 58:180]  # Remaining 122 columns are the output variables

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data loaded successfully!")
print(f'Training set size: {X_train.shape[0]}')
print(f'Validation set size: {X_val.shape[0]}')

# ============================================================================
# WORKAROUND 1: Use separate optimization for each architecture
# ============================================================================

print("\n" + "="*60)
print("WORKAROUND 1: Separate BayesSearchCV for each architecture")
print("="*60)

# Define architectures to test
architectures = [
    (32, 32),
    (64, 64), 
    (16, 16, 16),
    (32, 32, 32),
    (128, 64),
    (100,),
    (200,)
]

# Parameters to optimize (excluding hidden_layer_sizes)
param_spaces = {
    'alpha': Real(1e-6, 1e2, prior='log-uniform'),
    'max_iter': Integer(100, 1000),
    'tol': Real(1e-6, 1e-2, prior='log-uniform'),
    'n_iter_no_change': Integer(5, 50)
}

best_score = float('-inf')
best_params = {}
best_model = None

start_time = time.time()

for i, architecture in enumerate(architectures):
    print(f"\nTesting architecture {i+1}/{len(architectures)}: {architecture}")
    
    # Create model with fixed architecture
    model = MLPRegressor(
        hidden_layer_sizes=architecture,
        activation='relu',
        solver='adam',
        early_stopping=True,
        validation_fraction=0.2,
        batch_size=64,
        random_state=42,
        verbose=False
    )
    
    # Optimize other parameters
    opt = BayesSearchCV(
        model,
        search_spaces=param_spaces,
        n_iter=15,  # Reduced for faster execution
        cv=3,
        n_jobs=1,
        verbose=0,
        random_state=42
    )
    
    try:
        opt.fit(X_train, y_train)
        
        if opt.best_score_ > best_score:
            best_score = opt.best_score_
            best_params = opt.best_params_.copy()
            best_params['hidden_layer_sizes'] = architecture
            best_model = opt.best_estimator_
        
        print(f"  Best score: {opt.best_score_:.4f}")
        print(f"  Best params: {opt.best_params_}")
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

optimization_time = time.time() - start_time
print(f"\nWorkaround 1 completed in {optimization_time:.2f} seconds")

if best_model:
    print(f"Best overall parameters: {best_params}")
    print(f"Best CV score: {best_score:.4f}")
    
    # Test on validation set
    y_pred = best_model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"Validation R2: {r2:.4f}")
    print(f"Validation MAE: {mae:.4f}")

# ============================================================================
# WORKAROUND 2: Use integer encoding for architectures
# ============================================================================

print("\n" + "="*60)
print("WORKAROUND 2: Integer encoding for architectures")
print("="*60)

# Custom MLPRegressor wrapper that maps integers to architectures
class MLPRegressorWithIntegerArchitecture(MLPRegressor):
    # Architecture mapping
    ARCHITECTURES = {
        0: (32, 32),
        1: (64, 64),
        2: (16, 16, 16),
        3: (32, 32, 32),
        4: (128, 64),
        5: (100,),
        6: (200,)
    }
    
    def __init__(self, architecture_idx=0, **kwargs):
        # First initialize with default architecture
        self.architecture_idx = architecture_idx
        hidden_layer_sizes = self.ARCHITECTURES[architecture_idx]
        
        # Initialize parent with the mapped architecture
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, **kwargs)
    
    def get_params(self, deep=True):
        # Get all MLPRegressor parameters
        params = super().get_params(deep=deep)
        
        # Replace hidden_layer_sizes with architecture_idx
        if 'hidden_layer_sizes' in params:
            del params['hidden_layer_sizes']
        params['architecture_idx'] = self.architecture_idx
        
        return params
    
    def set_params(self, **params):
        # Handle architecture_idx parameter
        if 'architecture_idx' in params:
            self.architecture_idx = params.pop('architecture_idx')
            # Map to actual architecture
            params['hidden_layer_sizes'] = self.ARCHITECTURES[self.architecture_idx]
        
        # Set all parameters on the parent class
        result = super().set_params(**params)
        return result

# Create the wrapper model
wrapper_model = MLPRegressorWithIntegerArchitecture(
    activation='relu',
    solver='adam',
    early_stopping=True,
    validation_fraction=0.2,
    batch_size=64,
    random_state=42,
    verbose=False
)

# Define parameter space with integer architecture
param_spaces_wrapper = {
    'architecture_idx': Integer(0, 6),  # 0-6 map to different architectures
    'alpha': Real(1e-6, 1e2, prior='log-uniform'),
    'max_iter': Integer(100, 1000),
    'tol': Real(1e-6, 1e-2, prior='log-uniform'),
    'n_iter_no_change': Integer(5, 50)
}

# Run BayesSearchCV with wrapper
print("Running BayesSearchCV with integer architecture encoding...")
start_time = time.time()

opt_wrapper = BayesSearchCV(
    wrapper_model,
    search_spaces=param_spaces_wrapper,
    n_iter=30,
    cv=3,
    n_jobs=1,
    verbose=1,
    random_state=42
)

try:
    opt_wrapper.fit(X_train, y_train)
    
    optimization_time = time.time() - start_time
    print(f"\nWorkaround 2 completed in {optimization_time:.2f} seconds")
    
    print(f"Best parameters: {opt_wrapper.best_params_}")
    print(f"Best CV score: {opt_wrapper.best_score_:.4f}")
    
    # Convert back to readable architecture
    best_arch_idx = opt_wrapper.best_params_['architecture_idx']
    best_architecture = MLPRegressorWithIntegerArchitecture.ARCHITECTURES[best_arch_idx]
    print(f"Best architecture: {best_architecture}")
    
    # Test on validation set
    y_pred_wrapper = opt_wrapper.best_estimator_.predict(X_val)
    mae_wrapper = mean_absolute_error(y_val, y_pred_wrapper)
    r2_wrapper = r2_score(y_val, y_pred_wrapper)
    
    print(f"Validation R2: {r2_wrapper:.4f}")
    print(f"Validation MAE: {mae_wrapper:.4f}")
    
except Exception as e:
    print(f"Workaround 2 failed: {e}")

# ============================================================================
# WORKAROUND 3: Use only single-layer architectures (no tuples)
# ============================================================================

print("\n" + "="*60)
print("WORKAROUND 3: Single-layer architectures only")
print("="*60)

# Single layer model
single_layer_model = MLPRegressor(
    activation='relu',
    solver='adam',
    early_stopping=True,
    validation_fraction=0.2,
    batch_size=64,
    random_state=42,
    verbose=False
)

# Parameter space with single integers only
param_spaces_single = {
    'hidden_layer_sizes': Integer(32, 256),  # Single layer with 32-256 neurons
    'alpha': Real(1e-6, 1e2, prior='log-uniform'),
    'max_iter': Integer(100, 1000),
    'tol': Real(1e-6, 1e-2, prior='log-uniform'),
    'n_iter_no_change': Integer(5, 50)
}

print("Running BayesSearchCV with single-layer architectures...")
start_time = time.time()

opt_single = BayesSearchCV(
    single_layer_model,
    search_spaces=param_spaces_single,
    n_iter=30,
    cv=3,
    n_jobs=1,
    verbose=1,
    random_state=42
)

try:
    opt_single.fit(X_train, y_train)
    
    optimization_time = time.time() - start_time
    print(f"\nWorkaround 3 completed in {optimization_time:.2f} seconds")
    
    print(f"Best parameters: {opt_single.best_params_}")
    print(f"Best CV score: {opt_single.best_score_:.4f}")
    
    # Test on validation set
    y_pred_single = opt_single.best_estimator_.predict(X_val)
    mae_single = mean_absolute_error(y_val, y_pred_single)
    r2_single = r2_score(y_val, y_pred_single)
    
    print(f"Validation R2: {r2_single:.4f}")
    print(f"Validation MAE: {mae_single:.4f}")
    
except Exception as e:
    print(f"Workaround 3 failed: {e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("The issue with BayesSearchCV and hidden_layer_sizes is a known bug")
print("in the scikit-optimize library when dealing with tuple parameters.")
print("The workarounds above show different approaches to handle this:")
print("1. Separate optimization for each architecture")
print("2. Integer encoding wrapper")
print("3. Single-layer architectures only")
print("Choose the approach that works best for your specific use case!")