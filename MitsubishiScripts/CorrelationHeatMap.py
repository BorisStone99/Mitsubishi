import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# LOAD YOUR NORMALIZED DATA HERE
# Option 1: Load from CSV
# df = pd.read_csv('your_normalized_data.csv')

# Option 2: Load from existing numpy array or DataFrame
# df = pd.DataFrame(your_normalized_array, columns=[f'feature_{i+1}' for i in range(60)])

# Option 3: Example - replace this with your actual data loading
# For demonstration, creating normalized sample data
np.random.seed(42)
normalized_data = np.random.randn(1000, 66)  # Changed to 66 features
feature_names = [f'feature_{i+1}' for i in range(66)]
df = pd.DataFrame(normalized_data, columns=feature_names)

# Verify data is normalized
print("Data normalization check:")
print(f"Mean across all features: {df.mean().mean():.6f}")
print(f"Std deviation across all features: {df.std().mean():.6f}")
print(f"Feature means range: {df.mean().min():.6f} to {df.mean().max():.6f}")
print(f"Feature std devs range: {df.std().min():.6f} to {df.std().max():.6f}")

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# 1. BASIC CORRELATION MATRIX
print("\n" + "="*50)
print("CORRELATION MATRIX ANALYSIS")
print("="*50)

# Calculate Pearson correlation matrix
correlation_matrix = df.corr(method='pearson')
print(f"Correlation matrix shape: {correlation_matrix.shape}")

# Basic statistics about correlations
corr_values = correlation_matrix.values
# Remove diagonal (self-correlations = 1.0)
mask = np.triu(np.ones_like(corr_values, dtype=bool), k=1)
upper_triangle = corr_values[mask]

print(f"\nCorrelation Statistics:")
print(f"Mean correlation: {np.mean(upper_triangle):.3f}")
print(f"Std correlation: {np.std(upper_triangle):.3f}")
print(f"Max correlation: {np.max(upper_triangle):.3f}")
print(f"Min correlation: {np.min(upper_triangle):.3f}")

# 2. IDENTIFY HIGH CORRELATIONS
print("\n" + "="*50)
print("HIGH CORRELATION PAIRS")
print("="*50)

def find_high_correlations(corr_matrix, threshold=0.8):
    """Find pairs of features with correlation above threshold"""
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    # Handle case where no high correlations are found
    if not high_corr_pairs:
        return pd.DataFrame(columns=['feature1', 'feature2', 'correlation'])
    
    return pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)

high_corr_df = find_high_correlations(correlation_matrix, threshold=0.7)
print(f"Found {len(high_corr_df)} feature pairs with |correlation| >= 0.7:")
print(high_corr_df.head(10))

# 3. DIFFERENT CORRELATION METHODS
print("\n" + "="*50)
print("DIFFERENT CORRELATION METHODS")
print("="*50)

# Pearson (linear relationships)
pearson_corr = df.corr(method='pearson')

# Spearman (monotonic relationships)
spearman_corr = df.corr(method='spearman')

# Kendall (rank-based, robust to outliers)
kendall_corr = df.corr(method='kendall')

print("Correlation method comparison for first 5 features:")
sample_features = correlation_matrix.columns[:5]
print("\nPearson correlations:")
print(pearson_corr.loc[sample_features, sample_features].round(3))
print("\nSpearman correlations:")
print(spearman_corr.loc[sample_features, sample_features].round(3))

# 4. HEATMAP FUNCTIONS (MATPLOTLIB ONLY)
def create_heatmap(data, title, ax, show_colorbar=True, annotate=False, 
                   cmap='RdBu_r', vmin=-1, vmax=1):
    """Create a heatmap using matplotlib only"""
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    # Set title
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    
    # Set ticks and labels
    if data.shape[0] <= 20:  # Only show labels for smaller matrices
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))
        ax.set_xticklabels(data.columns if hasattr(data, 'columns') else range(data.shape[1]), 
                          rotation=45, ha='right')
        ax.set_yticklabels(data.index if hasattr(data, 'index') else range(data.shape[0]))
    else:
        # For large matrices, show fewer ticks
        n_ticks = 12  # Increased for 66 features
        tick_positions = np.linspace(0, data.shape[0]-1, n_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([f'F{i+1}' for i in tick_positions])
        ax.set_yticklabels([f'F{i+1}' for i in tick_positions])
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    
    # Add annotations if requested and matrix is small enough
    if annotate and data.shape[0] <= 15:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = ax.text(j, i, f'{data.iloc[i, j]:.2f}' if hasattr(data, 'iloc') else f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(data.iloc[i, j] if hasattr(data, 'iloc') else data[i, j]) < 0.5 else "white")
    
    return im

# 5. CREATE HEATMAPS WITH MATPLOTLIB
print("\n" + "="*50)
print("CREATING HEATMAPS")
print("="*50)

# Set up the plotting
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# 1. Full correlation heatmap
create_heatmap(correlation_matrix, 'Full Correlation Matrix (66x66)', axes[0,0])

# 2. Subset heatmap with annotations (first 15 features)
subset_corr = correlation_matrix.iloc[:15, :15]
create_heatmap(subset_corr, 'Subset Correlation Matrix (First 15 of 66 Features)', 
               axes[0,1], annotate=True)

# 3. High correlation pairs only
if len(high_corr_df) > 0:
    # Create a matrix showing only high correlations (use float dtype)
    high_corr_matrix = pd.DataFrame(0.0, index=correlation_matrix.index, columns=correlation_matrix.columns, dtype=float)
    for _, row in high_corr_df.iterrows():
        high_corr_matrix.loc[row['feature1'], row['feature2']] = row['correlation']
        high_corr_matrix.loc[row['feature2'], row['feature1']] = row['correlation']
    
    # Set diagonal to 1
    np.fill_diagonal(high_corr_matrix.values, 1.0)
    
    create_heatmap(high_corr_matrix, 'High Correlations Only (|r| >= 0.7)', axes[1,0])
else:
    axes[1,0].text(0.5, 0.5, 'No high correlations found', 
                   ha='center', va='center', transform=axes[1,0].transAxes, 
                   fontsize=14)
    axes[1,0].set_title('High Correlations Only (|r| >= 0.7)', fontsize=12, fontweight='bold')

# 4. Correlation distribution histogram
axes[1,1].hist(upper_triangle, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[1,1].axvline(np.mean(upper_triangle), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(upper_triangle):.3f}')
axes[1,1].axvline(0, color='black', linestyle='-', alpha=0.3)
axes[1,1].set_xlabel('Correlation Coefficient')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Distribution of Correlation Coefficients', fontsize=12, fontweight='bold')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. CLUSTERED HEATMAP (MATPLOTLIB VERSION)
print("\nCreating clustered correlation heatmap...")

# Create distance matrix for clustering (ensure positive distances)
distance_matrix = 1 - np.abs(correlation_matrix)
# Ensure all distances are non-negative (clip any small negative values due to floating point errors)
distance_matrix = np.clip(distance_matrix, 0, None)

# Convert to condensed form for linkage
condensed_distances = squareform(distance_matrix.values)

# Ensure no negative distances in condensed form
condensed_distances = np.clip(condensed_distances, 0, None)

try:
    linkage_matrix = linkage(condensed_distances, method='average')
    
    # Create figure for clustered heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                   gridspec_kw={'width_ratios': [1, 4]})
    
    # Plot dendrogram
    dendro = dendrogram(linkage_matrix, orientation='left', ax=ax1, 
                       labels=correlation_matrix.index, leaf_font_size=8)
    ax1.set_title('Feature Clustering', fontweight='bold')
    
    # Get the order from dendrogram
    order = dendro['leaves']
    ordered_corr = correlation_matrix.iloc[order, order]
    
    # Plot ordered correlation matrix
    create_heatmap(ordered_corr, 'Clustered Correlation Matrix', ax2, show_colorbar=True)
    
    plt.tight_layout()
    plt.show()
    
except ValueError as e:
    print(f"Warning: Could not create clustered heatmap due to: {e}")
    print("This may happen with certain correlation patterns. Skipping clustered visualization.")
    
    # Create a simple ordered heatmap instead
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    create_heatmap(correlation_matrix, 'Correlation Matrix (Unclustered)', ax, show_colorbar=True)
    plt.tight_layout()
    plt.show()

# 7. CORRELATION QUALITY ASSESSMENT
print("\n" + "="*50)
print("CORRELATION QUALITY ASSESSMENT")
print("="*50)

def assess_correlation_quality(corr_matrix, high_threshold=0.9, moderate_threshold=0.7):
    """Assess the quality issues in correlation matrix"""
    
    # Get upper triangle (avoid double counting)
    mask = np.triu(np.ones_like(corr_matrix.values, dtype=bool), k=1)
    upper_triangle = corr_matrix.values[mask]
    
    # Count different correlation levels
    very_high = np.sum(np.abs(upper_triangle) >= high_threshold)
    high = np.sum((np.abs(upper_triangle) >= moderate_threshold) & 
                  (np.abs(upper_triangle) < high_threshold))
    moderate = np.sum((np.abs(upper_triangle) >= 0.5) & 
                     (np.abs(upper_triangle) < moderate_threshold))
    
    total_pairs = len(upper_triangle)
    
    print(f"Total feature pairs: {total_pairs}")
    print(f"Very high correlations (|r| >= {high_threshold}): {very_high} ({very_high/total_pairs*100:.1f}%)")
    print(f"High correlations ({moderate_threshold} <= |r| < {high_threshold}): {high} ({high/total_pairs*100:.1f}%)")
    print(f"Moderate correlations (0.5 <= |r| < {moderate_threshold}): {moderate} ({moderate/total_pairs*100:.1f}%)")
    
    # Recommendations
    print(f"\nQuality Assessment:")
    if very_high > 0:
        print(f"⚠️  WARNING: {very_high} feature pairs have very high correlations (>= {high_threshold})")
        print("   Consider removing redundant features")
    
    if high > total_pairs * 0.1:  # More than 10% high correlations
        print(f"⚠️  CONCERN: {high/total_pairs*100:.1f}% of pairs have high correlations")
        print("   Dataset may have significant redundancy")
    
    if np.mean(np.abs(upper_triangle)) > 0.3:
        print(f"⚠️  INFO: Average absolute correlation is {np.mean(np.abs(upper_triangle)):.3f}")
        print("   Features are moderately correlated overall")
    
    return {
        'very_high_corr_count': very_high,
        'high_corr_count': high,
        'moderate_corr_count': moderate,
        'mean_abs_correlation': np.mean(np.abs(upper_triangle))
    }

quality_stats = assess_correlation_quality(correlation_matrix)

# 8. FEATURE SELECTION BASED ON CORRELATIONS
print("\n" + "="*50)
print("FEATURE SELECTION RECOMMENDATIONS")
print("="*50)

def recommend_feature_removal(corr_matrix, threshold=0.9):
    """Recommend features to remove based on high correlations"""
    features_to_remove = set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                # Remove the feature with higher mean absolute correlation
                feat1_mean_corr = np.mean(np.abs(corr_matrix.iloc[i, :]))
                feat2_mean_corr = np.mean(np.abs(corr_matrix.iloc[j, :]))
                
                if feat1_mean_corr > feat2_mean_corr:
                    features_to_remove.add(corr_matrix.columns[i])
                else:
                    features_to_remove.add(corr_matrix.columns[j])
    
    return list(features_to_remove)

features_to_remove = recommend_feature_removal(correlation_matrix, threshold=0.85)
print(f"Recommended features to remove (correlation >= 0.85): {len(features_to_remove)}")
if features_to_remove:
    print("Features:", features_to_remove[:10], "..." if len(features_to_remove) > 10 else "")
    print(f"This would reduce features from 66 to {66 - len(features_to_remove)}")

# 9. ADDITIONAL VISUALIZATION: CORRELATION PATTERNS
print("\nCreating additional correlation pattern visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Correlation vs Feature Index
mean_abs_corr_per_feature = np.mean(np.abs(correlation_matrix), axis=1)
axes[0,0].bar(range(len(mean_abs_corr_per_feature)), mean_abs_corr_per_feature, 
              color='steelblue', alpha=0.7)
axes[0,0].set_xlabel('Feature Index')
axes[0,0].set_ylabel('Mean Absolute Correlation')
axes[0,0].set_title('Mean Absolute Correlation per Feature', fontweight='bold')
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Correlation matrix diagonal view (adjusted for 66 features)
diagonal_offset_corrs = []
for offset in range(1, min(25, correlation_matrix.shape[0])):  # Increased range for 66 features
    diag_corrs = [correlation_matrix.iloc[i, i+offset] for i in range(correlation_matrix.shape[0]-offset)]
    diagonal_offset_corrs.append(np.mean(np.abs(diag_corrs)))

axes[0,1].plot(range(1, len(diagonal_offset_corrs)+1), diagonal_offset_corrs, 
               marker='o', color='red', linewidth=2)
axes[0,1].set_xlabel('Feature Distance (Offset)')
axes[0,1].set_ylabel('Mean Absolute Correlation')
axes[0,1].set_title('Correlation vs Feature Distance', fontweight='bold')
axes[0,1].grid(True, alpha=0.3)

# Plot 3: High correlation network (simplified)
if len(high_corr_df) > 0:
    # Count connections per feature
    feature_connections = {}
    for _, row in high_corr_df.iterrows():
        feat1, feat2 = row['feature1'], row['feature2']
        feature_connections[feat1] = feature_connections.get(feat1, 0) + 1
        feature_connections[feat2] = feature_connections.get(feat2, 0) + 1
    
    if feature_connections:
        features, connections = zip(*feature_connections.items())
        y_pos = range(len(features))
        axes[1,0].barh(y_pos, connections, color='orange', alpha=0.7)
        axes[1,0].set_yticks(y_pos)
        # Handle different feature naming patterns safely
        feature_labels = []
        for f in features:
            if '_' in f and len(f.split('_')) > 1:
                # Handle feature_1, feature_2, etc.
                feature_labels.append(f'F{f.split("_")[1]}')
            else:
                # Handle any other naming pattern
                feature_labels.append(str(f)[:10])  # Truncate long names
        
        axes[1,0].set_yticklabels(feature_labels)
        axes[1,0].set_xlabel('Number of High Correlations')
        axes[1,0].set_title('Features with Most High Correlations', fontweight='bold')
else:
    axes[1,0].text(0.5, 0.5, 'No high correlations found', 
                   ha='center', va='center', transform=axes[1,0].transAxes)
    axes[1,0].set_title('Features with Most High Correlations', fontweight='bold')

# Plot 4: Correlation strength distribution by bins
corr_bins = [-1, -0.7, -0.3, 0, 0.3, 0.7, 1]
bin_labels = ['Strong Neg', 'Mod Neg', 'Weak Neg', 'Weak Pos', 'Mod Pos', 'Strong Pos']
bin_counts = []

for i in range(len(corr_bins)-1):
    count = np.sum((upper_triangle >= corr_bins[i]) & (upper_triangle < corr_bins[i+1]))
    bin_counts.append(count)

axes[1,1].bar(bin_labels, bin_counts, color=['red', 'lightcoral', 'lightblue', 
                                            'lightgreen', 'green', 'darkgreen'], alpha=0.7)
axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Correlation Strength Distribution', fontweight='bold')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)