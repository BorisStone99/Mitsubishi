import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

def is_pareto_efficient(costs, return_mask=True, tolerance=1e-8):
    """
    Find the Pareto-efficient points for MINIMIZATION problems
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :param tolerance: Small tolerance for numerical precision
    :return: An array of indices of Pareto-efficient points.
    """
    n_points, n_objectives = costs.shape
    is_efficient = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_efficient[i]:
            # For minimization: Point i is dominated if there exists another point j such that:
            # ALL objectives of j are <= objectives of i, AND at least one is strictly <
            
            # Find points that dominate point i
            dominates_i = np.logical_and(
                # All objectives of other points <= point i (componentwise)
                np.all(costs <= costs[i] + tolerance, axis=1),
                # At least one objective of other points < point i (strictly better)
                np.any(costs < costs[i] - tolerance, axis=1)
            )
            
            # If any point dominates point i, then point i is not Pareto efficient
            if np.any(dominates_i):
                is_efficient[i] = False
    
    if return_mask:
        return is_efficient
    else:
        return np.where(is_efficient)[0]

def is_pareto_efficient_extended(costs, return_mask=True, epsilon=0.1):
    """
    Find extended Pareto-efficient points (epsilon-dominance) for MINIMIZATION
    This includes more points that are "nearly" Pareto optimal
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :param epsilon: Tolerance for epsilon-dominance (as percentage of range)
    :return: An array of indices of epsilon-Pareto-efficient points.
    """
    n_points, n_objectives = costs.shape
    
    # Calculate epsilon values as percentage of the range for each objective
    ranges = np.ptp(costs, axis=0)  # peak-to-peak (max - min) for each objective
    epsilon_values = epsilon * ranges
    
    is_efficient = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_efficient[i]:
            # For minimization with epsilon-dominance:
            # Point i is epsilon-dominated if there exists another point j such that:
            # ALL objectives of j <= objectives of i + epsilon, AND 
            # at least one objective of j < objectives of i - epsilon
            
            epsilon_dominates_i = np.logical_and(
                # All objectives of other points <= point i + epsilon
                np.all(costs <= costs[i] + epsilon_values, axis=1),
                # At least one objective of other points < point i - epsilon
                np.any(costs < costs[i] - epsilon_values, axis=1)
            )
            
            if np.any(epsilon_dominates_i):
                is_efficient[i] = False
    
    if return_mask:
        return is_efficient
    else:
        return np.where(is_efficient)[0]

def plot_3d_pareto(csv_file, x_col, y_col, z_col, maximize_cols=None, include_extended=True, epsilon=0.05):
    """
    Plot 3D scatter plot with Pareto frontier
    
    Parameters:
    csv_file: path to CSV file
    x_col, y_col, z_col: column names for the three dimensions
    maximize_cols: list of column names to maximize (others will be minimized)
    include_extended: if True, also show epsilon-Pareto points
    epsilon: tolerance for epsilon-dominance (as fraction of range)
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the three columns
    data = df[[x_col, y_col, z_col]].dropna()
    
    # Convert to numpy array
    points = data.values
    
    # Handle maximization vs minimization
    if maximize_cols is None:
        maximize_cols = []
    
    # Create a copy for Pareto calculation (negate columns we want to maximize)
    pareto_points = points.copy()
    if x_col in maximize_cols:
        pareto_points[:, 0] = -pareto_points[:, 0]
    if y_col in maximize_cols:
        pareto_points[:, 1] = -pareto_points[:, 1]
    if z_col in maximize_cols:
        pareto_points[:, 2] = -pareto_points[:, 2]
    
    # Find strict Pareto efficient points
    pareto_mask = is_pareto_efficient(pareto_points)
    strict_pareto_points = points[pareto_mask]
    
    # Find extended Pareto efficient points if requested
    if include_extended:
        extended_pareto_mask = is_pareto_efficient_extended(pareto_points, epsilon=epsilon)
        extended_pareto_points = points[extended_pareto_mask]
        # Remove strict pareto points from extended set for visualization
        extended_only_mask = extended_pareto_mask & ~pareto_mask
        extended_only_points = points[extended_only_mask]
    
    # Create the 3D plot with better styling
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set better background and grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)
    
    # Plot all points with better styling
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c='lightblue', alpha=0.3, s=20, label='All Points',
               edgecolor='none')
    
    # Plot extended Pareto points if included
    if include_extended and len(extended_only_points) > 0:
        ax.scatter(extended_only_points[:, 0], 
                   extended_only_points[:, 1], 
                   extended_only_points[:, 2],
                   c='orange', s=80, alpha=0.8, 
                   label=f'ε-Pareto Points (ε={epsilon})',
                   edgecolor='darkorange', linewidth=0.5)
    
    # Plot strict Pareto efficient points with better visibility
    ax.scatter(strict_pareto_points[:, 0], 
               strict_pareto_points[:, 1], 
               strict_pareto_points[:, 2],
               c='red', s=120, alpha=1.0, 
               label='Strict Pareto Points',
               edgecolor='darkred', linewidth=1.0,
               marker='o')
    
    # Create surfaces for better visualization
    surface_points = extended_pareto_points if include_extended else strict_pareto_points
    
    if len(surface_points) >= 4:
        try:
            # Create convex hull of Pareto points
            hull = ConvexHull(surface_points)
            
            print(f"Successfully created convex hull with {len(hull.simplices)} faces")
            
            # Plot the triangular faces of the convex hull with better styling
            for i, simplex in enumerate(hull.simplices):
                triangle = surface_points[simplex]
                
                # Check if triangle is valid (not degenerate)
                if not np.allclose(triangle[0], triangle[1]) and not np.allclose(triangle[1], triangle[2]) and not np.allclose(triangle[0], triangle[2]):
                    # Create a more visible surface with gradient coloring
                    ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                                   alpha=0.4, color='yellow', 
                                   linewidth=1.0, edgecolor='orange',
                                   antialiased=True)
            
            # Add wireframe for better structure visibility - but only if we have enough points
            if len(surface_points) >= 8:
                try:
                    ax.plot_trisurf(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2],
                                   alpha=0.1, color='red', linewidth=0.8, 
                                   edgecolor='darkred', antialiased=True)
                except:
                    print("Could not create wireframe overlay")
            
        except Exception as e:
            print(f"Could not create convex hull surface: {e}")
            print(f"Number of surface points: {len(surface_points)}")
            print("Attempting alternative surface visualization...")
            
            # Fallback: create a simpler surface visualization
            try:
                from scipy.spatial.distance import pdist, squareform
                
                # Create a mesh-like connection between nearby Pareto points
                if len(surface_points) >= 3:
                    # Find nearest neighbors for each point
                    distances = squareform(pdist(surface_points))
                    
                    print(f"Creating mesh connections between {len(surface_points)} points")
                    
                    for i in range(len(surface_points)):
                        # Connect to 2-3 nearest neighbors (fewer connections to avoid clutter)
                        n_connections = min(3, len(surface_points) - 1)
                        nearest_indices = np.argsort(distances[i])[1:n_connections+1]  # Skip self (index 0)
                        
                        for j in nearest_indices:
                            if j < len(surface_points) and distances[i][j] > 0:  # Avoid self-connections
                                ax.plot([surface_points[i, 0], surface_points[j, 0]],
                                       [surface_points[i, 1], surface_points[j, 1]], 
                                       [surface_points[i, 2], surface_points[j, 2]],
                                       color='orange', alpha=0.6, linewidth=1.5)
                                
            except Exception as e2:
                print(f"Could not create mesh surface: {e2}")
                print("Skipping surface visualization - showing points only")
    else:
        print(f"Not enough points ({len(surface_points)}) for surface visualization. Need at least 4 points.")
    
    # Add additional surface visualization techniques
    if len(surface_points) >= 8:
        try:
            # Create a 3D interpolated surface using griddata
            from scipy.interpolate import griddata
            
            # Create a regular grid in the bounding box of Pareto points
            x_min, x_max = surface_points[:, 0].min(), surface_points[:, 0].max()
            y_min, y_max = surface_points[:, 1].min(), surface_points[:, 1].max()
            z_min, z_max = surface_points[:, 2].min(), surface_points[:, 2].max()
            
            # Create grid points
            grid_density = 10
            x_grid = np.linspace(x_min, x_max, grid_density)
            y_grid = np.linspace(y_min, y_max, grid_density)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            
            # Interpolate Z values
            Z_grid = griddata((surface_points[:, 0], surface_points[:, 1]), 
                             surface_points[:, 2], 
                             (X_grid, Y_grid), method='linear')
            
            # Plot the interpolated surface
            ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, 
                           cmap='viridis', linewidth=0.5, 
                           edgecolor='none', antialiased=True)
            
        except Exception as e:
            print(f"Could not create interpolated surface: {e}")
    
    # Set labels and title with clear indication of minimization
    ax.set_xlabel(f'{x_col}' + (' (maximize)' if x_col in maximize_cols else ' (minimize ↓)'))
    ax.set_ylabel(f'{y_col}' + (' (maximize)' if y_col in maximize_cols else ' (minimize ↓)'))
    ax.set_zlabel(f'{z_col}' + (' (maximize)' if z_col in maximize_cols else ' (minimize ↓)'))
    
    title = '3D Pareto Frontier Analysis - MINIMIZATION'
    if include_extended:
        title += f' (with ε-dominance, ε={epsilon})'
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add legend with better positioning
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                      frameon=True, fancybox=True, shadow=True,
                      fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Improve the view with better angles
    ax.view_init(elev=25, azim=45)
    
    # Add some styling improvements
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.zaxis.label.set_size(12)
    
    # Set equal aspect ratio and zoom to Pareto region
    try:
        # Focus on the Pareto points region for better visualization
        pareto_region_points = extended_pareto_points if include_extended else strict_pareto_points
        
        if len(pareto_region_points) > 0:
            # Calculate the bounding box of Pareto points with some padding
            x_min, x_max = pareto_region_points[:, 0].min(), pareto_region_points[:, 0].max()
            y_min, y_max = pareto_region_points[:, 1].min(), pareto_region_points[:, 1].max()
            z_min, z_max = pareto_region_points[:, 2].min(), pareto_region_points[:, 2].max()
            
            # Add padding (20% of range)
            x_padding = (x_max - x_min) * 0.2
            y_padding = (y_max - y_min) * 0.2
            z_padding = (z_max - z_min) * 0.2
            
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
            ax.set_zlim(z_min - z_padding, z_max + z_padding)
        else:
            # Fallback to full data range
            x_range = points[:, 0].max() - points[:, 0].min()
            y_range = points[:, 1].max() - points[:, 1].min()
            z_range = points[:, 2].max() - points[:, 2].min()
            
            max_range = max(x_range, y_range, z_range)
            x_center = (points[:, 0].max() + points[:, 0].min()) / 2
            y_center = (points[:, 1].max() + points[:, 1].min()) / 2
            z_center = (points[:, 2].max() + points[:, 2].min()) / 2
            
            ax.set_xlim(x_center - max_range/2*1.1, x_center + max_range/2*1.1)
            ax.set_ylim(y_center - max_range/2*1.1, y_center + max_range/2*1.1)
            ax.set_zlim(z_center - max_range/2*1.1, z_center + max_range/2*1.1)
    except:
        pass
    
    
    # Add interactive instructions
    print(f"\n{'='*60}")
    print("INTERACTIVE ZOOM INSTRUCTIONS:")
    print("• Left mouse: Rotate the view")
    print("• Right mouse: Pan/translate the view")  
    print("• Scroll wheel: Zoom in/out")
    print("• Middle mouse: Pan the view")
    print("• Press 'r' key: Reset view to default")
    print(f"{'='*60}")
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics and verify the results
    print(f"Total points: {len(points)}")
    print(f"Data ranges:")
    print(f"  {x_col}: {points[:, 0].min():.4f} to {points[:, 0].max():.4f}")
    print(f"  {y_col}: {points[:, 1].min():.4f} to {points[:, 1].max():.4f}")
    print(f"  {z_col}: {points[:, 2].min():.4f} to {points[:, 2].max():.4f}")
    
    print(f"\nStrict Pareto efficient points: {len(strict_pareto_points)}")
    print(f"Best (minimum) values in Pareto set:")
    print(f"  {x_col}: {strict_pareto_points[:, 0].min():.4f}")
    print(f"  {y_col}: {strict_pareto_points[:, 1].min():.4f}")
    print(f"  {z_col}: {strict_pareto_points[:, 2].min():.4f}")
    
    if include_extended:
        print(f"Extended (ε-Pareto) efficient points: {len(extended_pareto_points)}")
        print(f"Additional ε-Pareto points: {len(extended_only_points)}")
        print(f"Extended Pareto efficiency ratio: {len(extended_pareto_points)/len(points)*100:.2f}%")
        print(f"Best (minimum) values in extended Pareto set:")
        print(f"  {x_col}: {extended_pareto_points[:, 0].min():.4f}")
        print(f"  {y_col}: {extended_pareto_points[:, 1].min():.4f}")
        print(f"  {z_col}: {extended_pareto_points[:, 2].min():.4f}")
    
    print(f"Strict Pareto efficiency ratio: {len(strict_pareto_points)/len(points)*100:.2f}%")
    
    results = {
        'df': df,
        'all_points': points,
        'strict_pareto_points': strict_pareto_points,
        'strict_pareto_mask': pareto_mask
    }
    
    if include_extended:
        results['extended_pareto_points'] = extended_pareto_points
        results['extended_pareto_mask'] = extended_pareto_mask
        return results
    
    return results

def analyze_pareto_tradeoffs(results, x_col, y_col, z_col):
    """
    Create a separate, detailed plot focused on the Pareto surface
    """
    surface_points = results.get('extended_pareto_points', results['strict_pareto_points'])
    strict_points = results['strict_pareto_points']
    
    if len(surface_points) < 4:
        print("Not enough Pareto points for detailed surface visualization")
        return
    
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Pareto surface with convex hull
    ax1 = fig.add_subplot(131, projection='3d')
    
    try:
        hull = ConvexHull(surface_points)
        
        # Plot the surface faces
        for simplex in hull.simplices:
            triangle = surface_points[simplex]
            ax1.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                            alpha=0.6, color='lightcoral', 
                            linewidth=1.2, edgecolor='red')
        
        # Add the Pareto points
        ax1.scatter(strict_points[:, 0], strict_points[:, 1], strict_points[:, 2],
                   c='darkred', s=100, alpha=1.0, edgecolor='black', linewidth=1)
        
        ax1.set_title('Convex Hull Surface', fontsize=12, fontweight='bold')
        ax1.set_xlabel(f'{x_col} (minimize ↓)')
        ax1.set_ylabel(f'{y_col} (minimize ↓)')
        ax1.set_zlabel(f'{z_col} (minimize ↓)')
        ax1.view_init(elev=20, azim=45)
        
    except Exception as e:
        print(f"Could not create convex hull: {e}")
    
    # Plot 2: Wireframe connections
    ax2 = fig.add_subplot(132, projection='3d')
    
    try:
        from scipy.spatial.distance import pdist, squareform
        
        # Create wireframe between nearby points
        distances = squareform(pdist(surface_points))
        
        # Plot connections
        for i in range(len(surface_points)):
            nearest_indices = np.argsort(distances[i])[1:4]  # Connect to 3 nearest
            
            for j in nearest_indices:
                if j < len(surface_points):
                    ax2.plot([surface_points[i, 0], surface_points[j, 0]],
                            [surface_points[i, 1], surface_points[j, 1]], 
                            [surface_points[i, 2], surface_points[j, 2]],
                            color='blue', alpha=0.7, linewidth=1.5)
        
        # Add the points
        ax2.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2],
                   c='blue', s=80, alpha=0.9, edgecolor='navy', linewidth=1)
        
        ax2.set_title('Wireframe Network', fontsize=12, fontweight='bold')
        ax2.set_xlabel(f'{x_col} (minimize ↓)')
        ax2.set_ylabel(f'{y_col} (minimize ↓)')
        ax2.set_zlabel(f'{z_col} (minimize ↓)')
        ax2.view_init(elev=20, azim=45)
        
    except Exception as e:
        print(f"Could not create wireframe: {e}")
    
    # Plot 3: Interpolated surface
    ax3 = fig.add_subplot(133, projection='3d')
    
    try:
        from scipy.interpolate import griddata
        
        # Create interpolated surface
        x_min, x_max = surface_points[:, 0].min(), surface_points[:, 0].max()
        y_min, y_max = surface_points[:, 1].min(), surface_points[:, 1].max()
        
        grid_density = 15
        x_grid = np.linspace(x_min, x_max, grid_density)
        y_grid = np.linspace(y_min, y_max, grid_density)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        Z_grid = griddata((surface_points[:, 0], surface_points[:, 1]), 
                         surface_points[:, 2], 
                         (X_grid, Y_grid), method='cubic', fill_value=np.nan)
        
        # Remove NaN values
        mask = ~np.isnan(Z_grid)
        if np.any(mask):
            surf = ax3.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.7, 
                                   cmap='viridis', linewidth=0.3, 
                                   edgecolor='black', antialiased=True)
            
            # Add colorbar
            fig.colorbar(surf, ax=ax3, shrink=0.8, aspect=20, 
                        label=f'{z_col} values')
        
        # Add the original points
        ax3.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2],
                   c='red', s=60, alpha=1.0, edgecolor='darkred', linewidth=1)
        
        ax3.set_title('Interpolated Surface', fontsize=12, fontweight='bold')
        ax3.set_xlabel(f'{x_col} (minimize ↓)')
        ax3.set_ylabel(f'{y_col} (minimize ↓)')
        ax3.set_zlabel(f'{z_col} (minimize ↓)')
        ax3.view_init(elev=20, azim=45)
        
    except Exception as e:
        print(f"Could not create interpolated surface: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Create a 2D projection plot for better understanding
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY projection
    axes[0].scatter(surface_points[:, 0], surface_points[:, 1], 
                   c=surface_points[:, 2], s=80, alpha=0.8, 
                   cmap='viridis', edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel(f'{x_col} (minimize ↓)')
    axes[0].set_ylabel(f'{y_col} (minimize ↓)')
    axes[0].set_title(f'XY Projection (colored by {z_col})')
    axes[0].grid(True, alpha=0.3)
    
    # XZ projection
    axes[1].scatter(surface_points[:, 0], surface_points[:, 2], 
                   c=surface_points[:, 1], s=80, alpha=0.8, 
                   cmap='plasma', edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel(f'{x_col} (minimize ↓)')
    axes[1].set_ylabel(f'{z_col} (minimize ↓)')
    axes[1].set_title(f'XZ Projection (colored by {y_col})')
    axes[1].grid(True, alpha=0.3)
    
    # YZ projection
    axes[2].scatter(surface_points[:, 1], surface_points[:, 2], 
                   c=surface_points[:, 0], s=80, alpha=0.8, 
                   cmap='coolwarm', edgecolor='black', linewidth=0.5)
    axes[2].set_xlabel(f'{y_col} (minimize ↓)')
    axes[2].set_ylabel(f'{z_col} (minimize ↓)')
    axes[2].set_title(f'YZ Projection (colored by {x_col})')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_zoomed_pareto_plot(results, x_col, y_col, z_col, zoom_factor=2.0):
    """
    Create a zoomed-in plot focused specifically on the Pareto surface region
    """
    strict_points = results['strict_pareto_points']
    extended_points = results.get('extended_pareto_points', strict_points)
    all_points = results['all_points']
    
    if len(strict_points) == 0:
        print("No Pareto points found for zoomed visualization")
        return
    
    # Calculate the focus region around Pareto points
    focus_points = extended_points if len(extended_points) > len(strict_points) else strict_points
    
    x_center = focus_points[:, 0].mean()
    y_center = focus_points[:, 1].mean()
    z_center = focus_points[:, 2].mean()
    
    x_range = focus_points[:, 0].max() - focus_points[:, 0].min()
    y_range = focus_points[:, 1].max() - focus_points[:, 1].min()
    z_range = focus_points[:, 2].max() - focus_points[:, 2].min()
    
    # Expand the range by zoom_factor
    zoom_range = max(x_range, y_range, z_range) * zoom_factor / 2
    
    # Filter points within the zoom region
    x_mask = (all_points[:, 0] >= x_center - zoom_range) & (all_points[:, 0] <= x_center + zoom_range)
    y_mask = (all_points[:, 1] >= y_center - zoom_range) & (all_points[:, 1] <= y_center + zoom_range)
    z_mask = (all_points[:, 2] >= z_center - zoom_range) & (all_points[:, 2] <= z_center + zoom_range)
    
    zoom_mask = x_mask & y_mask & z_mask
    zoom_points = all_points[zoom_mask]
    
    print(f"\nZoomed view: Showing {len(zoom_points)} points out of {len(all_points)} total points")
    print(f"Focus region centered at: ({x_center:.4f}, {y_center:.4f}, {z_center:.4f})")
    print(f"Zoom range: ±{zoom_range:.4f} in each direction")
    
    # Create the zoomed plot
    fig = plt.figure(figsize=(18, 12))
    
    # Main zoomed 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot zoomed region points
    if len(zoom_points) > 0:
        ax1.scatter(zoom_points[:, 0], zoom_points[:, 1], zoom_points[:, 2], 
                   c='lightblue', alpha=0.5, s=40, label='Nearby Points')
    
    # Plot Pareto points with enhanced visibility
    ax1.scatter(strict_points[:, 0], strict_points[:, 1], strict_points[:, 2],
               c='red', s=200, alpha=1.0, 
               label='Strict Pareto Points', 
               edgecolor='darkred', linewidth=2, marker='o')
    
    if 'extended_pareto_points' in results:
        extended_only_mask = results['extended_pareto_mask'] & ~results['strict_pareto_mask']
        if np.any(extended_only_mask):
            extended_only = all_points[extended_only_mask]
            ax1.scatter(extended_only[:, 0], extended_only[:, 1], extended_only[:, 2],
                       c='orange', s=120, alpha=0.8, 
                       label='ε-Pareto Points',
                       edgecolor='darkorange', linewidth=1.5, marker='s')
    
    # Create detailed surface in zoomed region
    surface_points = extended_points
    if len(surface_points) >= 4:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(surface_points)
            
            for simplex in hull.simplices:
                triangle = surface_points[simplex]
                ax1.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                               alpha=0.6, color='yellow', 
                               linewidth=2.0, edgecolor='orange')
        except Exception as e:
            print(f"Could not create surface in zoomed view: {e}")
    
    ax1.set_xlim(x_center - zoom_range, x_center + zoom_range)
    ax1.set_ylim(y_center - zoom_range, y_center + zoom_range)
    ax1.set_zlim(z_center - zoom_range, z_center + zoom_range)
    
    ax1.set_xlabel(f'{x_col} (minimize ↓)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'{y_col} (minimize ↓)', fontsize=12, fontweight='bold')
    ax1.set_zlabel(f'{z_col} (minimize ↓)', fontsize=12, fontweight='bold')
    ax1.set_title('ZOOMED: Pareto Surface Detail', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.view_init(elev=20, azim=45)
    ax1.grid(True, alpha=0.3)
    
    # Side view 1 (XY plane)
    ax2 = fig.add_subplot(222)
    if len(zoom_points) > 0:
        ax2.scatter(zoom_points[:, 0], zoom_points[:, 1], 
                   c='lightblue', alpha=0.5, s=30, label='Nearby Points')
    ax2.scatter(strict_points[:, 0], strict_points[:, 1], 
               c='red', s=100, alpha=1.0, edgecolor='darkred', linewidth=1.5)
    ax2.set_xlabel(f'{x_col} (minimize ↓)')
    ax2.set_ylabel(f'{y_col} (minimize ↓)')
    ax2.set_title('XY View (Top Down)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_center - zoom_range, x_center + zoom_range)
    ax2.set_ylim(y_center - zoom_range, y_center + zoom_range)
    
    # Side view 2 (XZ plane)
    ax3 = fig.add_subplot(223)
    if len(zoom_points) > 0:
        ax3.scatter(zoom_points[:, 0], zoom_points[:, 2], 
                   c='lightblue', alpha=0.5, s=30, label='Nearby Points')
    ax3.scatter(strict_points[:, 0], strict_points[:, 2], 
               c='red', s=100, alpha=1.0, edgecolor='darkred', linewidth=1.5)
    ax3.set_xlabel(f'{x_col} (minimize ↓)')
    ax3.set_ylabel(f'{z_col} (minimize ↓)')
    ax3.set_title('XZ View (Side View)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(x_center - zoom_range, x_center + zoom_range)
    ax3.set_ylim(z_center - zoom_range, z_center + zoom_range)
    
    # Side view 3 (YZ plane)
    ax4 = fig.add_subplot(224)
    if len(zoom_points) > 0:
        ax4.scatter(zoom_points[:, 1], zoom_points[:, 2], 
                   c='lightblue', alpha=0.5, s=30, label='Nearby Points')
    ax4.scatter(strict_points[:, 1], strict_points[:, 2], 
               c='red', s=100, alpha=1.0, edgecolor='darkred', linewidth=1.5)
    ax4.set_xlabel(f'{y_col} (minimize ↓)')
    ax4.set_ylabel(f'{z_col} (minimize ↓)')
    ax4.set_title('YZ View (Front View)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(y_center - zoom_range, y_center + zoom_range)
    ax4.set_ylim(z_center - zoom_range, z_center + zoom_range)
    
    plt.suptitle(f'Zoomed Pareto Analysis (Zoom Factor: {zoom_factor}x)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Create an ultra-zoomed plot focusing only on Pareto points
    fig2, ax = plt.subplots(1, 1, figsize=(12, 10), subplot_kw={'projection': '3d'})
    
    # Calculate tighter bounds around just the Pareto points
    strict_padding = max(x_range, y_range, z_range) * 0.1
    
    ax.scatter(strict_points[:, 0], strict_points[:, 1], strict_points[:, 2],
               c='red', s=300, alpha=1.0, 
               label='Strict Pareto Points', 
               edgecolor='black', linewidth=2)
    
    # Add point labels for better identification
    for i, point in enumerate(strict_points):
        ax.text(point[0], point[1], point[2], f'P{i+1}', 
               fontsize=8, ha='right', va='bottom')
    
    # Create surface if possible
    if len(strict_points) >= 4:
        try:
            hull = ConvexHull(strict_points)
            for simplex in hull.simplices:
                triangle = strict_points[simplex]
                ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                               alpha=0.7, color='lightcoral', 
                               linewidth=1.5, edgecolor='red')
        except:
            pass
    
    # Ultra-tight zoom
    ax.set_xlim(focus_points[:, 0].min() - strict_padding, 
                focus_points[:, 0].max() + strict_padding)
    ax.set_ylim(focus_points[:, 1].min() - strict_padding, 
                focus_points[:, 1].max() + strict_padding)
    ax.set_zlim(focus_points[:, 2].min() - strict_padding, 
                focus_points[:, 2].max() + strict_padding)
    
    ax.set_xlabel(f'{x_col} (minimize ↓)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{y_col} (minimize ↓)', fontsize=12, fontweight='bold')
    ax.set_zlabel(f'{z_col} (minimize ↓)', fontsize=12, fontweight='bold')
    ax.set_title('ULTRA-ZOOM: Pareto Points Only', fontsize=14, fontweight='bold')
    ax.legend()
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return zoom_points, strict_points
    """
    Create a separate, detailed plot focused on the Pareto surface
    """
    surface_points = results.get('extended_pareto_points', results['strict_pareto_points'])
    strict_points = results['strict_pareto_points']
    
    if len(surface_points) < 4:
        print("Not enough Pareto points for detailed surface visualization")
        return
    
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Pareto surface with convex hull
    ax1 = fig.add_subplot(131, projection='3d')
    
    try:
        hull = ConvexHull(surface_points)
        
        # Plot the surface faces
        for simplex in hull.simplices:
            triangle = surface_points[simplex]
            ax1.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                            alpha=0.6, color='lightcoral', 
                            linewidth=1.2, edgecolor='red')
        
        # Add the Pareto points
        ax1.scatter(strict_points[:, 0], strict_points[:, 1], strict_points[:, 2],
                   c='darkred', s=100, alpha=1.0, edgecolor='black', linewidth=1)
        
        ax1.set_title('Convex Hull Surface', fontsize=12, fontweight='bold')
        ax1.set_xlabel(f'{x_col} (minimize ↓)')
        ax1.set_ylabel(f'{y_col} (minimize ↓)')
        ax1.set_zlabel(f'{z_col} (minimize ↓)')
        ax1.view_init(elev=20, azim=45)
        
    except Exception as e:
        print(f"Could not create convex hull: {e}")
    
    # Plot 2: Wireframe connections
    ax2 = fig.add_subplot(132, projection='3d')
    
    try:
        from scipy.spatial.distance import pdist, squareform
        
        # Create wireframe between nearby points
        distances = squareform(pdist(surface_points))
        
        # Plot connections
        for i in range(len(surface_points)):
            nearest_indices = np.argsort(distances[i])[1:4]  # Connect to 3 nearest
            
            for j in nearest_indices:
                if j < len(surface_points):
                    ax2.plot([surface_points[i, 0], surface_points[j, 0]],
                            [surface_points[i, 1], surface_points[j, 1]], 
                            [surface_points[i, 2], surface_points[j, 2]],
                            color='blue', alpha=0.7, linewidth=1.5)
        
        # Add the points
        ax2.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2],
                   c='blue', s=80, alpha=0.9, edgecolor='navy', linewidth=1)
        
        ax2.set_title('Wireframe Network', fontsize=12, fontweight='bold')
        ax2.set_xlabel(f'{x_col} (minimize ↓)')
        ax2.set_ylabel(f'{y_col} (minimize ↓)')
        ax2.set_zlabel(f'{z_col} (minimize ↓)')
        ax2.view_init(elev=20, azim=45)
        
    except Exception as e:
        print(f"Could not create wireframe: {e}")
    
    # Plot 3: Interpolated surface
    ax3 = fig.add_subplot(133, projection='3d')
    
    try:
        from scipy.interpolate import griddata
        
        # Create interpolated surface
        x_min, x_max = surface_points[:, 0].min(), surface_points[:, 0].max()
        y_min, y_max = surface_points[:, 1].min(), surface_points[:, 1].max()
        
        grid_density = 15
        x_grid = np.linspace(x_min, x_max, grid_density)
        y_grid = np.linspace(y_min, y_max, grid_density)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        Z_grid = griddata((surface_points[:, 0], surface_points[:, 1]), 
                         surface_points[:, 2], 
                         (X_grid, Y_grid), method='cubic', fill_value=np.nan)
        
        # Remove NaN values
        mask = ~np.isnan(Z_grid)
        if np.any(mask):
            surf = ax3.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.7, 
                                   cmap='viridis', linewidth=0.3, 
                                   edgecolor='black', antialiased=True)
            
            # Add colorbar
            fig.colorbar(surf, ax=ax3, shrink=0.8, aspect=20, 
                        label=f'{z_col} values')
        
        # Add the original points
        ax3.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2],
                   c='red', s=60, alpha=1.0, edgecolor='darkred', linewidth=1)
        
        ax3.set_title('Interpolated Surface', fontsize=12, fontweight='bold')
        ax3.set_xlabel(f'{x_col} (minimize ↓)')
        ax3.set_ylabel(f'{y_col} (minimize ↓)')
        ax3.set_zlabel(f'{z_col} (minimize ↓)')
        ax3.view_init(elev=20, azim=45)
        
    except Exception as e:
        print(f"Could not create interpolated surface: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Create a 2D projection plot for better understanding
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY projection
    axes[0].scatter(surface_points[:, 0], surface_points[:, 1], 
                   c=surface_points[:, 2], s=80, alpha=0.8, 
                   cmap='viridis', edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel(f'{x_col} (minimize ↓)')
    axes[0].set_ylabel(f'{y_col} (minimize ↓)')
    axes[0].set_title(f'XY Projection (colored by {z_col})')
    axes[0].grid(True, alpha=0.3)
    
    # XZ projection
    axes[1].scatter(surface_points[:, 0], surface_points[:, 2], 
                   c=surface_points[:, 1], s=80, alpha=0.8, 
                   cmap='plasma', edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel(f'{x_col} (minimize ↓)')
    axes[1].set_ylabel(f'{z_col} (minimize ↓)')
    axes[1].set_title(f'XZ Projection (colored by {y_col})')
    axes[1].grid(True, alpha=0.3)
    
    # YZ projection
    axes[2].scatter(surface_points[:, 1], surface_points[:, 2], 
                   c=surface_points[:, 0], s=80, alpha=0.8, 
                   cmap='coolwarm', edgecolor='black', linewidth=0.5)
    axes[2].set_xlabel(f'{y_col} (minimize ↓)')
    axes[2].set_ylabel(f'{z_col} (minimize ↓)')
    axes[2].set_title(f'YZ Projection (colored by {x_col})')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    """
    Analyze the trade-offs in the Pareto frontier
    """
    strict_pareto_points = results['strict_pareto_points']
    
    print("\n=== Strict Pareto Frontier Analysis (MINIMIZATION) ===")
    print(f"Strict Pareto efficient points: {len(strict_pareto_points)}")
    print(f"These are points where you cannot improve (reduce) any objective")
    print(f"without making at least one other objective worse (higher).\n")
    
    if len(strict_pareto_points) > 0:
        pareto_df = pd.DataFrame(strict_pareto_points, columns=[x_col, y_col, z_col])
        if len(strict_pareto_points) <= 20:
            print("All strict Pareto efficient points:")
            print(pareto_df.round(4))
        else:
            print("Sample of strict Pareto efficient points (first 10 and last 10):")
            print(pareto_df.head(10).round(4))
            print("...")
            print(pareto_df.tail(10).round(4))
        
        print(f"\nRange of strict Pareto efficient points:")
        print(f"{x_col}: {strict_pareto_points[:, 0].min():.4f} to {strict_pareto_points[:, 0].max():.4f}")
        print(f"{y_col}: {strict_pareto_points[:, 1].min():.4f} to {strict_pareto_points[:, 1].max():.4f}")
        print(f"{z_col}: {strict_pareto_points[:, 2].min():.4f} to {strict_pareto_points[:, 2].max():.4f}")
        
        print(f"\nBest (lowest) individual values found in Pareto set:")
        print(f"Minimum {x_col}: {strict_pareto_points[:, 0].min():.4f}")
        print(f"Minimum {y_col}: {strict_pareto_points[:, 1].min():.4f}")
        print(f"Minimum {z_col}: {strict_pareto_points[:, 2].min():.4f}")
    else:
        print("No strict Pareto efficient points found!")
    
    # If extended Pareto points exist, analyze them too
    if 'extended_pareto_points' in results and len(results['extended_pareto_points']) > 0:
        extended_pareto_points = results['extended_pareto_points']
        print(f"\n=== Extended (ε-Pareto) Frontier Analysis (MINIMIZATION) ===")
        print(f"Extended Pareto efficient points: {len(extended_pareto_points)}")
        print(f"These include nearly-optimal points within {epsilon*100:.1f}% tolerance.\n")
        
        extended_df = pd.DataFrame(extended_pareto_points, columns=[x_col, y_col, z_col])
        print(f"Range of extended Pareto efficient points:")
        print(f"{x_col}: {extended_pareto_points[:, 0].min():.4f} to {extended_pareto_points[:, 0].max():.4f}")
        print(f"{y_col}: {extended_pareto_points[:, 1].min():.4f} to {extended_pareto_points[:, 1].max():.4f}")
        print(f"{z_col}: {extended_pareto_points[:, 2].min():.4f} to {extended_pareto_points[:, 2].max():.4f}")
        
        print(f"\nBest (lowest) individual values in extended Pareto set:")
        print(f"Minimum {x_col}: {extended_pareto_points[:, 0].min():.4f}")
        print(f"Minimum {y_col}: {extended_pareto_points[:, 1].min():.4f}")
        print(f"Minimum {z_col}: {extended_pareto_points[:, 2].min():.4f}")
        
        # Show some sample extended Pareto points
        if len(extended_pareto_points) <= 15:
            print(f"\nAll extended Pareto points:")
            print(extended_df.round(4))
        else:
            print(f"\nSample extended Pareto points (showing 10 best):")
            # Sort by sum to show "best" points first
            point_sums = extended_pareto_points.sum(axis=1)
            best_indices = np.argsort(point_sums)[:10]
            print(extended_df.iloc[best_indices].round(4))

# Example usage with your specific data:
if __name__ == "__main__":
    # CORRECT usage - pass the file path first, not results
    results = plot_3d_pareto(
        r'C:\Users\U375383\Documents\work\results\results1000.csv', 
        'objD', 'objSHw', 'objHF',
        maximize_cols=[],  # All objectives will be minimized
        include_extended=True,  # Include epsilon-Pareto points
        epsilon=0.05  # 5% tolerance - increase for more points, decrease for fewer
    )
    
    # Analyze the results
    analyze_pareto_tradeoffs(results, 'objD', 'objSHw', 'objHF')
    
    # Create detailed surface visualization
    print("\n" + "="*60)
    print("Creating detailed Pareto surface visualizations...")
    create_detailed_pareto_surface_plot(results, 'objD', 'objSHw', 'objHF', epsilon=0.05)
    
    # Create zoomed-in plots
    print("\n" + "="*60) 
    print("Creating zoomed-in Pareto surface analysis...")
    zoom_points, pareto_points = create_zoomed_pareto_plot(results, 'objD', 'objSHw', 'objHF', zoom_factor=2.0)
    
    # Option to create even more zoomed plot
    print("\n" + "="*40)
    print("Creating ultra-zoomed view...")
    create_zoomed_pareto_plot(results, 'objD', 'objSHw', 'objHF', zoom_factor=1.2)
    
    # You can also try different epsilon values:
    print("\n" + "="*50)
    print("Trying with larger epsilon for more points...")
    results_extended = plot_3d_pareto(
        r'C:\Users\U375383\Documents\work\results\results1000.csv', 
        'objD', 'objSHw', 'objHF',
        maximize_cols=[],
        include_extended=True,
        epsilon=0.1  # 10% tolerance for even more points
    )
    
    # Analyze the extended results too
    analyze_pareto_tradeoffs(results_extended, 'objD', 'objSHw', 'objHF')

# To use with your CSV file with different settings:
# results = plot_3d_pareto(r'C:\Users\U375383\Documents\work\results\results1000.csv', 'objD', 'objSHw', 'objHF', epsilon=0.1)

# To use with your CSV file with different settings:
# results = plot_3d_pareto(r'C:\Users\U375383\Documents\work\results\results1000.csv', 'objD', 'objSHw', 'objHF', epsilon=0.1)

print("\n=== CORRECT USAGE INSTRUCTIONS ===")
print("❌ WRONG: plot_3d_pareto(results, 'objD', 'objSHw', 'objHF')")  
print("✅ CORRECT: results = plot_3d_pareto(r'path/to/file.csv', 'objD', 'objSHw', 'objHF')")
print("")
print("The first parameter must be the CSV file path, not the results dictionary!")
print("")
print("🔍 ZOOM AND INTERACTION GUIDE:")
print("The code now provides multiple zoom levels and interaction methods:")
print("")
print("🔍 INTERACTIVE CONTROLS (matplotlib 3D plots):")
print("   • Left mouse drag: Rotate the 3D view") 
print("   • Right mouse drag: Pan/translate the view")
print("   • Mouse wheel: Zoom in/out")
print("   • Middle mouse: Pan")
print("")
print("📊 AUTOMATIC ZOOM PLOTS:")
print("   • Main plot: Automatically zoomed to Pareto region")
print("   • Detailed surface plot: 3-panel analysis")
print("   • Zoomed plot: 4-panel view focused on Pareto surface")
print("   • Ultra-zoom plot: Tight focus on just Pareto points")
print("")
print("⚙️  CUSTOMIZATION:")
print("   • Adjust zoom_factor parameter (1.2 = tight, 3.0 = wide)")
print("   • Change epsilon for more/fewer Pareto points")
print("   • All plots are interactive - use mouse to explore!")
print("")
print("🎯 TIP: Look for the red points (strict Pareto) and orange points (ε-Pareto)")
print("      The yellow/coral surfaces show the Pareto frontier boundary")
print("      If convex hull fails, the code will create mesh connections instead")