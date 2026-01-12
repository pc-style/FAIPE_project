"""
Task 3 - Self-Organizing Map on Synthetic Data
1D and 2D SOM training and visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from som import SOM
from utils import (generate_synthetic_data, save_fig, redirect_print,
                   PALETTE, COLORS, add_title_box)

def run():
    with redirect_print('03_som_synthetic_output.txt'):
        print("=" * 60)
        print("  TASK 3: SOM on Synthetic Data")
        print("=" * 60)

        # 1. Load synthetic data
        print("\n[1] Loading synthetic digit-like data...")
        data, labels = generate_synthetic_data()
        print(f"   Shape: {data.shape}")

        # Normalize data
        data_norm = (data - data.mean(axis=0)) / data.std(axis=0)

        # ============================================
        # 1D SOM
        # ============================================
        print("\n" + "-" * 50)
        print("  1D SOM (10 neurons)")
        print("-" * 50)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.patch.set_facecolor('white')
        epochs_to_show = [0, 5, 10, 25, 50, 100]
        plot_idx = 0

        som_1d = SOM(grid_size=10, input_dim=2, sigma=3.0, lr=0.5)

        for epoch in range(101):
            if epoch in epochs_to_show:
                ax = axes.flatten()[plot_idx]
                
                # Plot data
                for i, color in enumerate([PALETTE[0], PALETTE[1], PALETTE[2]]):
                    mask = labels == i
                    ax.scatter(data_norm[mask, 0], data_norm[mask, 1], 
                              c=color, alpha=0.3, s=25, edgecolors='none')
                
                # Plot SOM chain
                weights = som_1d.weights
                ax.plot(weights[:, 0], weights[:, 1], 'ko-', markersize=10, linewidth=2.5,
                       markerfacecolor=COLORS['warning'], markeredgecolor='white', markeredgewidth=1.5)
                
                for i in range(len(weights)):
                    ax.annotate(str(i), (weights[i, 0] + 0.12, weights[i, 1] + 0.12), 
                               fontsize=8, fontweight='bold', color=COLORS['dark'])
                
                add_title_box(ax, f'Epoch {epoch}')
                plot_idx += 1
            
            # Train one epoch
            if epoch < 100:
                progress = epoch / 100
                lr = 0.5 * (1 - progress)
                sigma = 3.0 * (1 - 0.9 * progress)
                
                for x in data_norm:
                    bmu = som_1d.find_bmu(x)
                    h = som_1d.neighborhood_function(bmu, sigma)
                    som_1d.weights += lr * h[:, np.newaxis] * (x - som_1d.weights)

        fig.suptitle('1D SOM Evolution - Chain Adapts to Data Distribution', 
                    fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        save_fig('03_som_1d_evolution', fig)

        print("   [OK] 1D SOM creates a 'chain' threading through data regions")

        # ============================================
        # 2D SOM
        # ============================================
        print("\n" + "-" * 50)
        print("  2D SOM (5x5 grid)")
        print("-" * 50)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.patch.set_facecolor('white')
        plot_idx = 0

        som_2d = SOM(grid_size=(5, 5), input_dim=2, sigma=2.0, lr=0.5)

        for epoch in range(101):
            if epoch in epochs_to_show:
                ax = axes.flatten()[plot_idx]
                
                # Plot data
                for i, color in enumerate([PALETTE[0], PALETTE[1], PALETTE[2]]):
                    mask = labels == i
                    ax.scatter(data_norm[mask, 0], data_norm[mask, 1], 
                              c=color, alpha=0.3, s=25, edgecolors='none')
                
                # Plot SOM grid
                weights = som_2d.weights
                rows, cols = 5, 5
                
                # Draw connections first
                for i in range(rows):
                    for j in range(cols):
                        if j < cols - 1:
                            ax.plot([weights[i, j, 0], weights[i, j+1, 0]], 
                                   [weights[i, j, 1], weights[i, j+1, 1]], 
                                   color=COLORS['dark'], linewidth=1.5, alpha=0.7)
                        if i < rows - 1:
                            ax.plot([weights[i, j, 0], weights[i+1, j, 0]], 
                                   [weights[i, j, 1], weights[i+1, j, 1]], 
                                   color=COLORS['dark'], linewidth=1.5, alpha=0.7)
                
                # Draw neurons
                for i in range(rows):
                    for j in range(cols):
                        ax.plot(weights[i, j, 0], weights[i, j, 1], 'o', 
                               markersize=9, markerfacecolor=COLORS['secondary'], 
                               markeredgecolor='white', markeredgewidth=1.2)
                
                add_title_box(ax, f'Epoch {epoch}')
                plot_idx += 1
            
            # Train one epoch
            if epoch < 100:
                progress = epoch / 100
                lr = 0.5 * (1 - progress)
                sigma = 2.0 * (1 - 0.9 * progress)
                
                for x in data_norm:
                    bmu = som_2d.find_bmu(x)
                    h = som_2d.neighborhood_function(bmu, sigma)
                    som_2d.weights += lr * h[:, :, np.newaxis] * (x - som_2d.weights)

        fig.suptitle('2D SOM Evolution - Grid Unfolds to Cover Data', 
                    fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        save_fig('03_som_2d_evolution', fig)

        print("   [OK] 2D SOM unfolds like a sheet onto the data manifold")

        # ============================================
        # COMPARISON
        # ============================================
        print("\n" + "-" * 50)
        print("  COMPARISON: 1D vs 2D")
        print("-" * 50)

        # Fresh training for final comparison
        som_1d_final = SOM(grid_size=10, input_dim=2, sigma=3.0, lr=0.5)
        som_1d_final.train(data_norm, epochs=100, verbose=False)

        som_2d_final = SOM(grid_size=(5, 5), input_dim=2, sigma=2.0, lr=0.5)
        som_2d_final.train(data_norm, epochs=100, verbose=False)

        te_1d = som_1d_final.topographic_error(data_norm)
        qe_1d = som_1d_final.quantization_error(data_norm)
        te_2d = som_2d_final.topographic_error(data_norm)
        qe_2d = som_2d_final.quantization_error(data_norm)

        print(f"\n   1D SOM: TE = {te_1d:.3f}, QE = {qe_1d:.3f}")
        print(f"   2D SOM: TE = {te_2d:.3f}, QE = {qe_2d:.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('white')

        for ax_idx, (ax, som, title, te, qe) in enumerate([
            (axes[0], som_1d_final, '1D SOM', te_1d, qe_1d),
            (axes[1], som_2d_final, '2D SOM', te_2d, qe_2d)
        ]):
            # Data
            for i, color in enumerate([PALETTE[0], PALETTE[1], PALETTE[2]]):
                mask = labels == i
                ax.scatter(data_norm[mask, 0], data_norm[mask, 1], 
                          c=color, alpha=0.4, s=30, edgecolors='none')
            
            # SOM
            if som.is_1d:
                ax.plot(som.weights[:, 0], som.weights[:, 1], 'ko-', 
                       markersize=12, linewidth=3, markerfacecolor=COLORS['warning'],
                       markeredgecolor='white', markeredgewidth=2)
            else:
                weights = som.weights
                for i in range(5):
                    for j in range(5):
                        ax.plot(weights[i, j, 0], weights[i, j, 1], 'o', 
                               markersize=10, markerfacecolor=COLORS['secondary'],
                               markeredgecolor='white', markeredgewidth=1.5)
                        if j < 4:
                            ax.plot([weights[i, j, 0], weights[i, j+1, 0]], 
                                   [weights[i, j, 1], weights[i, j+1, 1]], 
                                   color=COLORS['dark'], linewidth=2)
                        if i < 4:
                            ax.plot([weights[i, j, 0], weights[i+1, j, 0]], 
                                   [weights[i, j, 1], weights[i+1, j, 1]], 
                                   color=COLORS['dark'], linewidth=2)
            
            add_title_box(ax, f'{title}', f'TE={te:.3f}, QE={qe:.3f}')

        fig.suptitle('1D vs 2D SOM Comparison', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        save_fig('03_som_1d_vs_2d', fig)

        print("\n" + "=" * 60)
        print("  SOM Synthetic Complete!")
        print("=" * 60)
        print("\nKey Observations:")
        print("   * 1D SOM: Creates an ordering path through data")
        print("   * 2D SOM: Preserves 2D neighborhood structure")
        print("   * Learning rate decay -> gradual convergence")
        print("   * Sigma decay -> shrinking neighborhood radius")

if __name__ == '__main__':
    run()
