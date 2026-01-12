"""
Task 6 - Neighborhood Functions and Regularization
Gaussian vs Mexican Hat comparison
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import label
from som import SOM
from utils import save_fig, redirect_print, PALETTE, COLORS, add_title_box

def run():
    with redirect_print('06_neighborhoods_output.txt'):
        print("=" * 60)
        print("  TASK 6: Neighborhood Functions and Regularization")
        print("=" * 60)

        # 1. Load digits
        print("\n[1] Loading digits dataset...")
        digits = load_digits()
        X = digits.data
        y = digits.target

        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)

        np.random.seed(42)
        subset_idx = np.random.choice(len(X), 800, replace=False)
        X_subset = X_norm[subset_idx]
        y_subset = y[subset_idx]
        print(f"   Using {len(X_subset)} samples")

        # 2. Visualize neighborhood functions
        print("\n[2] Visualizing neighborhood functions...")
        distances = np.linspace(0, 5, 200)
        sigma = 1.5

        gaussian = np.exp(-distances**2 / (2 * sigma**2))
        mexican_hat = (1 - (distances/sigma)**2) * np.exp(-distances**2 / (2 * sigma**2))

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        
        ax.plot(distances, gaussian, color=COLORS['primary'], linewidth=3, label='Gaussian')
        ax.plot(distances, mexican_hat, color=COLORS['secondary'], linewidth=3, label='Mexican Hat')
        ax.axhline(y=0, color=COLORS['dark'], linestyle='--', alpha=0.3)
        
        # Highlight inhibition zone
        ax.fill_between(distances, mexican_hat, 0, 
                       where=(mexican_hat < 0), 
                       color=COLORS['secondary'], alpha=0.2, label='Inhibition zone')
        ax.fill_between(distances, gaussian, 0, 
                       color=COLORS['primary'], alpha=0.1)
        
        add_title_box(ax, 'Neighborhood Function Comparison', 
                     f'sigma = {sigma}: Gaussian vs Mexican Hat')
        ax.set_xlabel('Distance from BMU')
        ax.set_ylabel('Influence (weight update factor)')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.5, 1.1)
        save_fig('06_neighborhood_functions', fig)

        print("   [OK] Gaussian: Always positive, smooth decay")
        print("   [OK] Mexican Hat: Inhibition zone creates sharper boundaries")

        # 3. Train both SOMs
        print("\n[3] Training SOMs with both functions...")
        
        print("\n   Training Gaussian SOM (8x8)...")
        som_gaussian = SOM(grid_size=(8, 8), input_dim=64, sigma=3.0, lr=0.5, neighborhood='gaussian')
        som_gaussian.train(X_subset, epochs=100, verbose=False)
        te_gauss = som_gaussian.topographic_error(X_subset)
        qe_gauss = som_gaussian.quantization_error(X_subset)
        print(f"   -> TE = {te_gauss:.3f}, QE = {qe_gauss:.3f}")

        print("\n   Training Mexican Hat SOM (8x8)...")
        som_mexican = SOM(grid_size=(8, 8), input_dim=64, sigma=3.0, lr=0.5, neighborhood='mexican_hat')
        som_mexican.train(X_subset, epochs=100, verbose=False)
        te_mex = som_mexican.topographic_error(X_subset)
        qe_mex = som_mexican.quantization_error(X_subset)
        print(f"   -> TE = {te_mex:.3f}, QE = {qe_mex:.3f}")

        # 4. Digit distribution comparison
        print("\n[4] Comparing digit distributions...")
        
        def get_dominant_grid(som, X, y, grid_size=8):
            predictions = som.predict(X)
            counts = np.zeros((grid_size * grid_size, 10))
            for pred, lbl in zip(predictions, y):
                counts[pred, lbl] += 1
            return counts.argmax(axis=1).reshape(grid_size, grid_size)

        dom_gauss = get_dominant_grid(som_gaussian, X_subset, y_subset)
        dom_mex = get_dominant_grid(som_mexican, X_subset, y_subset)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('white')

        im1 = ax1.imshow(dom_gauss, cmap='tab10', vmin=0, vmax=9)
        for i in range(8):
            for j in range(8):
                ax1.text(j, i, str(dom_gauss[i, j]), ha='center', va='center', 
                        color='white', fontsize=9, fontweight='bold')
        add_title_box(ax1, 'Gaussian Neighborhood', 
                     f'TE={te_gauss:.3f}, QE={qe_gauss:.3f}')
        ax1.set_xlabel('Neuron column')
        ax1.set_ylabel('Neuron row')

        im2 = ax2.imshow(dom_mex, cmap='tab10', vmin=0, vmax=9)
        for i in range(8):
            for j in range(8):
                ax2.text(j, i, str(dom_mex[i, j]), ha='center', va='center', 
                        color='white', fontsize=9, fontweight='bold')
        add_title_box(ax2, 'Mexican Hat Neighborhood', 
                     f'TE={te_mex:.3f}, QE={qe_mex:.3f}')
        ax2.set_xlabel('Neuron column')

        # Shared colorbar
        cbar = fig.colorbar(im2, ax=[ax1, ax2], ticks=range(10), shrink=0.8)
        cbar.set_label('Dominant Digit')

        fig.suptitle('Digit Distribution: Gaussian vs Mexican Hat', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_fig('06_gaussian_vs_mexican', fig)

        # 5. Cluster fragmentation analysis
        print("\n[5] Analyzing cluster fragmentation...")
        
        def count_fragments(dominant_grid):
            counts = {}
            for digit in range(10):
                mask = (dominant_grid == digit).astype(int)
                labeled, n_clusters = label(mask)
                counts[digit] = n_clusters
            return counts

        frags_gauss = count_fragments(dom_gauss)
        frags_mex = count_fragments(dom_mex)

        print("\n   Digit fragmentation (1 = single connected cluster):")
        print("   +-------+----------+-------------+")
        print("   | Digit | Gaussian | Mexican Hat |")
        print("   +-------+----------+-------------+")
        for d in range(10):
            print(f"   |   {d}   |    {frags_gauss[d]}     |      {frags_mex[d]}      |")
        print("   +-------+----------+-------------+")
        print(f"   | Total |   {sum(frags_gauss.values())}    |     {sum(frags_mex.values())}      |")
        print("   +-------+----------+-------------+")

        # 6. Summary comparison
        print("\n[6] Creating summary visualization...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('white')

        # TE comparison
        ax = axes[0]
        bars = ax.bar(['Gaussian', 'Mexican Hat'], [te_gauss, te_mex], 
                     color=[COLORS['primary'], COLORS['secondary']], 
                     edgecolor='white', linewidth=2)
        ax.set_ylabel('Topographic Error')
        add_title_box(ax, 'Topology Preservation', 'Lower = better')
        for bar, val in zip(bars, [te_gauss, te_mex]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', fontweight='bold')

        # QE comparison
        ax = axes[1]
        bars = ax.bar(['Gaussian', 'Mexican Hat'], [qe_gauss, qe_mex], 
                     color=[COLORS['primary'], COLORS['secondary']], 
                     edgecolor='white', linewidth=2)
        ax.set_ylabel('Quantization Error')
        add_title_box(ax, 'Data Representation', 'Lower = better')
        for bar, val in zip(bars, [qe_gauss, qe_mex]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val:.3f}', ha='center', fontweight='bold')

        # Fragmentation
        ax = axes[2]
        total_gauss = sum(frags_gauss.values())
        total_mex = sum(frags_mex.values())
        bars = ax.bar(['Gaussian', 'Mexican Hat'], [total_gauss, total_mex], 
                     color=[COLORS['primary'], COLORS['secondary']], 
                     edgecolor='white', linewidth=2)
        ax.axhline(y=10, color=COLORS['success'], linestyle='--', linewidth=2, label='Ideal (10)')
        ax.set_ylabel('Total Clusters')
        add_title_box(ax, 'Cluster Fragmentation', 'Closer to 10 = cleaner separation')
        ax.legend(loc='upper right')
        for bar, val in zip(bars, [total_gauss, total_mex]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(val), ha='center', fontweight='bold')

        plt.tight_layout()
        save_fig('06_comparison_summary', fig)

        print("\n" + "=" * 60)
        print("  Neighborhood Functions Analysis Complete!")
        print("=" * 60)
        print("\nKey Conclusions:")
        print("   * Gaussian: Smoother, better topology (lower TE)")
        print("   * Mexican Hat: Sharper boundaries, better separation")
        print("   * Mexican Hat inhibition = 'lateral competition'")
        print("   * This acts as REGULARIZATION:")
        print("     - Prevents 'digit blur' across adjacent neurons")
        print("     - Similar to L1/L2 preventing overfitting in supervised learning")

if __name__ == '__main__':
    run()
