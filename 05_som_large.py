"""
Task 5 - Large SOM and Topology Quality
10x10 grid, varying sigma, topographic error analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from som import SOM
from utils import save_fig, redirect_print, PALETTE, COLORS, add_title_box

def run():
    with redirect_print('05_som_large_output.txt'):
        print("=" * 60)
        print("  TASK 5: Large SOM and Topology Quality")
        print("=" * 60)

        # 1. Load digits
        print("\n[1] Loading digits dataset...")
        digits = load_digits()
        X = digits.data
        y = digits.target

        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)

        # Subset for speed
        np.random.seed(42)
        subset_idx = np.random.choice(len(X), 1000, replace=False)
        X_subset = X_norm[subset_idx]
        y_subset = y[subset_idx]
        print(f"   Using {len(X_subset)} samples (8x8 images = 64 features)")

        # 2. Train with different sigma values
        print("\n[2] Training 10x10 SOMs with different final sigma...")
        sigma_finals = [0.5, 1.0, 2.0, 3.0]
        results = []

        for sigma_final in sigma_finals:
            print(f"\n   Training sigma_final = {sigma_final}...")
            
            som = SOM(grid_size=(10, 10), input_dim=64, sigma=5.0, lr=0.5)
            
            n_epochs = 100
            for epoch in range(n_epochs):
                progress = epoch / n_epochs
                lr = 0.5 * (1 - progress)
                sigma = 5.0 - (5.0 - sigma_final) * progress
                
                indices = np.random.permutation(len(X_subset))
                for idx in indices:
                    x = X_subset[idx]
                    bmu = som.find_bmu(x)
                    h = som.neighborhood_function(bmu, sigma)
                    som.weights += lr * h[:, :, np.newaxis] * (x - som.weights)
            
            te = som.topographic_error(X_subset)
            qe = som.quantization_error(X_subset)
            results.append({'sigma_final': sigma_final, 'som': som, 'te': te, 'qe': qe})
            print(f"   -> TE = {te:.3f}, QE = {qe:.3f}")

        # 3. Comparison bar chart
        print("\n[3] Creating comparison charts...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('white')

        x_pos = range(len(sigma_finals))
        tes = [r['te'] for r in results]
        qes = [r['qe'] for r in results]

        bars1 = ax1.bar(x_pos, tes, color=COLORS['secondary'], edgecolor='white', linewidth=2)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'sigma = {s}' for s in sigma_finals])
        ax1.set_ylabel('Topographic Error')
        add_title_box(ax1, 'Topographic Error', 'Lower = better topology preservation')
        
        # Highlight best
        best_te_idx = np.argmin(tes)
        bars1[best_te_idx].set_color(COLORS['success'])

        bars2 = ax2.bar(x_pos, qes, color=COLORS['primary'], edgecolor='white', linewidth=2)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'sigma = {s}' for s in sigma_finals])
        ax2.set_ylabel('Quantization Error')
        add_title_box(ax2, 'Quantization Error', 'Lower = better data representation')
        
        best_qe_idx = np.argmin(qes)
        bars2[best_qe_idx].set_color(COLORS['success'])

        plt.tight_layout()
        save_fig('05_sigma_comparison', fig)

        # 4. PCA visualization of SOM weights
        print("\n[4] Visualizing SOM grid in PCA space...")
        best_som = results[1]['som']  # sigma=1.0 usually good

        weights_flat = best_som.weights.reshape(-1, 64)
        pca = PCA(n_components=2)
        weights_2d = pca.fit_transform(weights_flat).reshape(10, 10, 2)

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('white')

        # Draw connections
        for i in range(10):
            for j in range(10):
                if j < 9:
                    ax.plot([weights_2d[i, j, 0], weights_2d[i, j+1, 0]],
                           [weights_2d[i, j, 1], weights_2d[i, j+1, 1]], 
                           color=COLORS['dark'], linewidth=1, alpha=0.5)
                if i < 9:
                    ax.plot([weights_2d[i, j, 0], weights_2d[i+1, j, 0]],
                           [weights_2d[i, j, 1], weights_2d[i+1, j, 1]], 
                           color=COLORS['dark'], linewidth=1, alpha=0.5)

        # Draw neurons
        for i in range(10):
            for j in range(10):
                ax.plot(weights_2d[i, j, 0], weights_2d[i, j, 1], 'o',
                       markersize=8, markerfacecolor=COLORS['primary'],
                       markeredgecolor='white', markeredgewidth=1)

        add_title_box(ax, '10x10 SOM Grid in PCA Space', 
                     'Smooth unfolding = good topology preservation')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        save_fig('05_som_pca_projection', fig)

        # 5. Digit distribution
        print("\n[5] Analyzing digit distribution across neurons...")
        predictions = best_som.predict(X_subset)

        digit_counts = np.zeros((100, 10))
        for pred, label in zip(predictions, y_subset):
            digit_counts[pred, label] += 1

        dominant_digits = digit_counts.argmax(axis=1).reshape(10, 10)

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('white')
        
        im = ax.imshow(dominant_digits, cmap='tab10', vmin=0, vmax=9)
        cbar = plt.colorbar(im, ax=ax, ticks=range(10))
        cbar.set_label('Dominant Digit', fontsize=11)

        for i in range(10):
            for j in range(10):
                neuron_id = i * 10 + j
                if digit_counts[neuron_id].sum() > 0:
                    ax.text(j, i, str(dominant_digits[i, j]), ha='center', va='center', 
                           color='white', fontsize=9, fontweight='bold')

        add_title_box(ax, 'Dominant Digit per Neuron', 
                     'Similar digits should cluster in adjacent regions')
        ax.set_xlabel('Neuron column')
        ax.set_ylabel('Neuron row')
        save_fig('05_digit_distribution', fig)

        # 6. Confusion pair analysis
        print("\n[6] Checking digit neighborhood relationships...")
        pairs = [(4, 9), (3, 8), (1, 7)]
        
        print("\n   Confusion pair adjacency analysis:")
        for d1, d2 in pairs:
            neurons_d1 = set(np.where(dominant_digits.flatten() == d1)[0])
            neurons_d2 = set(np.where(dominant_digits.flatten() == d2)[0])
            
            adjacent = False
            for n1 in neurons_d1:
                r1, c1 = n1 // 10, n1 % 10
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r1 + dr, c1 + dc
                    if 0 <= nr < 10 and 0 <= nc < 10:
                        neighbor = nr * 10 + nc
                        if neighbor in neurons_d2:
                            adjacent = True
                            break
            
            status = "Adjacent (expected similar)" if adjacent else "Separated"
            print(f"   {d1} vs {d2}: {status}")

        print("\n" + "=" * 60)
        print("  Large SOM Analysis Complete!")
        print("=" * 60)
        print("\nKey Trade-offs:")
        print("   * Small sigma (0.5): High resolution, more topology errors")
        print("   * Large sigma (3.0): Smoother map, less detail")
        print("   * Medium sigma (1.0-2.0): Best balance for digit data")

if __name__ == '__main__':
    run()
