"""
Task 4 - SOM on Iris Dataset (MANDATORY)
Small 3x3 grid, compare with real species
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from som import SOM
from utils import save_fig, redirect_print, PALETTE, COLORS, add_title_box

def run():
    with redirect_print('04_som_iris_output.txt'):
        print("=" * 60)
        print("  TASK 4: SOM on Iris Dataset (Mandatory)")
        print("=" * 60)

        # 1. Load and normalize
        print("\n[1] Loading Iris dataset...")
        iris = load_iris()
        X = iris.data
        y = iris.target
        species = iris.target_names

        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)

        print(f"   Shape: {X.shape}")
        print(f"   Features: {list(iris.feature_names)}")
        print(f"   Species: {list(species)}")

        # 2. Train 3x3 SOM
        print("\n[2] Training 3x3 SOM...")
        som = SOM(grid_size=(3, 3), input_dim=4, sigma=1.5, lr=0.5, neighborhood='gaussian')
        som.train(X_norm, epochs=200, verbose=False)
        print("   Training complete (200 epochs)")

        # 3. Assign samples
        predictions = som.predict(X_norm)

        # 4. Species distribution heatmap
        print("\n[3] Creating species distribution heatmap...")
        species_counts = np.zeros((3, 3, 3))
        for idx, (neuron, label) in enumerate(zip(predictions, y)):
            i, j = neuron // 3, neuron % 3
            species_counts[i, j, label] += 1

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('white')
        
        species_colors = ['Blues', 'Greens', 'Purples']
        
        for sp_idx, (ax, sp_name, cmap) in enumerate(zip(axes, species, species_colors)):
            im = ax.imshow(species_counts[:, :, sp_idx], cmap=cmap, vmin=0, vmax=50)
            
            for i in range(3):
                for j in range(3):
                    count = int(species_counts[i, j, sp_idx])
                    color = 'white' if count > 25 else COLORS['dark']
                    ax.text(j, i, str(count), ha='center', va='center', 
                           color=color, fontsize=16, fontweight='bold')
            
            add_title_box(ax, sp_name.capitalize())
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xlabel('Neuron column')
            ax.set_ylabel('Neuron row')

        fig.suptitle('Species Distribution Across 3x3 SOM Grid', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_fig('04_iris_som_heatmap', fig)

        # 5. Dominant species visualization
        print("\n[4] Analyzing dominant species per neuron...")
        dominant = species_counts.argmax(axis=2)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        
        species_palette = {0: COLORS['primary'], 1: COLORS['accent'], 2: COLORS['secondary']}
        
        for i in range(3):
            for j in range(3):
                neuron_id = i * 3 + j
                total = species_counts[i, j].sum()
                dom_sp = dominant[i, j]
                dom_count = species_counts[i, j, dom_sp]
                purity = dom_count / total if total > 0 else 0
                
                # Draw cell
                rect = plt.Rectangle((j, 2-i), 1, 1, 
                                     facecolor=species_palette[dom_sp], 
                                     alpha=0.3 + 0.5*purity, edgecolor='white', linewidth=3)
                ax.add_patch(rect)
                
                # Text
                ax.text(j + 0.5, 2.5 - i, 
                       f'N{neuron_id}\n{species[dom_sp]}\n{int(dom_count)}/{int(total)}', 
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       color=COLORS['dark'])
                
                print(f"   Neuron {neuron_id}: {species[dom_sp]} ({int(dom_count)}/{int(total)}, {purity:.0%} pure)")

        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Legend
        for sp_idx, sp_name in enumerate(species):
            ax.plot([], [], 's', color=species_palette[sp_idx], markersize=20, 
                   label=sp_name, alpha=0.7)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=11)
        
        add_title_box(ax, 'Dominant Species per Neuron', 'Opacity indicates cluster purity')
        plt.tight_layout()
        save_fig('04_iris_dominant_species', fig)

        # 6. Topology quality
        print("\n[5] Computing topology quality metrics...")
        te = som.topographic_error(X_norm)
        qe = som.quantization_error(X_norm)
        print(f"   Topographic Error: {te:.3f}")
        print(f"   Quantization Error: {qe:.3f}")

        # 7. U-Matrix
        print("\n[6] Creating U-Matrix...")
        weights = som.weights
        u_matrix = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                distances = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 3 and 0 <= nj < 3:
                        dist = np.sqrt(np.sum((weights[i, j] - weights[ni, nj]) ** 2))
                        distances.append(dist)
                u_matrix[i, j] = np.mean(distances) if distances else 0

        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor('white')
        
        im = ax.imshow(u_matrix, cmap='viridis')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Avg. distance to neighbors', fontsize=11)
        
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{u_matrix[i, j]:.2f}', ha='center', va='center', 
                       color='white', fontsize=14, fontweight='bold')

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        add_title_box(ax, 'U-Matrix: Neuron Distance Map', 
                     'Darker regions = cluster boundaries')
        save_fig('04_iris_umatrix', fig)

        print("\n" + "=" * 60)
        print("  Iris SOM Complete!")
        print("=" * 60)
        print("\nKey Observations:")
        print("   * Setosa forms a distinct cluster (easily separable)")
        print("   * Versicolor/Virginica overlap (similar features)")
        print("   * SOM preserves topology: similar -> nearby neurons")
        print("   * This is dimensionality reduction, NOT classification")

if __name__ == '__main__':
    run()
