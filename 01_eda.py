"""
Task 1 - Exploratory Data Analysis
Visualize and understand the datasets before applying algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
from utils import (generate_synthetic_data, save_fig, redirect_print, 
                   PALETTE, COLORS, fancy_scatter, add_title_box, create_comparison_figure)

def run():
    with redirect_print('01_eda_output.txt'):
        print("=" * 60)
        print("  TASK 1: Exploratory Data Analysis")
        print("=" * 60)

        # 1. Load synthetic data
        print("\n[1] Loading synthetic digit-like data...")
        synthetic_data, synthetic_labels = generate_synthetic_data()
        print(f"   Shape: {synthetic_data.shape}")
        print(f"   Labels: 0=round (0,6,8,9), 1=linear (1,7), 2=curved (2,3,5)")

        # 2. Visualize synthetic data - FANCY VERSION
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        
        label_names = ['Round digits (0,6,8,9)', 'Linear digits (1,7)', 'Curved digits (2,3,5)']
        colors = [PALETTE[0], PALETTE[1], PALETTE[2]]
        
        for i in range(3):
            mask = synthetic_labels == i
            ax.scatter(synthetic_data[mask, 0], synthetic_data[mask, 1], 
                      c=colors[i], label=label_names[i], alpha=0.7, s=60,
                      edgecolors='white', linewidths=0.8)
        
        add_title_box(ax, 'Synthetic Digit-like Data', 
                     'Simulating visual characteristics of handwritten digits')
        ax.set_xlabel('Feature 1 (Horizontal extent)')
        ax.set_ylabel('Feature 2 (Vertical extent)')
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='none')
        save_fig('01_synthetic_data', fig)

        # 3. Load Iris (mandatory)
        print("\n[2] Loading Iris dataset (mandatory)...")
        iris = load_iris()
        print(f"   Shape: {iris.data.shape}")
        print(f"   Features: {iris.feature_names}")
        print(f"   Classes: {list(iris.target_names)}")

        # Pairplot for Iris - FANCY VERSION
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.patch.set_facecolor('white')
        axes = axes.flatten()
        
        pair_idx = 0
        species_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        
        for i in range(4):
            for j in range(i+1, 4):
                if pair_idx < 6:
                    ax = axes[pair_idx]
                    for sp_idx, (sp_name, color) in enumerate(zip(iris.target_names, species_colors)):
                        mask = iris.target == sp_idx
                        ax.scatter(iris.data[mask, i], iris.data[mask, j], 
                                  c=color, label=sp_name if pair_idx == 0 else '', 
                                  alpha=0.7, s=50, edgecolors='white', linewidths=0.5)
                    ax.set_xlabel(iris.feature_names[i].replace(' (cm)', ''))
                    ax.set_ylabel(iris.feature_names[j].replace(' (cm)', ''))
                    pair_idx += 1
        
        fig.legend(iris.target_names, loc='upper center', ncol=3, 
                  framealpha=0.9, edgecolor='none', bbox_to_anchor=(0.5, 0.98))
        fig.suptitle('Iris Dataset - Pairwise Feature Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_fig('01_iris_pairplot', fig)

        # 4. Load digits dataset
        print("\n[3] Loading sklearn digits dataset...")
        digits = load_digits()
        print(f"   Shape: {digits.data.shape} (8x8 images = 64 features)")
        print(f"   Samples per digit: {list(np.bincount(digits.target))}")

        # Show example digits - FANCY VERSION
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        fig.patch.set_facecolor('white')
        
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(digits.images[i], cmap='Blues', interpolation='nearest')
            ax.set_title(f'Label: {digits.target[i]}', fontsize=11, fontweight='bold')
            ax.axis('off')
        
        fig.suptitle('Example Handwritten Digits (8x8 pixels)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_fig('01_digits_examples', fig)

        # 5. PCA projection of digits - FANCY VERSION
        print("\n[4] PCA projection of digits...")
        pca = PCA(n_components=2)
        digits_2d = pca.fit_transform(digits.data)
        print(f"   Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('white')
        
        scatter = ax.scatter(digits_2d[:, 0], digits_2d[:, 1], 
                           c=digits.target, cmap='tab10', alpha=0.7, s=25,
                           edgecolors='white', linewidths=0.3)
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
        cbar.set_label('Digit', fontsize=11)
        
        add_title_box(ax, 'Digits Dataset - PCA Projection', 
                     f'2D projection capturing {pca.explained_variance_ratio_.sum():.1%} of variance')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        save_fig('01_digits_pca', fig)

        # 6. Analyze confusion pairs - FANCY VERSION
        print("\n[5] Analyzing digit confusion pairs...")
        print("   Expected overlaps based on visual similarity:")
        print("   * 4 vs 9 (open vs closed top loop)")
        print("   * 3 vs 8 (number of enclosed loops)")
        print("   * 1 vs 7 (vertical stroke orientation)")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('white')
        
        pairs = [(4, 9, 'Similar strokes'), (3, 8, 'Loop structure'), (1, 7, 'Vertical shape')]
        
        for ax, (d1, d2, reason) in zip(axes, pairs):
            mask1 = digits.target == d1
            mask2 = digits.target == d2
            
            ax.scatter(digits_2d[mask1, 0], digits_2d[mask1, 1], 
                      c=COLORS['primary'], label=f'Digit {d1}', alpha=0.7, s=40,
                      edgecolors='white', linewidths=0.5)
            ax.scatter(digits_2d[mask2, 0], digits_2d[mask2, 1], 
                      c=COLORS['secondary'], label=f'Digit {d2}', alpha=0.7, s=40,
                      edgecolors='white', linewidths=0.5)
            
            add_title_box(ax, f'{d1} vs {d2}', reason)
            ax.legend(framealpha=0.9, edgecolor='none')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
        
        fig.suptitle('Digit Confusion Pairs in PCA Space', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_fig('01_confusion_pairs', fig)

        print("\n" + "=" * 60)
        print("  EDA Complete!")
        print("=" * 60)

if __name__ == '__main__':
    run()
