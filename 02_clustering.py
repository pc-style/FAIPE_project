"""
Task 2 - Classical Clustering Methods
K-Means, Hierarchical, DBSCAN comparison
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from utils import (generate_synthetic_data, save_fig, redirect_print,
                   PALETTE, COLORS, add_title_box)

def run():
    with redirect_print('02_clustering_output.txt'):
        print("=" * 60)
        print("  TASK 2: Classical Clustering Methods")
        print("=" * 60)

        # 1. Load data
        print("\n[1] Loading datasets...")
        synthetic_data, synthetic_labels = generate_synthetic_data()
        digits = load_digits()

        pca = PCA(n_components=2)
        digits_2d = pca.fit_transform(digits.data)

        # ============================================
        # K-MEANS
        # ============================================
        print("\n" + "-" * 50)
        print("  K-MEANS CLUSTERING")
        print("-" * 50)

        print("\n[2] Running elbow method + silhouette analysis...")
        k_range = range(2, 15)
        inertias = []
        silhouettes = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(digits.data)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(digits.data, kmeans.labels_))

        # Best k by silhouette
        best_k = list(k_range)[np.argmax(silhouettes)]
        print(f"   Best k by silhouette: {best_k} (score: {max(silhouettes):.3f})")

        # Plot - FANCY VERSION
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('white')

        ax1.plot(k_range, inertias, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)
        ax1.axvline(x=10, color=COLORS['secondary'], linestyle='--', linewidth=2, label='k=10 (actual digits)')
        ax1.fill_between(k_range, inertias, alpha=0.1, color=COLORS['primary'])
        add_title_box(ax1, 'Elbow Method', 'Lower inertia = tighter clusters')
        ax1.set_xlabel('Number of clusters (k)')
        ax1.set_ylabel('Inertia (within-cluster sum of squares)')
        ax1.legend(framealpha=0.9)

        ax2.plot(k_range, silhouettes, 'o-', color=COLORS['accent'], linewidth=2, markersize=8)
        ax2.axvline(x=10, color=COLORS['secondary'], linestyle='--', linewidth=2, label='k=10 (actual digits)')
        ax2.axvline(x=best_k, color=COLORS['warning'], linestyle=':', linewidth=2, label=f'Best k={best_k}')
        ax2.fill_between(k_range, silhouettes, alpha=0.1, color=COLORS['accent'])
        add_title_box(ax2, 'Silhouette Score', 'Higher = better cluster separation')
        ax2.set_xlabel('Number of clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.legend(framealpha=0.9)

        plt.tight_layout()
        save_fig('02_kmeans_elbow_silhouette', fig)

        # K-Means with k=10
        print("\n[3] K-Means with k=10 on digits...")
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(digits.data)

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('white')
        scatter = ax.scatter(digits_2d[:, 0], digits_2d[:, 1], 
                           c=kmeans_labels, cmap='tab10', alpha=0.7, s=25,
                           edgecolors='white', linewidths=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        add_title_box(ax, 'K-Means Clustering (k=10)', 
                     f'Silhouette: {silhouette_score(digits.data, kmeans_labels):.3f}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        save_fig('02_kmeans_digits', fig)

        # ============================================
        # HIERARCHICAL CLUSTERING
        # ============================================
        print("\n" + "-" * 50)
        print("  HIERARCHICAL CLUSTERING")
        print("-" * 50)

        print("\n[4] Building dendrogram (200 sample subset)...")
        np.random.seed(42)
        subset_idx = np.random.choice(len(digits.data), 200, replace=False)
        subset_data = digits.data[subset_idx]

        linkage_matrix = linkage(subset_data, method='ward')

        fig, ax = plt.subplots(figsize=(16, 8))
        fig.patch.set_facecolor('white')
        dendrogram(linkage_matrix, truncate_mode='lastp', p=30, 
                  ax=ax, leaf_rotation=90, leaf_font_size=9,
                  color_threshold=0.7*max(linkage_matrix[:,2]))
        add_title_box(ax, 'Hierarchical Clustering Dendrogram (Ward linkage)', 
                     'Shows cluster merge sequence - cut horizontally to get k clusters')
        ax.set_xlabel('Sample index or cluster size')
        ax.set_ylabel('Distance (Ward)')
        save_fig('02_hierarchical_dendrogram', fig)

        print("\n[5] Hierarchical clustering (10 clusters)...")
        hier = AgglomerativeClustering(n_clusters=10)
        hier_labels = hier.fit_predict(digits.data)
        hier_sil = silhouette_score(digits.data, hier_labels)
        print(f"   Silhouette score: {hier_sil:.3f}")

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('white')
        scatter = ax.scatter(digits_2d[:, 0], digits_2d[:, 1], 
                           c=hier_labels, cmap='tab10', alpha=0.7, s=25,
                           edgecolors='white', linewidths=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        add_title_box(ax, 'Hierarchical Clustering (10 clusters)', 
                     f'Silhouette: {hier_sil:.3f}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        save_fig('02_hierarchical_digits', fig)

        # ============================================
        # DBSCAN
        # ============================================
        print("\n" + "-" * 50)
        print("  DBSCAN CLUSTERING")
        print("-" * 50)

        print("\n[6] Testing DBSCAN parameters...")
        eps_values = [20, 30, 40, 50]
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.patch.set_facecolor('white')

        for ax, eps in zip(axes.flatten(), eps_values):
            dbscan = DBSCAN(eps=eps, min_samples=5)
            db_labels = dbscan.fit_predict(digits.data)
            n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
            n_noise = list(db_labels).count(-1)
            
            scatter = ax.scatter(digits_2d[:, 0], digits_2d[:, 1], 
                               c=db_labels, cmap='tab20', alpha=0.7, s=20,
                               edgecolors='white', linewidths=0.2)
            add_title_box(ax, f'eps={eps}', f'{n_clusters} clusters, {n_noise} noise')
            
            print(f"   eps={eps}: {n_clusters} clusters, {n_noise} noise points")

        fig.suptitle('DBSCAN Parameter Comparison', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        save_fig('02_dbscan_comparison', fig)

        # ============================================
        # COMPARISON ON SYNTHETIC DATA
        # ============================================
        print("\n" + "-" * 50)
        print("  COMPARISON ON SYNTHETIC DATA")
        print("-" * 50)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.patch.set_facecolor('white')

        configs = [
            ('Ground Truth', synthetic_labels, 'viridis'),
            ('K-Means (k=3)', KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(synthetic_data), 'tab10'),
            ('Hierarchical', AgglomerativeClustering(n_clusters=3).fit_predict(synthetic_data), 'tab10'),
            ('DBSCAN', DBSCAN(eps=0.5, min_samples=5).fit_predict(synthetic_data), 'tab10'),
        ]

        for ax, (title, labels, cmap) in zip(axes.flatten(), configs):
            ax.scatter(synthetic_data[:, 0], synthetic_data[:, 1], 
                      c=labels, cmap=cmap, alpha=0.7, s=50,
                      edgecolors='white', linewidths=0.5)
            add_title_box(ax, title)

        fig.suptitle('Clustering Comparison on Synthetic Data', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        save_fig('02_comparison_synthetic', fig)

        print("\n" + "=" * 60)
        print("  Clustering Complete!")
        print("=" * 60)
        print("\nKey Observations:")
        print("   * K-Means: Works well for spherical clusters, forced fixed k")
        print("   * Hierarchical: Reveals structure via dendrogram, no k needed upfront")
        print("   * DBSCAN: Finds dense regions, identifies noise/outliers")

if __name__ == '__main__':
    run()
