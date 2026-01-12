"""
Helper functions for lab12 project
Plotting and data generation utilities with FANCY styling
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
import os

# ============================================
# FANCY PLOT STYLING
# ============================================
plt.style.use('seaborn-v0_8-whitegrid')

# Custom colors - vibrant and modern
COLORS = {
    'primary': '#6366F1',    # indigo
    'secondary': '#EC4899',  # pink
    'accent': '#14B8A6',     # teal
    'warning': '#F59E0B',    # amber
    'success': '#22C55E',    # green
    'dark': '#1E293B',       # slate
    'light': '#F8FAFC',      # light gray
}

PALETTE = ['#6366F1', '#EC4899', '#14B8A6', '#F59E0B', '#22C55E', 
           '#EF4444', '#8B5CF6', '#06B6D4', '#84CC16', '#F97316']

def setup_fancy_style():
    """Apply fancy matplotlib styling"""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#FAFAFA',
        'axes.edgecolor': '#E2E8F0',
        'axes.labelcolor': '#334155',
        'axes.titlecolor': '#1E293B',
        'axes.titleweight': 'bold',
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'axes.grid': True,
        'grid.color': '#E2E8F0',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        'xtick.color': '#64748B',
        'ytick.color': '#64748B',
        'text.color': '#334155',
        'font.family': 'sans-serif',
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
    })

# Apply on import
setup_fancy_style()

# Directories
BASE_DIR = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(name, fig=None):
    """Save figure to figures folder"""
    if fig is None:
        fig = plt.gcf()
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  Saved: {name}.png')
    return path

def redirect_print(filename):
    """Context manager to redirect prints to file"""
    class OutputRedirector:
        def __init__(self, fname):
            self.filepath = os.path.join(OUTPUT_DIR, fname)
            self.file = None
            self.stdout = None
            
        def __enter__(self):
            import sys
            self.stdout = sys.stdout
            self.file = open(self.filepath, 'w')
            
            class Tee:
                def __init__(self, stdout, file):
                    self.stdout = stdout
                    self.file = file
                def write(self, data):
                    self.stdout.write(data)
                    self.file.write(data)
                def flush(self):
                    self.stdout.flush()
                    self.file.flush()
            
            sys.stdout = Tee(self.stdout, self.file)
            return self
            
        def __exit__(self, *args):
            import sys
            sys.stdout = self.stdout
            self.file.close()
            print(f'  Output saved: {os.path.basename(self.filepath)}')
    
    return OutputRedirector(filename)

def generate_synthetic_data():
    """
    Generate digit-like synthetic data:
    - Round shapes (like 0, 6, 8, 9)
    - Linear shapes (like 1, 7)
    - Curves (like 2, 3, 5)
    """
    np.random.seed(42)
    
    # Round digits - compact blob
    round_data, _ = make_blobs(n_samples=150, centers=[[0, 0]], cluster_std=0.6)
    round_labels = np.zeros(150)
    
    # Linear digits - elongated vertical
    linear_data = np.random.randn(100, 2) * [0.2, 1.5] + [3, 0]
    linear_labels = np.ones(100)
    
    # Curved digits - moons
    curved_data, _ = make_moons(n_samples=150, noise=0.15)
    curved_data = curved_data * 1.5 + [-1, 3]
    curved_labels = np.full(150, 2)
    
    # Combine all
    data = np.vstack([round_data, linear_data, curved_data])
    labels = np.concatenate([round_labels, linear_labels, curved_labels])
    
    return data, labels

def fancy_scatter(ax, x, y, c=None, cmap='viridis', **kwargs):
    """Fancy scatter plot with shadow effect"""
    default = {'alpha': 0.7, 's': 40, 'edgecolors': 'white', 'linewidths': 0.5}
    default.update(kwargs)
    return ax.scatter(x, y, c=c, cmap=cmap, **default)

def add_title_box(ax, title, subtitle=None):
    """Add a styled title with optional subtitle"""
    ax.set_title(title, fontsize=14, fontweight='bold', color=COLORS['dark'], pad=32)
    if subtitle:
        ax.text(0.5, 1.04, subtitle, transform=ax.transAxes, 
                fontsize=10, color='#64748B', ha='center', style='italic')

def create_comparison_figure(ncols=2, nrows=1, figsize=None):
    """Create a nicely spaced comparison figure"""
    if figsize is None:
        figsize = (7*ncols, 5*nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor('white')
    return fig, axes
