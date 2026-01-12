# Lab 12: Unsupervised Learning & SOM

Fast-paced exploration of unsupervised algorithms, dimensionality reduction, and topological maps. basically, clustering and some fancy neural maps (SOM).

## Quick Start
just run the main script to blast through all tasks at once:
```bash
python run_all.py
```
*Note: Make sure you've got the deps installed (numpy, sklearn, matplotlib, scipy).*

---

## Project Structure
here's the breakdown of what's happening:

- **01_eda.py**: quick look at the data before we break things.
- **02_clustering.py**: the classics—K-Means, Hierarchical, and DBSCAN. compared them on synthetic blobs and the digits dataset.
- **03_som_synthetic.py**: building and training a Self-Organizing Map on synthetic 2D data to see how it "unfolds".
- **04_som_iris.py**: mandatory Iris dataset task. SOM for classification/clustering.
- **05_som_large.py**: scaling things up. training larger networks and calculating Quantization Error and Topographic Error.
- **06_neighborhoods.py**: checking how different neighborhood functions (Gaussian vs Bubble) affect the map's final state.

## Components
- **som.py**: custom implementation of the SOM logic (weights, training loop, winners).
- **utils.py**: fancy plotting helpers and output redirects so we don't spam the terminal.
- **figures/**: all the pretty plots (elbow curves, dendrograms, U-matrices).
- **output/**: text logs with all the metrics and data stats.

---

## Key Takeaways
- **K-Means** is picky about k, but silhouettes help.
- **DBSCAN** is the goat for outliers/noise.
- **SOMs** are crazy for visualizing high-dim data in 2D space without losing the neighborhood vibe.

Ready to compile the final PDF once all runs are green!
