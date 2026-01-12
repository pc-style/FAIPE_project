"""
Self-Organizing Map (SOM) implementation
Reusable module for lab12 project
"""
import numpy as np

class SOM:
    def __init__(self, grid_size, input_dim, sigma=1.0, lr=0.5, neighborhood='gaussian'):
        """
        grid_size: tuple (rows, cols) for 2D, or int for 1D
        input_dim: number of features in input data
        sigma: initial neighborhood radius
        lr: initial learning rate
        neighborhood: 'gaussian' or 'mexican_hat'
        """
        if isinstance(grid_size, int):
            self.grid_size = (grid_size,)
            self.is_1d = True
        else:
            self.grid_size = grid_size
            self.is_1d = False
        
        self.input_dim = input_dim
        self.sigma_init = sigma
        self.lr_init = lr
        self.neighborhood_type = neighborhood
        
        # Initialize weights randomly
        self.weights = np.random.randn(*self.grid_size, input_dim) * 0.1
        
        # Create coordinate grid for distance calculations
        if self.is_1d:
            self.coords = np.arange(self.grid_size[0]).reshape(-1, 1)
        else:
            rows, cols = np.meshgrid(range(self.grid_size[0]), range(self.grid_size[1]), indexing='ij')
            self.coords = np.stack([rows, cols], axis=-1)
    
    def find_bmu(self, x):
        """Find Best Matching Unit for input x"""
        diff = self.weights - x
        distances = np.sum(diff ** 2, axis=-1)
        bmu_idx = np.unravel_index(np.argmin(distances), self.grid_size)
        return bmu_idx
    
    def neighborhood_function(self, bmu_idx, sigma):
        """Calculate neighborhood influence for all neurons"""
        if self.is_1d:
            bmu_coord = np.array([bmu_idx[0]])
            dist = np.abs(self.coords - bmu_coord).squeeze()  # Squeeze to 1D
        else:
            bmu_coord = np.array(bmu_idx)
            dist = np.sqrt(np.sum((self.coords - bmu_coord) ** 2, axis=-1))
        
        if self.neighborhood_type == 'gaussian':
            return np.exp(-dist ** 2 / (2 * sigma ** 2))
        elif self.neighborhood_type == 'mexican_hat':
            # Mexican hat: (1 - (d/sigma)^2) * exp(-d^2 / (2*sigma^2))
            normalized_dist = dist / sigma
            return (1 - normalized_dist ** 2) * np.exp(-dist ** 2 / (2 * sigma ** 2))
        else:
            raise ValueError(f"Unknown neighborhood: {self.neighborhood_type}")
    
    def train(self, data, epochs=100, verbose=True):
        """Train the SOM"""
        n_samples = len(data)
        history = []
        
        for epoch in range(epochs):
            # Decay learning rate and sigma
            progress = epoch / epochs
            lr = self.lr_init * (1 - progress)
            sigma = self.sigma_init * (1 - 0.9 * progress)  # Keep minimum sigma
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x = data[idx]
                bmu_idx = self.find_bmu(x)
                h = self.neighborhood_function(bmu_idx, sigma)
                
                # Update weights
                if self.is_1d:
                    self.weights += lr * h[:, np.newaxis] * (x - self.weights)
                else:
                    self.weights += lr * h[:, :, np.newaxis] * (x - self.weights)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, lr={lr:.4f}, sigma={sigma:.4f}")
            
            history.append({'epoch': epoch, 'lr': lr, 'sigma': sigma})
        
        return history
    
    def predict(self, data):
        """Assign each sample to its BMU"""
        labels = []
        for x in data:
            bmu = self.find_bmu(x)
            if self.is_1d:
                labels.append(bmu[0])
            else:
                labels.append(bmu[0] * self.grid_size[1] + bmu[1])
        return np.array(labels)
    
    def topographic_error(self, data):
        """Calculate topographic error (fraction of samples where BMU and 2nd BMU aren't neighbors)"""
        errors = 0
        for x in data:
            diff = self.weights - x
            distances = np.sum(diff ** 2, axis=-1).flatten()
            sorted_idx = np.argsort(distances)
            
            bmu = sorted_idx[0]
            second_bmu = sorted_idx[1]
            
            if self.is_1d:
                if abs(bmu - second_bmu) > 1:
                    errors += 1
            else:
                bmu_coord = np.array([bmu // self.grid_size[1], bmu % self.grid_size[1]])
                second_coord = np.array([second_bmu // self.grid_size[1], second_bmu % self.grid_size[1]])
                if np.max(np.abs(bmu_coord - second_coord)) > 1:
                    errors += 1
        
        return errors / len(data)
    
    def quantization_error(self, data):
        """Calculate average distance to BMU"""
        total_error = 0
        for x in data:
            bmu = self.find_bmu(x)
            total_error += np.sqrt(np.sum((x - self.weights[bmu]) ** 2))
        return total_error / len(data)
