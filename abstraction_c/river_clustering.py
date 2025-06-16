import numpy as np
from sklearn.cluster import MiniBatchKMeans
import time

def fast_kmeans_clustering(input_file='abstraction_c/data/equity.csv', output_file='data/flop_kmeans.csv', 
                          n_clusters=200, memory_limit_gb=8):
    """
    Fast k-means clustering using numpy for better performance.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (rivers_kmeans.csv)
        n_clusters: Number of clusters (default: 200)
        memory_limit_gb: RAM limit in GB (default: 8)
    """
    
    print(f"Loading data from {input_file}...")
    start_time = time.time()
    
    # Load data as numpy array (more efficient than pandas for single column)
    # Using int16 since values are 0-1980 (saves memory)
    data = np.loadtxt(input_file, dtype=np.int16)
    
    print(f"Loaded {len(data):,} values in {time.time() - start_time:.1f} seconds")
    print(f"Memory usage: {data.nbytes / 1e9:.2f} GB")
    
    # Reshape for sklearn
    X = data.reshape(-1, 1)
    
    # Initialize and fit MiniBatchKMeans
    print(f"\nClustering into {n_clusters} clusters...")
    start_time = time.time()
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=50000,
        max_iter=100,
        random_state=42,
        n_init=3,
        compute_labels=False,  # Don't compute labels during fit
        verbose=1
    )
    
    # Fit the model
    kmeans.fit(X)
    
    print(f"Training completed in {time.time() - start_time:.1f} seconds")
    
    # Predict clusters
    print("\nPredicting clusters...")
    start_time = time.time()
    
    # Process in batches if memory is limited
    batch_size = 10000000  # 10M rows at a time
    n_samples = len(X)
    labels = np.zeros(n_samples, dtype=np.uint8)  # uint8 sufficient for 200 clusters
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        labels[i:end_idx] = kmeans.predict(X[i:end_idx])
        
        if i > 0 and i % (batch_size * 5) == 0:
            print(f"Processed {i:,} / {n_samples:,} rows ({i/n_samples*100:.1f}%)")
    
    print(f"Prediction completed in {time.time() - start_time:.1f} seconds")
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    np.savetxt(output_file, labels, fmt='%d')
    
    # Save cluster centers
    centers_file = 'data/cluster_centers.csv'
    np.savetxt(centers_file, kmeans.cluster_centers_, delimiter=',', fmt='%.6f')
    
    print("\nComplete!")
    print(f"Cluster assignments saved to: {output_file}")
    print(f"Cluster centers saved to: {centers_file}")
    
    # Print cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster sizes (min: {counts.min():,}, max: {counts.max():,}, "
          f"mean: {counts.mean():,.0f})")
    
    return kmeans, labels

def memory_efficient_version(input_file='data/equity.csv', output_file='data/rivers_kmeans.csv',
                           n_clusters=200):
    """
    Ultra memory-efficient version that never loads full dataset.
    Slower but works with very limited RAM.
    """
    
    print("Ultra memory-efficient mode...")
    
    # First, sample data for training
    print("Phase 1: Sampling data for training...")
    n_samples = 5000000  # Use 5M samples for training
    
    # Count total lines
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    # Random sampling
    skip_prob = max(1, total_lines // n_samples)
    samples = []
    
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i % skip_prob == 0:
                samples.append(int(line.strip()))
                if len(samples) >= n_samples:
                    break
    
    # Train k-means
    print(f"Training on {len(samples):,} samples...")
    X_train = np.array(samples, dtype=np.int16).reshape(-1, 1)
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, 
                            random_state=42, verbose=1)
    kmeans.fit(X_train)
    
    # Process file line by line and write results
    print("\nPhase 2: Processing full file...")
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        batch = []
        for i, line in enumerate(f_in):
            batch.append(int(line.strip()))
            
            # Process in batches
            if len(batch) == 100000:
                X_batch = np.array(batch, dtype=np.int16).reshape(-1, 1)
                labels = kmeans.predict(X_batch)
                for label in labels:
                    f_out.write(f"{label}\n")
                batch = []
                
                if i % 1000000 == 0:
                    print(f"Processed {i:,} lines...")
        
        # Process remaining
        if batch:
            X_batch = np.array(batch, dtype=np.int16).reshape(-1, 1)
            labels = kmeans.predict(X_batch)
            for label in labels:
                f_out.write(f"{label}\n")
    
    print("Complete!")

# Run the appropriate version based on available memory
if __name__ == "__main__":
    import psutil
    import os
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Check available memory
    available_memory_gb = psutil.virtual_memory().available / 1e9
    print(f"Available memory: {available_memory_gb:.1f} GB")
    
    # 136M rows * 2 bytes (int16) = ~272 MB, but sklearn needs more for processing
    required_memory_gb = 2.0  # Conservative estimate
    
    if available_memory_gb > required_memory_gb:
        # Use fast version if enough memory
        print(f"Using fast version (requires ~{required_memory_gb} GB)")
        fast_kmeans_clustering()
    else:
        # Use memory-efficient version
        print(f"Using memory-efficient version (low RAM detected)")
        memory_efficient_version()